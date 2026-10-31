import os
import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd
import requests
import streamlit as st

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =====================================================
# CONFIGURACIÓN
# =====================================================

st.set_page_config(
    page_title="SmartBet AI",
    page_icon="📊",
    layout="wide"
)

API_KEY = st.secrets.get("API_SPORTS_KEY", "")
DB_PATH = os.getenv("SMARTBET_DB", "smartbet.db")
MODEL_PATH = "smartbet_model.joblib"

API_URL = "https://v3.football.api-sports.io"
API_HEADERS = {"x-apisports-key": API_KEY}


# =====================================================
# BASE DE DATOS
# =====================================================

def db():
    con = sqlite3.connect(DB_PATH)

    con.execute("""
        CREATE TABLE IF NOT EXISTS bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            sport TEXT,
            event TEXT,
            market TEXT,
            selection TEXT,
            odds REAL,
            probability REAL,
            stake REAL,
            ev_pct REAL,
            status TEXT DEFAULT 'Pendiente',
            result TEXT,
            profit REAL
        )
    """)

    con.commit()
    return con


# =====================================================
# FUNCIONES DE CÁLCULO
# =====================================================

def metrics(odds, probability):
    implied = 1 / odds
    ev = probability * odds - 1
    fair = 1 / probability if probability > 0 else None

    b = odds - 1
    kelly = (
        (b * probability - (1 - probability)) / b
        if b > 0 else 0
    )

    return implied, ev, fair, max(0, min(kelly, 0.25))


def recommend(row, min_ev, min_prob):
    try:
        odds = float(row["odds"])
        probability = float(row["probability"])

        if odds <= 1 or not 0 < probability < 1:
            return False

        ev = probability * odds - 1

        return ev > max(0.0, min_ev) and probability >= min_prob

    except Exception:
        return False


# =====================================================
# CONEXIÓN API
# =====================================================

def api_get(endpoint, params=None):
    if not API_KEY:
        raise ValueError("Falta configurar la clave API.")

    response = requests.get(
        f"{API_URL}/{endpoint}",
        headers=API_HEADERS,
        params=params,
        timeout=20
    )

    response.raise_for_status()
    data = response.json()

    if data.get("errors"):
        raise ValueError(str(data["errors"]))

    return data.get("response", [])


# =====================================================
# ESTADO DE LA API
# =====================================================

st.title("📊 SmartBet AI")
st.caption(
    "Analizador deportivo · Probabilidades · Valor esperado · Registro"
)

if API_KEY:
    try:
        api_get("status")
        st.sidebar.success("API-Sports: conexión correcta")
    except Exception as e:
        st.sidebar.error(f"API-Sports: {e}")
else:
    st.sidebar.warning("Falta configurar API_SPORTS_KEY en Secrets")


# =====================================================
# CONSULTA DE PARTIDOS
# =====================================================

st.header("⚽ Partidos")

if st.button("Comprobar acceso a partidos"):
    try:
        fecha = datetime.now().strftime("%Y-%m-%d")

        partidos = api_get(
            "fixtures",
            {"date": fecha}
        )

        st.session_state["fixture_ids"] = [
            p["fixture"]["id"] for p in partidos[:10]
        ]

        st.session_state["fixtures"] = partidos[:10]

        st.success(
            f"Consulta correcta. Partidos encontrados: {len(partidos)}"
        )

        if partidos:
            tabla = []

            for partido in partidos[:10]:
                tabla.append({
                    "ID": partido["fixture"]["id"],
                    "Partido": (
                        partido["teams"]["home"]["name"]
                        + " - "
                        + partido["teams"]["away"]["name"]
                    ),
                    "Competición": partido["league"]["name"],
                    "Fecha": partido["fixture"]["date"],
                    "Estado": partido["fixture"]["status"]["short"]
                })

            st.dataframe(
                pd.DataFrame(tabla),
                use_container_width=True
            )
        else:
            st.info("No se encontraron partidos para esta fecha.")

    except Exception as e:
        st.error(f"No se pudieron consultar los partidos: {e}")


# =====================================================
# CONSULTA DE ESTADÍSTICAS
# =====================================================

st.subheader("Estadísticas de partidos")

if st.button("Comprobar estadísticas"):
    fixture_ids = st.session_state.get("fixture_ids", [])

    if not fixture_ids:
        st.warning("Primero consulta los partidos.")
    else:
        for fixture_id in fixture_ids:
            st.markdown(f"**Partido ID: {fixture_id}**")

            try:
                estadisticas = api_get(
                    "fixtures/statistics",
                    {"fixture": fixture_id}
                )

                if estadisticas:
                    st.success("Estadísticas disponibles")

                    for equipo in estadisticas:
                        st.write(equipo)

                else:
                    st.warning(
                        "No hay estadísticas disponibles para este partido."
                    )

            except Exception as e:
                st.error(
                    f"Error al consultar el partido {fixture_id}: {e}"
                )


# =====================================================
# CONFIGURACIÓN DE APUESTAS
# =====================================================

with st.sidebar:
    st.header("Configuración")

    min_prob = st.slider(
        "Probabilidad mínima (%)",
        1, 99, 95
    ) / 100

    min_ev = st.slider(
        "EV mínimo (%)",
        0, 30, 0
    ) / 100

    bankroll = st.number_input(
        "Bankroll (€)",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    kelly_fraction = st.select_slider(
        "Kelly fraccional",
        options=[0.0, 0.1, 0.2, 0.25, 0.5, 1.0],
        value=0.25
    )

    st.caption(
        "Solo se muestran selecciones que cumplen los filtros "
        "de probabilidad y valor esperado."
    )


# =====================================================
# PESTAÑAS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Analizador",
    "🧠 Entrenar modelo",
    "🧾 Registro",
    "📈 Rendimiento"
])


# =====================================================
# PESTAÑA 1: ANALIZADOR
# =====================================================

with tab1:
    st.subheader("Analizar oportunidades")

    st.write(
        "Carga un CSV con estas columnas: "
        "sport,event,market,selection,odds,probability."
    )

    plantilla = (
        "sport,event,market,selection,odds,probability\n"
        "Fútbol,Equipo A vs Equipo B,1X2,Equipo A,2.10,0.52\n"
        "Tenis,Jugador A vs Jugador B,Ganador,Jugador A,1.80,0.60\n"
    )

    st.download_button(
        "Descargar plantilla CSV",
        plantilla,
        "plantilla_apuestas.csv",
        "text/csv"
    )

    file = st.file_uploader(
        "CSV de cuotas y probabilidades",
        type=["csv"],
        key="analysis"
    )

    if file:
        try:
            df = pd.read_csv(file)

            required = [
                "sport",
                "event",
                "market",
                "selection",
                "odds",
                "probability"
            ]

            missing = [
                column for column in required
                if column not in df.columns
            ]

            if missing:
                st.error(
                    "Faltan columnas: " + ", ".join(missing)
                )

            else:
                df["odds"] = pd.to_numeric(
                    df["odds"],
                    errors="coerce"
                )

                df["probability"] = pd.to_numeric(
                    df["probability"],
                    errors="coerce"
                )

                df = df.dropna(
                    subset=["odds", "probability"]
                )

                df = df[
                    (df["odds"] > 1)
                    & (df["probability"] > 0)
                    & (df["probability"] < 1)
                ].copy()

                df["implied_probability"] = 1 / df["odds"]

                df["ev_pct"] = (
                    df["probability"] * df["odds"] - 1
                ) * 100

                df["cuota_justa"] = 1 / df["probability"]

                df["recomendacion"] = df.apply(
                    lambda r: recommend(
                        {
                            "odds": r["odds"],
                            "probability": r["probability"]
                        },
                        min_ev,
                        min_prob
                    ),
                    axis=1
                )

                df["stake_sugerido"] = df.apply(
                    lambda r: (
                        bankroll
                        * metrics(
                            r["odds"],
                            r["probability"]
                        )[3]
                        * kelly_fraction
                    ),
                    axis=1
                )

                df["stake_sugerido"] = df[
                    "stake_sugerido"
                ].clip(upper=bankroll * 0.02)

                st.dataframe(
                    df.sort_values(
                        "ev_pct",
                        ascending=False
                    ),
                    use_container_width=True
                )

                picks = df[df["recomendacion"]]

                if len(picks) > 0:
                    st.success(
                        f"{len(picks)} oportunidades cumplen los filtros."
                    )

                    st.subheader("Oportunidades seleccionadas")

                    st.dataframe(
                        picks[
                            [
                                "sport",
                                "event",
                                "market",
                                "selection",
                                "odds",
                                "probability",
                                "ev_pct",
                                "cuota_justa",
                                "stake_sugerido"
                            ]
                        ],
                        use_container_width=True
                    )

                    chosen = st.selectbox(
                        "Selecciona una apuesta para registrar",
                        picks.index.tolist(),
                        format_func=lambda i: (
                            f"{picks.loc[i, 'sport']} | "
                            f"{picks.loc[i, 'event']} | "
                            f"{picks.loc[i, 'selection']}"
                        )
                    )

                    stake = st.number_input(
                        "Importe real (€)",
                        min_value=0.0,
                        value=float(
                            picks.loc[chosen, "stake_sugerido"]
                        ),
                        step=1.0
                    )

                    if st.button("Guardar apuesta"):
                        r = picks.loc[chosen]
                        con = db()

                        con.execute(
                            """
                            INSERT INTO bets (
                                created_at, sport, event, market,
                                selection, odds, probability,
                                stake, ev_pct
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                datetime.now(timezone.utc).isoformat(),
                                str(r["sport"]),
                                str(r["event"]),
                                str(r["market"]),
                                str(r["selection"]),
                                float(r["odds"]),
                                float(r["probability"]),
                                float(stake),
                                float(r["ev_pct"])
                            )
                        )

                        con.commit()
                        con.close()

                        st.success("Apuesta guardada en el registro.")

                else:
                    st.warning(
                        "No hay apuestas que cumplan el perfil "
                        "de probabilidad mínima y EV positivo."
                    )

        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")

    st.info(
        "Las probabilidades introducidas en el CSV no están "
        "verificadas automáticamente. Un 95% estimado no garantiza "
        "un 95% real de acierto."
    )


# =====================================================
# PESTAÑA 2: ENTRENAMIENTO
# =====================================================

with tab2:
    st.subheader("Entrenamiento supervisado")

    st.write(
        "Carga datos históricos con variables numéricas y una columna "
        "target: 1 significa acierto y 0 significa fallo."
    )

    train_file = st.file_uploader(
        "Datos históricos CSV",
        type=["csv"],
        key="training"
    )

    if train_file:
        try:
            data = pd.read_csv(train_file)

            target = st.selectbox(
                "Columna objetivo",
                data.columns.tolist(),
                index=(
                    data.columns.tolist().index("target")
                    if "target" in data.columns
                    else len(data.columns) - 1
                )
            )

            features = st.multiselect(
                "Variables predictoras numéricas",
                [
                    c for c in data.columns
                    if c != target
                    and pd.api.types.is_numeric_dtype(data[c])
                ]
            )

            if st.button(
                "Entrenar y evaluar",
                disabled=not features
            ):
                work = data[features + [target]].copy()
                work = work.dropna(subset=[target])

                y = pd.to_numeric(
                    work[target],
                    errors="raise"
                ).astype(int)

                if not y.isin([0, 1]).all():
                    raise ValueError(
                        "La columna objetivo solo puede contener 0 y 1."
                    )

                if y.nunique() != 2:
                    raise ValueError(
                        "Se necesitan ejemplos de las dos clases: 0 y 1."
                    )

                X = work[features]

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    random_state=42,
                    stratify=y
                )

                model = Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scale",
                        StandardScaler()
                    ),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            max_iter=150,
                            learning_rate=0.06,
                            max_leaf_nodes=15,
                            l2_regularization=1.0,
                            random_state=42
                        )
                    )
                ])

                model.fit(X_train, y_train)

                probs = model.predict_proba(X_test)[:, 1]

                st.metric(
                    "Exactitud",
                    f"{accuracy_score(y_test, (probs >= 0.5).astype(int)):.1%}"
                )

                st.metric(
                    "Brier score",
                    f"{brier_score_loss(y_test, probs):.4f}"
                )

                st.metric(
                    "Log loss",
                    f"{log_loss(y_test, probs):.4f}"
                )

                joblib.dump(
                    {
                        "model": model,
                        "features": features
                    },
                    MODEL_PATH
                )

                st.success("Modelo entrenado y guardado.")

                st.warning(
                    "Esta evaluación aleatoria es inicial. "
                    "Antes de usar el modelo hay que validarlo con "
                    "datos separados cronológicamente."
                )

        except Exception as e:
            st.error(f"Error de entrenamiento: {e}")

    if os.path.exists(MODEL_PATH):
        try:
            obj = joblib.load(MODEL_PATH)

            st.write(
                "Modelo disponible. Variables:",
                ", ".join(obj["features"])
            )

            pred_file = st.file_uploader(
                "CSV de eventos futuros",
                type=["csv"],
                key="prediction"
            )

            if pred_file:
                future = pd.read_csv(pred_file)

                missing = [
                    f for f in obj["features"]
                    if f not in future.columns
                ]

                if missing:
                    st.error(
                        "Faltan variables: " + ", ".join(missing)
                    )

                else:
                    future["probability"] = (
                        obj["model"].predict_proba(
                            future[obj["features"]]
                        )[:, 1]
                    )

                    st.dataframe(
                        future,
                        use_container_width=True
                    )

                    st.download_button(
                        "Descargar probabilidades",
                        future.to_csv(index=False),
                        "predicciones_smartbet.csv",
                        "text/csv"
                    )

        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")


# =====================================================
# PESTAÑA 3: REGISTRO
# =====================================================

with tab3:
    st.subheader("Historial de apuestas")

    con = db()

    hist = pd.read_sql_query(
        "SELECT * FROM bets ORDER BY id DESC",
        con
    )

    if len(hist) > 0:
        st.dataframe(
            hist,
            use_container_width=True
        )

        pending = hist[
            hist["status"] == "Pendiente"
        ]

        if len(pending) > 0:
            ids = pending["id"].tolist()

            bet_id = st.selectbox(
                "Apuesta pendiente",
                ids
            )

            outcome = st.radio(
                "Resultado",
                ["Ganada", "Perdida", "Anulada"],
                horizontal=True
            )

            if st.button("Liquidar apuesta"):
                r = hist[hist["id"] == bet_id].iloc[0]

      
