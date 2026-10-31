import streamlit as st
import requests
import pandas as pd
import json
from datetime import date

st.set_page_config(
    page_title="SmartBet AI",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 SMARTBET AI")
st.caption("Fútbol y fútbol australiano")

# =====================================
# CONFIGURACIÓN
# =====================================

try:
    API_KEY = st.secrets["API_SPORTS_KEY"]
except Exception:
    API_KEY = ""

FOOTBALL_URL = "https://v3.football.api-sports.io"
AFL_URL = "https://v1.afl.api-sports.io"

HEADERS = {"x-apisports-key": API_KEY}

# =====================================
# CONEXIÓN Y CACHÉ
# =====================================

def pedir_api(url, endpoint, parametros):
    if not API_KEY:
        raise Exception("No está configurada API_SPORTS_KEY.")

    respuesta = requests.get(
        f"{url}/{endpoint}",
        headers=HEADERS,
        params=parametros,
        timeout=25
    )

    if respuesta.status_code == 429:
        raise Exception(
            "Límite de consultas alcanzado. "
            "Espera al menos un minuto."
        )

    respuesta.raise_for_status()
    datos = respuesta.json()

    if datos.get("errors"):
        raise Exception(str(datos["errors"]))

    return datos.get("response", [])


@st.cache_data(ttl=60, show_spinner=False)
def consultar_futbol(fecha_texto):
    return pedir_api(
        FOOTBALL_URL,
        "fixtures",
        {
            "date": fecha_texto,
            "timezone": "Europe/Madrid"
        }
    )


@st.cache_data(ttl=60, show_spinner=False)
def consultar_afl(fecha_texto):
    return pedir_api(
        AFL_URL,
        "games",
        {
            "date": fecha_texto,
            "timezone": "Europe/Madrid"
        }
    )


@st.cache_data(ttl=60, show_spinner=False)
def consultar_prediccion(fixture_id):
    return pedir_api(
        FOOTBALL_URL,
        "predictions",
        {"fixture": fixture_id}
    )


# =====================================
# FUNCIONES DE FÚTBOL
# =====================================

def porcentaje(valor):
    if valor is None:
        return None
    try:
        return float(str(valor).replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def analizar_futbol(partido):
    predicciones = consultar_prediccion(
        partido["fixture"]["id"]
    )

    if not predicciones:
        return None

    pred = predicciones[0]
    equipos = partido["teams"]
    local = equipos["home"]["name"]
    visitante = equipos["away"]["name"]

    p = pred.get("predictions", {}).get("percent", {})

    opciones = {
        local: porcentaje(p.get("home")),
        "Empate": porcentaje(p.get("draw")),
        visitante: porcentaje(p.get("away"))
    }

    opciones = {
        nombre: valor
        for nombre, valor in opciones.items()
        if valor is not None
    }

    if not opciones:
        return None

    favorito = max(opciones, key=opciones.get)
    prob = opciones[favorito]

    return {
        "Partido": f"{local} - {visitante}",
        "Pronóstico": (
            "Empate" if favorito == "Empate"
            else f"Gana {favorito}"
        ),
        "Probabilidad": prob,
        "Liga": partido.get("league", {}).get("name", ""),
        "Consejo": pred.get("predictions", {}).get("advice", "")
    }


# =====================================
# FUNCIONES DE AFL
# =====================================

def buscar_valor(objeto, claves):
    if not isinstance(objeto, dict):
        return None

    for clave in claves:
        if clave in objeto:
            return objeto[clave]

    return None


def mostrar_partido_afl(partido):
    equipos = partido.get("teams", {})
    juego = partido.get("game", partido.get("fixture", {}))
    if not isinstance(juego, dict):
        juego = {}

    local = buscar_valor(
        equipos,
        ["home", "local"]
    )
    visitante = buscar_valor(
        equipos,
        ["away", "visitor", "visitors"]
    )

    def nombre_equipo(equipo):
        if isinstance(equipo, dict):
            return equipo.get("name", "No disponible")
        return str(equipo or "No disponible")

    estado = juego.get("status", {})
    if isinstance(estado, dict):
        estado = estado.get("long", estado.get("short", "No indicado"))

    return {
        "Partido": (
            f"{nombre_equipo(local)} - "
            f"{nombre_equipo(visitante)}"
        ),
        "Estado": estado,
        "Datos recibidos": json.dumps(
            partido,
            ensure_ascii=False
        )
    }


# =====================================
# MENÚ LATERAL
# =====================================

st.sidebar.header("Configuración")

fecha = st.sidebar.date_input(
    "Fecha de partidos",
    value=date.today()
)

fecha_texto = fecha.strftime("%Y-%m-%d")

st.sidebar.write("Conexión API:")

if API_KEY:
    st.sidebar.success("Clave configurada")
else:
    st.sidebar.error("Falta API_SPORTS_KEY")


# =====================================
# PESTAÑAS
# =====================================

tab_futbol, tab_afl, tab_registro = st.tabs(
    ["⚽ FÚTBOL", "🏉 AFL", "📊 RESULTADOS"]
)

# =====================================
# FÚTBOL
# =====================================

with tab_futbol:

    st.header("Pronósticos de fútbol")

    umbral = st.slider(
        "Probabilidad mínima (%)",
        50, 99, 95
    )

    cantidad = st.selectbox(
        "Número de partidos",
        [1, 2, 3, 4],
        index=2
    )

    if st.button("Buscar partidos de fútbol"):

        try:
            with st.spinner("Buscando partidos..."):
                partidos = consultar_futbol(fecha_texto)

            pendientes = [
                p for p in partidos
                if p.get("fixture", {})
                .get("status", {})
                .get("short") == "NS"
            ]

            pendientes = pendientes[:cantidad]

            if not pendientes:
                st.info("No hay partidos pendientes en esa fecha.")

            else:
                resultados = []
                barra = st.progress(0)

                for i, partido in enumerate(pendientes):
                    try:
                        resultado = analizar_futbol(partido)

                        if resultado:
                            resultados.append(resultado)

                    except Exception as error:
                        st.warning(str(error))
                        if "429" in str(error):
                            break

                    barra.progress((i + 1) / len(pendientes))

                st.session_state["futbol_resultados"] = resultados

                if resultados:
                    st.success("Pronósticos recibidos.")
                else:
                    st.warning("No se recibieron pronósticos.")

        except Exception as error:
            st.error(f"Error en fútbol: {error}")

    resultados = st.session_state.get("futbol_resultados", [])

    for r in resultados:
        st.subheader(r["Partido"])
        st.write(f"**Pronóstico:** {r['Pronóstico']}")
        st.write(f"**Probabilidad estimada:** {r['Probabilidad']}%")
        st.write(f"**Liga:** {r['Liga']}")

        if r["Consejo"]:
            st.write(f"Consejo del modelo: {r['Consejo']}")

        if r["Probabilidad"] >= umbral:
            st.warning(
                "Alcanza el filtro de probabilidad. "
                "Falta comprobar cuota y valor esperado."
            )
        else:
            st.info("NO APOSTAR: no alcanza el filtro.")

        st.divider()


# =====================================
# AFL
# =====================================

with tab_afl:

    st.header("Fútbol australiano (AFL)")

    st.write(
        "Consulta encuentros de AFL disponibles en API-Sports."
    )

    if st.button("Buscar partidos AFL"):

        try:
            with st.spinner("Consultando calendario AFL..."):
                partidos_afl = consultar_afl(fecha_texto)

            if not partidos_afl:
                st.info(
                    "No hay partidos AFL disponibles para esa fecha."
                )

            else:
                filas = [
                    mostrar_partido_afl(p)
                    for p in partidos_afl
                ]

                st.session_state["afl_resultados"] = filas
                st.success(
                    f"Se encontraron {len(filas)} encuentros."
                )

        except Exception as error:
            st.error(f"Error en AFL: {error}")

    filas = st.session_state.get("afl_resultados", [])

    if filas:
        for partido in filas:
            st.subheader(partido["Partido"])
            st.write(f"**Estado:** {partido['Estado']}")

            with st.expander("Ver datos recibidos de la API"):
                st.code(
                    partido["Datos recibidos"],
                    language="json"
                )

            st.divider()

        st.caption(
            "Esta sección muestra datos de partidos. "
            "Todavía no genera probabilidades de victoria para AFL."
        )


# =====================================
# RESULTADOS
# =====================================

with tab_registro:

    st.header("Resultados de consultas")

    futbol = st.session_state.get("futbol_resultados", [])
    afl = st.session_state.get("afl_resultados", [])

    st.metric("Pronósticos de fútbol", len(futbol))
    st.metric("Partidos AFL consultados", len(afl))

    if futbol:
        st.subheader("Fútbol")
        st.dataframe(
            pd.DataFrame(futbol),
            use_container_width=True,
            hide_index=True
        )

    if afl:
        st.subheader("AFL")
        st.dataframe(
            pd.DataFrame([
                {
                    "Partido": p["Partido"],
                    "Estado": p["Estado"]
                }
                for p in afl
            ]),
            use_container_width=True,
            hide_index=True
        )


# =====================================
# AVISO
# =====================================

st.divider()

st.caption(
    "SmartBet AI no garantiza resultados ni realiza apuestas. "
    "Las predicciones son estimaciones y requieren validación."
)
