import streamlit as st
import pandas as pd
import sqlite3, json, os
from datetime import datetime, timezone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import requests
API_KEY = st.secrets.get( "API_SPORTS_KEY")
DB_PATH = os.getenv("SMARTBET_DB", "smartbet.db")
MODEL_PATH = "smartbet_model.joblib"

st.set_page_config(page_title="SmartBet AI", page_icon="📊", layout="wide")
if API_KEY:
    try:
        respuesta_api = requests.get(
            "https://v3.football.api-sports.io/status",
            headers={"x-apisports-key": API_KEY},
            timeout=15
        )
        datos_api = respuesta_api.json()
        if datos_api.get("errors"):
            st.sidebar.write("API-Sports: clave o acceso no autorizado")
        else:
            st.sidebar.write("API-Sports: conexión correcta")
    except Exception:
        st.sidebar.write("API-Sports: no se pudo conectar")
else:
    st.sidebar.write("API-Sports: falta configurar la clave")
st.title("📊 SmartBet AI")
st.caption("Analizador probabilístico multideporte · valor esperado · registro de apuestas")

def db():
    con=sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS bets (
      id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, sport TEXT, event TEXT,
      market TEXT, selection TEXT, odds REAL, probability REAL, stake REAL,
      ev_pct REAL, status TEXT DEFAULT 'Pendiente', result TEXT, profit REAL
    )""")
    con.commit()
    return con

def metrics(odds,p):
    implied=1/odds
    ev=p*odds-1
    fair=1/p if p>0 else None
    # Kelly fraction; clamp to avoid nonsensical output
    b=odds-1
    k=(b*p-(1-p))/b if b>0 else 0
    return implied,ev,fair,max(0,min(k,0.25))

def recommend(row, min_ev, min_prob):
    try:
        odds=float(row["odds"]); p=float(row["probability"])
        if odds<=1 or not 0<p<1: return False
        return (p*odds-1)>max(0.0,min_ev) and p>=min_prob
    except Exception: return False

with st.sidebar:
    st.header("Configuración")
    min_ev=st.slider("EV mínimo (%)",0,30,0)/100
    min_prob=st.slider("Probabilidad mínima (%)",1,99,95)/100
    st.caption("Perfil activo: máxima seguridad. Solo se muestran selecciones con probabilidad estimada ≥95% y EV estrictamente positivo.")
    bankroll=st.number_input("Bankroll (€)",min_value=0.0,value=100.0,step=10.0)
    kelly_fraction=st.select_slider("Kelly fraccional",options=[0.0,0.1,0.2,0.25,0.5,1.0],value=0.25)
    st.caption("Perfil SmartBet AI 2.0: probabilidad mínima 95% y EV estrictamente positivo. No se recomendará ninguna apuesta si no cumple ambos filtros. Las probabilidades deben validarse con datos reales.")

tab1,tab2,tab3,tab4=st.tabs(["🔎 Analizador","🧠 Entrenar modelo","🧾 Registro","📈 Rendimiento"])

with tab1:
    st.subheader("Analizar oportunidades")
    st.write("Sube un CSV con una fila por selección. Columnas obligatorias: `sport,event,market,selection,odds,probability`.")
    st.download_button("Descargar plantilla CSV", "sport,event,market,selection,odds,probability\nFútbol,Equipo A vs Equipo B,1X2,Equipo A,2.10,0.52\nTenis,Jugador A vs Jugador B,Ganador,Jugador A,1.80,0.60\n", "plantilla_apuestas.csv", "text/csv")
    file=st.file_uploader("CSV de cuotas y probabilidades",type=["csv"],key="analysis")
    if file:
        try:
            df=pd.read_csv(file)
            required=["sport","event","market","selection","odds","probability"]
            missing=[c for c in required if c not in df.columns]
            if missing: st.error("Faltan columnas: "+", ".join(missing))
            else:
                df["odds"]=pd.to_numeric(df["odds"],errors="coerce")
                df["probability"]=pd.to_numeric(df["probability"],errors="coerce")
                df=df.dropna(subset=["odds","probability"])
                df["implied_probability"]=1/df["odds"]
                df["ev_pct"]=(df["probability"]*df["odds"]-1)*100
                df["cuota_justa"]=1/df["probability"]
                df["recomendacion"]=df.apply(lambda r: recommend({"odds":r.odds,"probability":r.probability},min_ev,min_prob),axis=1)
                df["stake_sugerido"]=df.apply(lambda r: bankroll*metrics(r.odds,r.probability)[3]*kelly_fraction,axis=1)
                df["stake_sugerido"]=df["stake_sugerido"].clip(upper=bankroll*0.02)
                st.dataframe(df.sort_values("ev_pct",ascending=False),use_container_width=True)
                picks=df[df.recomendacion]
                st.success(f"{len(picks)} oportunidades cumplen tus filtros de máxima seguridad.") if len(picks) else st.warning("No hay apuestas que cumplan el perfil 95% + EV positivo. SmartBet AI no recomienda apostar hoy con estos datos.")
                if len(picks):
                    st.subheader("Mejores oportunidades según el modelo/probabilidad introducida")
                    st.dataframe(picks[["sport","event","market","selection","odds","probability","ev_pct","cuota_justa","stake_sugerido"]],use_container_width=True)
                    chosen=st.selectbox("Selecciona una apuesta para registrar",picks.index.tolist(),format_func=lambda i:f"{picks.loc[i,'sport']} | {picks.loc[i,'event']} | {picks.loc[i,'selection']}")
                    stake=st.number_input("Importe real (€)",min_value=0.0,value=float(picks.loc[chosen,"stake_sugerido"]),step=1.0)
                    if st.button("Guardar apuesta"):
                        r=picks.loc[chosen]
                        con=db()
                        con.execute("INSERT INTO bets(created_at,sport,event,market,selection,odds,probability,stake,ev_pct) VALUES(?,?,?,?,?,?,?,?,?)",
                         (datetime.now(timezone.utc).isoformat(),str(r.sport),str(r.event),str(r.market),str(r.selection),float(r.odds),float(r.probability),float(stake),float(r.ev_pct)))
                        con.commit(); con.close()
                        st.success("Apuesta guardada en el registro.")
        except Exception as e: st.error(f"No se pudo leer el CSV: {e}")
    st.info("Perfil conservador activo: mínimo 95% de probabilidad estimada y EV > 0. Las probabilidades de la plantilla son ilustrativas. No se conectan automáticamente a casas de apuestas ni representan predicciones verificadas.")

with tab2:
    st.subheader("Entrenamiento supervisado")
    st.write("Carga un CSV histórico: una fila por evento/selección, columnas numéricas de características y una columna objetivo binaria (`target`: 1 = acierto, 0 = fallo). No incluyas información posterior al evento.")
    train_file=st.file_uploader("Datos históricos CSV",type=["csv"],key="training")
    if train_file:
        data=pd.read_csv(train_file)
        target=st.selectbox("Columna objetivo",data.columns.tolist(),index=(data.columns.tolist().index("target") if "target" in data.columns else len(data.columns)-1))
        features=st.multiselect("Variables predictoras (solo numéricas previas al evento)",[c for c in data.columns if c!=target and pd.api.types.is_numeric_dtype(data[c])])
        if st.button("Entrenar y evaluar",disabled=(not features)):
            try:
                work=data[features+[target]].copy().dropna(subset=[target])
                y=pd.to_numeric(work[target],errors="raise").astype(int)
                if y.nunique()!=2: raise ValueError("La variable objetivo debe tener exactamente dos clases (0 y 1).")
                X=work[features]
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
                model=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("clf",HistGradientBoostingClassifier(max_iter=150,learning_rate=0.06,max_leaf_nodes=15,l2_regularization=1.0,random_state=42))])
                model.fit(X_train,y_train)
                probs=model.predict_proba(X_test)[:,1]
                st.metric("Exactitud (umbral 50%)",f"{accuracy_score(y_test,(probs>=0.5).astype(int)):.1%}")
                st.metric("Brier score (menor es mejor)",f"{brier_score_loss(y_test,probs):.4f}")
                st.metric("Log loss (menor es mejor)",f"{log_loss(y_test,probs):.4f}")
                joblib.dump({"model":model,"features":features},MODEL_PATH)
                st.success(f"Modelo guardado en {MODEL_PATH}. Usa sus probabilidades como columna `probability` en el analizador.")
                st.warning("Evalúa también por temporadas/fechas (validación temporal) antes de confiar en el modelo. Esta evaluación aleatoria es solo una comprobación inicial.")
            except Exception as e: st.error(f"Error de entrenamiento: {e}")
    if os.path.exists(MODEL_PATH):
        obj=joblib.load(MODEL_PATH)
        st.write("Modelo disponible. Variables:",", ".join(obj["features"]))
        pred_file=st.file_uploader("CSV de eventos futuros con esas variables",type=["csv"],key="prediction")
        if pred_file:
            future=pd.read_csv(pred_file)
            missing=[f for f in obj["features"] if f not in future.columns]
            if missing: st.error("Faltan variables: "+", ".join(missing))
            else:
                future["probability"]=obj["model"].predict_proba(future[obj["features"]])[:,1]
                st.dataframe(future,use_container_width=True)
                st.download_button("Descargar probabilidades",future.to_csv(index=False),"predicciones_smartbet.csv","text/csv")

with tab3:
    st.subheader("Historial de apuestas")
    con=db()
    hist=pd.read_sql_query("SELECT * FROM bets ORDER BY id DESC",con)
    if len(hist):
        st.dataframe(hist,use_container_width=True)
        pending=hist[hist.status=="Pendiente"]
        if len(pending):
            ids=pending.id.tolist()
            bet_id=st.selectbox("Apuesta pendiente",ids)
            outcome=st.radio("Resultado",["Ganada","Perdida","Anulada"],horizontal=True)
            if st.button("Liquidar apuesta"):
                r=hist[hist.id==bet_id].iloc[0]
                profit=(r.stake*(r.odds-1) if outcome=="Ganada" else (-r.stake if outcome=="Perdida" else 0.0))
                con.execute("UPDATE bets SET status=?,result=?,profit=? WHERE id=?",(outcome,outcome,profit,int(bet_id)))
                con.commit(); st.success("Resultado actualizado. Recarga la pestaña para ver los cambios.")
        st.download_button("Exportar historial CSV",hist.to_csv(index=False),"historial_smartbet.csv","text/csv")
    else: st.info("Todavía no hay apuestas registradas.")
    con.close()

with tab4:
    st.subheader("Estadísticas de rendimiento")
    con=db(); settled=pd.read_sql_query("SELECT * FROM bets WHERE status!='Pendiente'",con); con.close()
    if len(settled):
        total=float(settled.stake.sum()); profit=float(settled.profit.sum())
        wins=int((settled.status=="Ganada").sum())
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Apuestas liquidadas",len(settled))
        c2.metric("Beneficio neto",f"{profit:+.2f} €")
        c3.metric("ROI",f"{(profit/total*100 if total else 0):.2f}%")
        c4.metric("Acierto",f"{wins/len(settled)*100:.1f}%")
        st.bar_chart(settled.groupby("status").size())
    else: st.info("Liquida apuestas registradas para empezar a medir ROI y acierto.")
