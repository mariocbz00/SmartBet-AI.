
import streamlit as st
import requests
import pandas as pd
import json
from datetime import date

# --------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------

st.set_page_config(
    page_title="SmartBet AI",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ SmartBet AI")
st.caption("Pronósticos automáticos de fútbol")

API_URL = "https://v3.football.api-sports.io"

try:
    API_KEY = st.secrets["API_SPORTS_KEY"]
except Exception:
    API_KEY = ""

# --------------------------------------------------
# CONEXIÓN CON API-FOOTBALL
# --------------------------------------------------

def consultar_api(endpoint, parametros=None):
    if not API_KEY:
        raise Exception(
            "Falta la clave API_SPORTS_KEY en los Secrets de Streamlit."
        )

    url = f"{API_URL}/{endpoint}"

    respuesta = requests.get(
        url,
        headers={"x-apisports-key": API_KEY},
        params=parametros or {},
        timeout=25
    )

    if respuesta.status_code == 429:
        raise Exception(
            "Límite de consultas alcanzado (429). "
            "Espera antes de volver a consultar."
        )

    respuesta.raise_for_status()

    datos = respuesta.json()

    if datos.get("errors"):
        raise Exception(str(datos["errors"]))

    return datos.get("response", [])


# --------------------------------------------------
# FUNCIONES DE PRONÓSTICO
# --------------------------------------------------

def convertir_porcentaje(valor):
    if valor is None:
        return None

    try:
        return float(str(valor).replace("%", "").strip())
    except (ValueError, TypeError):
        return None


def obtener_pronostico(fixture_id):
    resultados = consultar_api(
        "predictions",
        {"fixture": fixture_id}
    )

    if not resultados:
        return None

    return resultados[0]


def extraer_porcentajes(prediccion):
    porcentajes = (
        prediccion.get("predictions", {})
        .get("percent", {})
    )

    return {
        "local": convertir_porcentaje(porcentajes.get("home")),
        "empate": convertir_porcentaje(porcentajes.get("draw")),
        "visitante": convertir_porcentaje(porcentajes.get("away"))
    }


def elegir_favorito(prediccion, equipo_local, equipo_visitante):
    porcentajes = extraer_porcentajes(prediccion)

    opciones = {
        equipo_local: porcentajes["local"],
        "Empate": porcentajes["empate"],
        equipo_visitante: porcentajes["visitante"]
    }

    disponibles = {
        equipo: porcentaje
        for equipo, porcentaje in opciones.items()
        if porcentaje is not None
    }

    if not disponibles:
        return None, None

    favorito = max(disponibles, key=disponibles.get)

    return favorito, disponibles[favorito]


# --------------------------------------------------
# PANEL LATERAL
# --------------------------------------------------

st.sidebar.header("Configuración")

fecha = st.sidebar.date_input(
    "Fecha de los partidos",
    value=date.today()
)

umbral = st.sidebar.slider(
    "Probabilidad mínima (%)",
    min_value=50,
    max_value=99,
    value=95
)

limite_partidos = st.sidebar.selectbox(
    "Máximo de partidos por consulta",
    [3, 5, 8, 10],
    index=1
)

if API_KEY:
    st.sidebar.success("Conexión API configurada")
else:
    st.sidebar.error("Falta configurar la clave API")


# --------------------------------------------------
# EXPLICACIÓN
# --------------------------------------------------

st.info(
    "SmartBet AI consulta los partidos y sus predicciones. "
    "Solo destacará los pronósticos que alcancen el porcentaje "
    "mínimo seleccionado. Una predicción no garantiza un resultado."
)

# --------------------------------------------------
# BOTÓN PRINCIPAL
# --------------------------------------------------

if st.button("🔎 BUSCAR PARTIDOS Y GENERAR PRONÓSTICOS", type="primary"):

    if not API_KEY:
        st.error(
            "Configura API_SPORTS_KEY en los Secrets de Streamlit."
        )

    else:
        try:
            with st.spinner("Buscando partidos..."):

                partidos = consultar_api(
                    "fixtures",
                    {
                        "date": fecha.strftime("%Y-%m-%d"),
                        "timezone": "Europe/Madrid"
                    }
                )

            # Solo partidos que todavía no han comenzado
            partidos_pendientes = [
                p for p in partidos
                if p.get("fixture", {}).get("status", {}).get("short") == "NS"
            ]

            partidos_pendientes = partidos_pendientes[:limite_partidos]

            if not partidos_pendientes:
                st.warning(
                    "No se encontraron partidos pendientes "
                    "para esa fecha."
                )

            else:
                st.success(
                    f"Se han encontrado {len(partidos_pendientes)} partidos."
                )

                resultados = []

                barra = st.progress(0)

                for indice, partido in enumerate(partidos_pendientes):

                    fixture = partido.get("fixture", {})
                    equipos = partido.get("teams", {})
                    liga = partido.get("league", {})

                    equipo_local = equipos.get("home", {}).get(
                        "name", "Local"
                    )

                    equipo_visitante = equipos.get("away", {}).get(
                        "name", "Visitante"
                    )

                    fixture_id = fixture.get("id")

                    try:
                        prediccion = obtener_pronostico(fixture_id)

                        if not prediccion:
                            resultados.append({
                                "Partido": f"{equipo_local} - {equipo_visitante}",
                                "Pronóstico": "Sin datos",
                                "Probabilidad": None,
                                "Liga": liga.get("name", ""),
                                "Consejo API": "",
                                "Estado": "NO APOSTAR"
                            })

                        else:
                            favorito, porcentaje = elegir_favorito(
                                prediccion,
                                equipo_local,
                                equipo_visitante
                            )

                            consejo = prediccion.get(
                                "predictions", {}
                            ).get("advice", "")

                            if favorito is None:
                                estado = "NO APOSTAR"
                                pronostico = "Sin datos suficientes"

                            elif porcentaje >= umbral:
                                estado = (
                                    "PENDIENTE DE COMPROBAR CUOTA"
                                )
                                pronostico = f"Gana {favorito}" if favorito != "Empate" else "Empate"

                            else:
                                estado = "NO APOSTAR"
                                pronostico = (
                                    f"Favorito: {favorito}"
                                )

                            resultados.append({
                                "Partido": f"{equipo_local} - {equipo_visitante}",
                                "Pronóstico": pronostico,
                                "Probabilidad": porcentaje,
                                "Liga": liga.get("name", ""),
                                "Consejo API": consejo,
                                "Estado": estado
                            })

                    except Exception as error:
                        mensaje = str(error)

                        st.warning(
                            f"No se pudo consultar "
                            f"{equipo_local} - {equipo_visitante}: {mensaje}"
                        )

                        # Si se alcanza el límite, detener las consultas
                        if "429" in mensaje or "Límite" in mensaje:
                            st.error(
                                "Se detuvieron las consultas para "
                                "evitar más errores."
                            )
                            break

                    barra.progress(
                        (indice + 1) / len(partidos_pendientes)
                    )

                st.session_state["resultados_pronosticos"] = resultados

        except Exception as error:
            st.error(f"Error al consultar los partidos: {error}")


# --------------------------------------------------
# MOSTRAR RESULTADOS
# --------------------------------------------------

if "resultados_pronosticos" in st.session_state:

    resultados = st.session_state["resultados_pronosticos"]

    st.divider()
    st.header("📊 Pronósticos encontrados")

    validos = [
        r for r in resultados
        if r["Probabilidad"] is not None
        and r["Probabilidad"] >= umbral
    ]

    if validos:
        st.subheader("Pronósticos que alcanzan el umbral")

        for resultado in validos:

            st.markdown(
                f"### {resultado['Partido']}"
            )

            st.write(
                f"**Pronóstico:** {resultado['Pronóstico']}"
            )

            st.write(
                f"**Probabilidad estimada:** "
                f"{resultado['Probabilidad']:.1f}%"
            )

            st.write(
                f"**Liga:** {resultado['Liga']}"
            )

            if resultado["Consejo API"]:
                st.write(
                    f"**Consejo del modelo:** "
                    f"{resultado['Consejo API']}"
                )

            st.warning(
                "Pendiente de comprobar una cuota real y el valor esperado. "
                "Esto todavía no es una apuesta validada."
            )

            st.divider()

    else:
        st.warning(
            f"Ningún partido ha alcanzado el {umbral}% "
            "de probabilidad estimada."
        )

    st.subheader("Todos los partidos consultados")

    tabla = pd.DataFrame(resultados)

    st.dataframe(
        tabla,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Descargar pronósticos en CSV",
        data=tabla.to_csv(index=False).encode("utf-8-sig"),
        file_name="pronosticos_smartbet.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# AVISO FINAL
# --------------------------------------------------

st.divider()

st.caption(
    "SmartBet AI es una herramienta informativa. "
    "Las probabilidades son estimaciones, no garantías. "
    "No realiza apuestas automáticamente."
)
