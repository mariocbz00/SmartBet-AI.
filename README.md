# SmartBet AI

Aplicación local en Python/Streamlit para análisis de cuotas, cálculo de valor esperado, entrenamiento inicial de un clasificador y seguimiento de apuestas.

## Instalación

Requiere Python 3.10 o superior.

### Windows
1. Descomprime el ZIP.
2. Abre PowerShell dentro de la carpeta.
3. Ejecuta:
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   streamlit run app.py
   ```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Se abrirá en el navegador, normalmente en http://localhost:8501. La base SQLite `smartbet.db` se crea automáticamente.

## Uso

### Analizador
Sube un CSV con estas columnas:
- `sport`: deporte
- `event`: evento
- `market`: mercado (ganador, hándicap, goles, puntos, props, etc.)
- `selection`: selección
- `odds`: cuota decimal
- `probability`: probabilidad estimada entre 0 y 1

La app calcula probabilidad implícita, cuota justa, EV y una sugerencia de stake con Kelly fraccional limitada al 2% del bankroll por apuesta. Solo marca oportunidades que superan los filtros elegidos.

### Entrenar modelo
Sube datos históricos etiquetados con `target` (1 si la selección ganó, 0 si perdió) y variables numéricas conocidas antes del evento. Selecciona las variables y entrena. El modelo se guarda localmente como `smartbet_model.joblib`. Para predecir, carga un CSV futuro con las mismas variables. Combina la probabilidad resultante con las cuotas en la pestaña Analizador.

**Importante:** el entrenamiento incluido es una base funcional, no un modelo universal de todos los deportes. Necesitas datos históricos representativos, features específicos por deporte/mercado, validación temporal, calibración de probabilidades y pruebas fuera de muestra. Evita usar variables posteriores al evento (data leakage).

### Registro
Guarda apuestas recomendadas, liquida cada una como ganada/perdida/anulada y exporta el historial. El ROI se calcula sobre el importe apostado liquidado.

## Fórmulas

- Probabilidad implícita: `1 / cuota`
- EV por unidad apostada: `p * cuota - 1`
- Cuota justa: `1 / p`
- Kelly: `(cuota * p - 1) / (cuota - 1)`

La probabilidad implícita de una sola cuota no elimina el margen de la casa. Para mercados con varios resultados, compara todas las cuotas del mismo mercado y elimina el overround antes de usar el mercado como referencia.

## Límites y seguridad

- No incluye conexión automática a casas de apuestas, feeds deportivos ni scraping.
- No garantiza ganancias ni identifica una "mejor apuesta" cierta.
- No realiza apuestas automáticas.
- Las probabilidades manuales o del modelo deben ser contrastadas y calibradas.
- Usa límites de stake conservadores y nunca arriesgues dinero que no puedas permitirte perder.


## Perfil SmartBet AI 2.0 — Máxima seguridad
La configuración predeterminada está ajustada a:
- Probabilidad estimada mínima: **95%**.
- Valor esperado: **estrictamente positivo (EV > 0)**.
- Estrategia: máxima selectividad; puede no recomendar ninguna apuesta.
- Stake orientativo: Kelly fraccional, limitado al 2% del bankroll por selección.

Un 95% es una estimación del modelo, no una garantía. Para que tenga sentido debe estar calibrada con datos suficientes y validación temporal fuera de muestra. Cuotas muy bajas también pueden tener EV negativo. No se debe interpretar el umbral como una promesa de acierto.


## Perfil SmartBet AI 2.0 — máxima seguridad

La configuración inicial queda fijada en:
- Probabilidad mínima estimada: **95%**.
- EV mínimo: **estrictamente positivo** (umbral inicial configurable desde +0,1%; por defecto +3%).
- Estrategia: priorizar probabilidad y calidad de datos, aunque no haya selecciones disponibles.
- Si ninguna selección supera ambos filtros, la aplicación muestra que no hay apuesta recomendada.

Un 95% es una estimación del modelo, no una garantía. Las probabilidades deben estar calibradas y validadas fuera de muestra; cuotas bajas también pueden tener EV negativo.
