
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.ar_model import AutoReg
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
import plotly.express as px

import plotly.graph_objects as go

st.set_page_config(page_title="EDA IPC Chile", layout="wide")

st.title("Análisis Exploratorio del IPC Chileno")
st.write("""Este análisis muestra una exploración del índice de precios al consumidor (IPC) de Chile a lo largo del tiempo,
teniendo en cuenta su rebalanceo en 2023 y su estacionalidad.""")

# Carga de datos
df_ipc_2023 = pd.read_csv("data/ipc_base_20237baa955a44fe4eada201c196338fb3be.csv", encoding="latin1", sep=";", decimal=",", on_bad_lines='skip')
df_ipc_2018 = pd.read_csv("data/ipc-csv.csv", encoding="latin1", sep=";", decimal=",", on_bad_lines='skip')

if df_ipc_2018.columns.equals(df_ipc_2023.columns):
    df_ipc = pd.concat([df_ipc_2018, df_ipc_2023], ignore_index=True, verify_integrity=True)
    df_ipc.rename(columns={"Índice": "IPC-NB"}, inplace=True)
else:
    st.error("Error: Las columnas entre los archivos IPC no coinciden.")
    st.stop()

# Filtrado
df_ipc = df_ipc[df_ipc["Glosa"] == "IPC General"][["Año", "Mes", "IPC-NB"]].astype({
    "Año": str,
    "Mes": str,
    "IPC-NB": float
})
df_ipc["Año-Mes"] = pd.to_datetime(df_ipc["Año"] + "-" + df_ipc["Mes"])
df_ipc.index = df_ipc["Año-Mes"]
df_ipc.drop(["Año", "Mes", "Año-Mes"], axis=1, inplace=True)

# Ajuste rebalanceo
valor_pre = df_ipc.loc["2023-12-01", "IPC-NB"]
valor_post = df_ipc.loc["2024-01-01", "IPC-NB"]
df_ipc["ipc_ajustado"] = df_ipc["IPC-NB"]
df_ipc.loc[df_ipc.index >= "2024-01-01", "ipc_ajustado"] = valor_pre + (df_ipc.loc[df_ipc.index >= "2024-01-01", "IPC-NB"] - 100)

# Gráfico comparativo IPC Original vs Ajustado
fig = go.Figure()

# Línea: IPC original
fig.add_trace(go.Scatter(
    x=df_ipc.index,
    y=df_ipc["IPC-NB"],
    mode='lines',
    name='IPC Original',
    line=dict(color='royalblue')
))

# Línea: IPC ajustado (tras rebalanceo)
fig.add_trace(go.Scatter(
    x=df_ipc.index,
    y=df_ipc["ipc_ajustado"],
    mode='lines',
    name='IPC Ajustado',
    line=dict(color='firebrick', dash='dash')
))

# Personalización del gráfico
fig.update_layout(
    title="Serie Temporal del IPC (Original vs Ajustado)",
    xaxis_title="Fecha",
    yaxis_title="Índice IPC",
    legend=dict(x=0.01, y=0.99),
    height=500
)

# Mostrar en Streamlit
st.subheader("Serie Temporal del IPC: Original vs Ajustado")
st.plotly_chart(fig)

# Descomposición estacional
st.subheader("Descomposición estacional Serie IPC ajustada")



# --- ACF y PACF ---
st.subheader("Autocorrelación (ACF) y Autocorrelación Parcial (PACF)")
st.write("Como se puede observar de los graficos los valores pasra el ACF decaen de forma progresiva por lo que se sugiere que esxiste persistencia o una tendencia en la serie" \
"Mientras que la PACF que mide la relación entre una observación y su rezagos (Datos anteriores), indica que para cada valor la mayor dependencia es hacia los ultimos dos valores o rezasgos" \
"Teniendo esto en cuenta podemos hacer el analisis de que esta serie temporal es candidata a una AutoRegresión en este caso como indica la PACF en un AutoReg(2), los valores fueron utilizando como" \
"variable nlags= 24 para ver hasta un maximo de 24 meses atras para cada valor y así encuentrar la cantidad exacta que explicarían los valores")
series = df_ipc['ipc_ajustado'].dropna()

# Calcular valores
acf_vals = acf(series, nlags=24)
pacf_vals = pacf(series, nlags=24)

# ACF interactivo
fig_acf = go.Figure()
fig_acf.add_trace(go.Bar(x=list(range(len(acf_vals))), y=acf_vals, name="ACF", marker_color="cornflowerblue"))
fig_acf.update_layout(title="Función de Autocorrelación (ACF)", xaxis_title="Rezago", yaxis_title="Correlación")
st.plotly_chart(fig_acf)

# PACF interactivo
fig_pacf = go.Figure()
fig_pacf.add_trace(go.Bar(x=list(range(len(pacf_vals))), y=pacf_vals, name="PACF", marker_color="salmon"))
fig_pacf.update_layout(title="Función de Autocorrelación Parcial (PACF)", xaxis_title="Rezago", yaxis_title="Correlación")
st.plotly_chart(fig_pacf)

# --- Estacionalidad mensual ---
st.subheader("Promedio mensual del IPC ajustado")

st.write("La verdad que esta grafica no es muy explicativa por como está actualmente, se puede indicar una pequeña variación o valle en los meses de Junio y Julio, pero es minimo " \
"esto debido a la tendencia lineal que presenta los valores de la serie temporal")
orden_meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
df_ipc["Mes_nombre"] = df_ipc.index.strftime('%b')
df_ipc["Mes_nombre"].replace({"Jan": "Ene", "Apr": "Abr", "Aug": "Ago", "Dec": "Dic"}, inplace=True)
df_ipc["Mes_nombre"] = pd.Categorical(df_ipc["Mes_nombre"], categories=orden_meses, ordered=True)

media_mensual = df_ipc.groupby("Mes_nombre")["ipc_ajustado"].mean().reset_index()
fig_media_mensual = px.bar(media_mensual, x="Mes_nombre", y="ipc_ajustado",
                           title="IPC Promedio por Mes", labels={"ipc_ajustado": "IPC Promedio"},
                           color_discrete_sequence=["cornflowerblue"])
st.plotly_chart(fig_media_mensual)

# --- Variación mensual promedio ---
st.subheader("Promedio de la variación mensual")
st.write("")
df_ipc["variacion_mensual"] = df_ipc["ipc_ajustado"].diff()
variacion_media = df_ipc.groupby("Mes_nombre")["variacion_mensual"].mean().reset_index()
fig_variacion_media = px.bar(variacion_media, x="Mes_nombre", y="variacion_mensual",
                             title="Variación Promedio Mensual del IPC", labels={"variacion_mensual": "Variación Promedio"},
                             color_discrete_sequence=["salmon"])
st.plotly_chart(fig_variacion_media)


st.subheader("Prueba de Estacionariedad: Dickey-Fuller Aumentada (ADF)")

adf_result = adfuller(df_ipc["ipc_ajustado"].dropna())

st.write("**Estadístico ADF:**", adf_result[0])
st.write("**p-valor:**", adf_result[1])
st.write("**Valores críticos:**")
for key, value in adf_result[4].items():
    st.write(f"   • {key}: {value}")

if adf_result[1] < 0.05:
    st.success("✅ Se rechaza la hipótesis nula: la serie es ESTACIONARIA.")
else:
    st.warning("⚠️ No se puede rechazar la hipótesis nula: la serie NO es estacionaria.")
    st.write("Al no ser estacionaria los valores presentan Tendencia Creciente, como se puede observar en el grafico de la comparación entre Original y Ajustado y tiene " \
    "una estacionalidad mensual como se puede observar minimamente en las tendencias")