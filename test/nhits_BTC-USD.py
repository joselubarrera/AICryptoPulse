import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel
from darts.metrics import mape, rmse, mae
from sklearn.metrics import r2_score

# Descargar datos históricos de BTC-USD
data_cripto = yf.download("BTC-USD", period="3y")
data_cripto = data_cripto.dropna()

# Procesar datos para crear TimeSeries
data_cripto.columns = ['_'.join(col).strip() for col in data_cripto.columns.values]
data_cripto_reset = data_cripto.reset_index()
series = TimeSeries.from_dataframe(data_cripto_reset, time_col='Date', value_cols='Close_BTC-USD')

# Dividir la serie en entrenamiento y test (90% - 10%)
train_size = int(len(series) * 0.9)
train = series[:train_size]
test = series[train_size:]

# Escalar los datos
scaler = Scaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Configurar y entrenar modelo N-HiTS
nhits = NHiTSModel(
    input_chunk_length=90,
    output_chunk_length=30,
    random_state=42,
    num_stacks = 4,
    num_blocks = 3,
    num_layers = 3
)

# Entrenar modelo
print("Entrenando modelo...")
nhits.fit(train_scaled, epochs=30)

# Realizar predicciones
print("Realizando predicciones...")
predictions = nhits.predict(n=len(test))
predictions = scaler.inverse_transform(predictions)

# Calcular métricas de error
mape_score = mape(test, predictions)

# Calcular coeficiente de correlación
correlation = np.corrcoef(test.values().flatten(), predictions.values().flatten())[0,1]

# Calcular MPE (Mean Percentage Error)
mpe = np.mean((test.values() - predictions.values()) / test.values()) * 100

print("\nMétricas de evaluación:")
print(f"MAPE: {mape_score:.2f}%")
print(f"Correlación: {correlation:.4f}")
print(f"MPE: {mpe:.2f}%")

# Visualizar resultados
plt.figure(figsize=(12,6))
test.plot(label="Test")
predictions.plot(label="Predicciones", lw=2)
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.title("Evaluación del modelo N-HiTS para BTC-USD")
plt.legend()
plt.savefig("evaluacion_nhits_btc.png")
plt.close()

print("\nGráfico guardado como 'evaluacion_nhits_btc.png'")