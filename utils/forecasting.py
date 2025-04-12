import numpy as np
import matplotlib.pyplot as plt
import os
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel

def train_predict_model(time_series: TimeSeries, prediction_days: int):
    """
    Escala los datos, entrena el modelo N-HiTS y realiza la predicción.
    """
    train = time_series
    train_scaler = Scaler()
    scaled_train = train_scaler.fit_transform(train)
    nhits = NHiTSModel(
        input_chunk_length=90,
        output_chunk_length=30,
        random_state=42,
        num_stacks = 4,
        num_blocks = 3,
        num_layers = 3
    )

    nhits.fit(scaled_train, epochs=30)
    forecast = nhits.predict(n=prediction_days)
    return train_scaler.inverse_transform(forecast)

def create_forecast_plot(historical_series: TimeSeries, forecast_series: TimeSeries, prediction_days: int, filename: str):
    """
    Crea y guarda la gráfica del pronóstico en la carpeta 'Forecasts'.
    """
    # Crear la carpeta 'Forecasts' si no existe.
    folder_name = "Forecasts"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Combinar la carpeta con el nombre de archivo.
    save_path = os.path.join(folder_name, filename)
    
    plt.figure(figsize=(12, 6))
    historical_series[-360:].plot(label="Histórico Año anterior")
    forecast_series.plot(label=f"Pronóstico ({prediction_days} días)", lw=2)
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.legend()
    plt.savefig(save_path)
    plt.close()