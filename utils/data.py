import yfinance as yf
import pandas as pd
from darts import TimeSeries

def download_crypto_data(crypto_ticker: str, period: str = "3y"):
    """
    Descarga datos históricos de la criptomoneda especificada usando yfinance.
    """
    data = yf.download(crypto_ticker, period=period)
    if data.empty:
        return f"Error: No se pudieron descargar datos para la criptomoneda {crypto_ticker}."
    return data.dropna()

def process_data_for_darts(df: pd.DataFrame, crypto_name: str):
    """
    Procesa el DataFrame para crear un objeto TimeSeries de Darts.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df_reset = df.reset_index()
    time_series = TimeSeries.from_dataframe(df_reset, time_col='Date', value_cols=f'Close_{crypto_name}')
    return time_series