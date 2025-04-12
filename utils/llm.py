import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig
import PIL.Image

from utils.data import download_crypto_data, process_data_for_darts
from utils.forecasting import train_predict_model, create_forecast_plot

def generate_llm_response(prompt: str, image: PIL.Image.Image = None):
    """
    Genera una respuesta del LLM utilizando la API de Gemini.
    Si se proporciona una imagen, se utiliza para generar recomendaciones de compra o venta.
    Si no se proporciona imagen, se utiliza el prompt para generar un JSON con los parámetros de la criptomoneda.
    """
    load_dotenv(dotenv_path=".env")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("No se encontró GEMINI_API_KEY en las variables de entorno.")
    client = genai.Client(api_key=gemini_api_key)

    contents = [prompt]
    if image:
        contents.append(image)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=GenerateContentConfig(
            system_instruction=[
                """
                Tu respuesta debe ser únicamente en formato JSON. El JSON debe contener los siguientes campos:
                - "cripto_name": El código de la criptomoneda que el usuario quiere predecir. Por ejemplo, si es bitcoin en dólares, el valor de este campo debe ser "BTC-USD"; si es ethereum en euros, el valor de este campo debe ser "ETH-EUR", etc. Si no se indica moneda, se asume que son dólares.
                - "period_historical": El período histórico para los datos (ej. 3y, 5y). Si el usuario no indica nada, el valor estándar será "3y".
                - "period_prediction": El periodo de tiempo para la predicción, el cual debe ser en días (supon que 1 año son 360 días, ya que es el año bancario).
                """
            ]
        ) if not image else [
                f"""Eres un experto invirtiendo en criptomonedas. A partir de la gráfica que se te proporciona,
                aconseja al usuario cuándo es el mejor momento para comprar o vender y ofrece la mejor estrategia para maximizar beneficios.
                Ten en cuenta que la criptomoneda es {prompt.get("cripto_name")} y que el periodo de predicción es de {prompt.get("period_prediction")} días.
                No indiques que se te proporciona una gráfica, simplemente ofrece recomendaciones basadas en la misma.
                """
            ]
    )
    return response.text.strip()

def process_llm_output(output_text: str):
    """
    Procesa la respuesta del LLM para extraer el JSON.
    """
    if output_text.startswith("```json") and output_text.endswith("```"):
        output_text = output_text[7:-3].strip()  # Eliminar los delimitadores ```json```
    return json.loads(output_text)

def get_llm_recommendation(prompt:str, image: PIL.Image.Image, crypto_name: str, prediction_period: str):
    """
    Obtiene una recomendación del LLM basada en la imagen del pronóstico.
    """
    load_dotenv(dotenv_path=".env")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("No se encontró GEMINI_API_KEY en las variables de entorno.")
    client = genai.Client(api_key=gemini_api_key)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[image],
        config=GenerateContentConfig(
            system_instruction=[
                f"""Eres un experto invirtiendo en criptomonedas. A partir de la gráfica que se te proporciona,
                aconseja al usuario cuándo es el mejor momento para comprar o vender y ofrece la mejor estrategia para maximizar beneficios.
                Ten en cuenta que la criptomoneda es {crypto_name} y que el periodo de predicción es de {prediction_period} días.
                No indiques que se te proporciona una gráfica, simplemente ofrece recomendaciones basadas en la misma.
                """
            ]
        ),
    )
    llm_response = response.text
    response_clean = llm_response.strip()
    return response_clean

def run_pipeline(prompt_text: str):
    """
    Función principal que coordina todo el pipeline.
    """
    try:
        # Obtener parámetros del LLM
        llm_response_json = generate_llm_response(prompt_text)
        json_cripto = process_llm_output(llm_response_json)

        # Descargar y procesar datos
        data_cripto = download_crypto_data(json_cripto.get("cripto_name"), json_cripto.get("period_historical"))
        if isinstance(data_cripto, str):
            return data_cripto
        time_series = process_data_for_darts(data_cripto, json_cripto.get("cripto_name"))

        # Entrenar y predecir con el modelo
        forecast = train_predict_model(time_series, int(json_cripto.get("period_prediction")))

        # Crear la gráfica del pronóstico
        plot_filename = f"prediccion_{json_cripto.get('cripto_name')}.png"
        create_forecast_plot(time_series, forecast, json_cripto.get("period_prediction"), plot_filename)

        # Construir la ruta completa donde se guardó la imagen.
        image_path = os.path.join("Forecasts", plot_filename)
        image = PIL.Image.open(image_path)
        recommendation = get_llm_recommendation(prompt_text, image, json_cripto.get("cripto_name"), json_cripto.get("period_prediction"))

        return recommendation

    except Exception as e:
        return f"Error: {str(e)}"
    