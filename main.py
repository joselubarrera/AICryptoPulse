import gradio as gr
from utils.llm import run_pipeline

# Definir la interfaz de Gradio
iface = gr.Interface(
    fn=run_pipeline,
    inputs=[
        gr.Textbox(label="Petición", placeholder="Escribe tu petición sobre predicción de criptomonedas"),
    ],
    outputs=gr.Textbox(label="Respuesta del LLM"),
    title="Predicción y Recomendación para Criptomonedas",
    description="Ingresa una petición sobre predicción de valores de una criptomoneda junto con la divisa (dólares, euros...). Se descargan los datos, se entrena un modelo N‑HiTS para el forecasting y se usa Gemini para ofrecer recomendaciones basadas en la gráfica."
)

if __name__ == "__main__":
    iface.launch()
    