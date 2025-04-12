# AICryptoPulse: Crypto Prediction and Recommendation using N-HiTS and Gemini

This project provides a pipeline for predicting cryptocurrency prices and offering investment recommendations using N-HiTS forecasting and Gemini API.

## Project Structure

- **main.py**: Contains the Gradio interface that serves as the frontend for users to input their queries. See [main.py].
- **utils/**
  - **llm.py**: Handles LLM interactions using the Gemini API to strucure user query and generate recommendations.
  - **forecasting.py**: Contains functions for training the N-HiTS forecasting model using Darts package and creating forecast plots.
  - **data.py**: Provides functions to download and process cryptocurrency data using Yahoo Finance.
- **Forecasts/**: Directory where forecast plots are saved.
- **.env**: Environment file for storing sensitive credentials (e.g., Gemini API key). See [`.env.example`] for guidance.
- **requirements.txt**: Lists the Python dependencies.

## Features

- **Download Crypto Data**: Uses [yfinance](https://pypi.org/project/yfinance/) to download historical cryptocurrency data.
- **Forecasting**: Trains an N-HiTS forecasting model from the [Darts library](https://github.com/unit8co/darts) to predict future prices.
- **Visualization**: Generates forecast plots with Matplotlib and saves them in the `Forecasts` folder.
- **LLM Integration**: Uses Google’s Gemini API to provide investment recommendations based on the forecast plot.
- **Gradio Interface**: A simple web UI powered by Gradio allowing users to input their prediction queries.

## Installation

1. **Clone the repository and change the directory:**

   ```sh
   cd AICryptoPulse
   ```

2. **Create and activate a virtual environment (highly recommended):**

    ```sh
    python -m venv venv

    venv\Scripts\activate # For Windows
    source venv/bin/activate # For MacOS/Linux
    ```

3. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Set up the `GEMINI_API_KEY` into the `.env.example` file and rename the file to `.env`**

## Usage

1. **Run the Gradio interface:**

    ```sh
    python main.py
    ```

2. Once the application is running, a local web interface will open automatically. Enter your query for a cryptocurrency prediction and recommendation and click `submit`.


