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

## Examples of Use

You can ask investment strategy questions like:
- "I have 1000€ to invest in Bitcoin. What's the best strategy to maximize profits in the next 30 days?"
- "I'm holding Ethereum. Should I sell or hold based on the next 2 weeks prediction?"
- "I need to recover from a 20% loss in Bitcoin. What strategy do you recommend for the next 60 days?"

## Testing

You can run the model evaluation script for BTC-USD values:

```sh
python test/nhits_BTC-USD.py
```

This will generate performance metrics including MAPE, correlation, and MPE.

### Model Performance Metrics

| Metric | Value |
|--------|-------|
| MAPE | 5.72% |
| Correlation | 0.8671 |
| MPE | 5.24% |

- **MAPE**: Mean Absolute Percentage Error (lower is better)
- **Correlation**: Pearson correlation between predicted and actual values (closer to 1 is better)
- **MPE**: Mean Percentage Error (measures prediction bias)

## License

This project is open source and available under the MIT License.