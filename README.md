# Stock Analyst Agent

An AI-powered financial analysis system that retrieves, processes, and analyzes financial data from multiple sources including SEC filings and market data.

## Features

- **SEC Filings Analysis**: Extracts and analyzes Management Discussion & Analysis, Risk Factors, and Balance Sheets
- **Technical Analysis**: Calculates technical indicators like RSI, MACD, moving averages, and more
- **Structured Data**: Uses structured JSON format for all data storage and retrieval
- **Modular Architecture**: Follows SOLID principles with clear separation of concerns

## Installation

1. Ensure you have Python 3.12+ installed
2. Clone this repository
3. Install dependencies:

```bash
pip install poetry
poetry install
```

4. Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
SEC_HEADER="your-name your@email.com"  # Required for SEC Edgar API
```

## Usage

### Basic Usage

Run a complete analysis for a stock:

```bash
poetry run python agents/main_agent.py TICKER
```

Replace `TICKER` with the stock symbol you want to analyze (e.g., AAPL, MSFT, NVDA).

### Run Only SEC Analysis

```bash
poetry run python agents TICKER
```

### Run Only Technical Analysis

```bash
poetry run python agents/technical_workflow/main_technical_workflow.py TICKER
```

### Output

The tool will generate a JSON output with the analysis results, including:

- Fundamental analysis from SEC filings
- Technical indicators and signals
- Overall recommendation (buy/hold/sell)
- Confidence level and key reasons for the recommendation

All data is also stored in the `data/data.json` file for future reference.

## Components

### SEC Agent

Analyzes SEC filings (10-K and 10-Q) to extract critical financial information:

- Extracts Management Discussion & Analysis
- Extracts Risk Factors
- Extracts Balance Sheets in structured JSON format
- Analyzes text using LLMs to extract insights

### Technical Agent

Retrieves and analyzes market data from Yahoo Finance:

- Retrieves historical price data
- Calculates technical indicators (RSI, MACD, etc.)
- Provides market context to supplement fundamental analysis

### Analyst Agent

In development. Will combine outputs from SEC and Technical agents to provide comprehensive analysis:

- Weighs both fundamental and technical factors
- Generates a final recommendation
- Provides reasoning for the recommendation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project uses [Edgar Tools](https://github.com/dgunning/edgartools) for SEC filings retrieval
- Market data is retrieved using [Yahoo Finance](https://pypi.org/project/yfinance/)
- Language models are powered by OpenAI's API via [LangChain](https://python.langchain.com/)
