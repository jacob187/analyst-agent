import sys
import yfinance as yf

# Define the ticker symbol
ticker_symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

# Create a Ticker object
ticker = yf.Ticker(ticker_symbol)

# Fetch historical market data
historical_data = ticker.history(period="1y")  # data for the last year
financials = ticker.financials
print(financials)
