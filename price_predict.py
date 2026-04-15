import yfinance as yf

# Download historical data for Apple
data = yf.download("AAPL", start="2023-01-01", end="2025-01-01")
print(data.head())