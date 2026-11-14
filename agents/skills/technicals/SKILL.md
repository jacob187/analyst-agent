---
name: technicals
description: >
  Stock price, valuation snapshot and technical analysis: current price, P/E,
  market cap, recent price history, RSI/MACD/Bollinger/moving averages,
  ADX/ATR/Stochastic/Fibonacci/volume profile, chart patterns, and
  multi-timeframe signals.
tools:
  - get_stock_info
  - get_stock_price_history
  - get_technical_analysis
  - get_advanced_technical_analysis
  - get_pattern_detection
  - get_multi_timeframe_analysis
---
# Price, valuation and technical analysis

Use this skill for stock price, valuation metrics and anything about technical indicators, chart patterns or trend.

## Tool selection

- `get_stock_info`: current price, trailing and forward P/E, price-to-book, market cap, 52-week range, volume, dividend yield, beta. Use it for price and valuation questions.
- `get_stock_price_history`: OHLC prices for the last 10 trading days.
- `get_technical_analysis`: RSI, MACD, Bollinger Bands, moving averages (5/10/20/50/200-day), volatility and trend signals. The default for "how does it look technically".
- `get_advanced_technical_analysis`: ADX (trend strength), ATR (volatility / position sizing), Stochastic (momentum), Volume Profile and Fibonacci levels.
- `get_pattern_detection`: Head & Shoulders, Double Top/Bottom, Golden/Death Cross and RSI divergences, with confidence scores.
- `get_multi_timeframe_analysis`: daily / weekly / hourly analysis with conflict detection and a weighted recommendation. Use it when the user asks about timeframes or conflicting signals.

## Explaining indicators

For every indicator you discuss, give:

1. CURRENT VALUE: what the indicator shows right now.
2. INTERPRETATION: what it means in plain language.
3. SIGNAL: bullish, bearish or neutral.
4. CONTEXT: historical behaviour or statistical edge.
5. ACTION: what a trader should consider.

Never just state "RSI is 32" without context. When indicators disagree, say which way the weight of evidence points and why.
