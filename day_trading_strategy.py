import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
from datetime import datetime  # إضافة مكتبة datetime

# Fetch historical price data
def fetch_historical_data(symbol, interval, limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df["close"] = pd.to_numeric(df["close"], errors='coerce')
    df["open"] = pd.to_numeric(df["open"], errors='coerce')
    df["high"] = pd.to_numeric(df["high"], errors='coerce')
    df["low"] = pd.to_numeric(df["low"], errors='coerce')
    df["volume"] = pd.to_numeric(df["volume"], errors='coerce')
    return df

# Feature engineering
def create_features(df):
    df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()  # إضافة EMA 200 لتأكيد الاتجاه العام
    df["rsi"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    stoch = ta.stoch(df["high"], df["low"], df["close"], fast_k=14, slow_k=3, slow_d=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]
    df.dropna(inplace=True)
    return df

# Generate trading signals
def generate_signals(df_small, df_large):
    df_small["signal"] = 0
    df_small.loc[(df_small["rsi"] < 30) & (df_small["stoch_k"] < 20) & (df_small["stoch_d"] < 20), "signal"] = 1  # إشارة شراء عند الانعكاس الصاعد
    df_small.loc[(df_small["rsi"] > 70) & (df_small["stoch_k"] > 80) & (df_small["stoch_d"] > 80), "signal"] = -1  # إشارة بيع عند الانعكاس الهابط
    return df_small

# Main function
def main():
    symbol = "BTCUSDT"
    interval_small = "15m"  # إطار زمني أصغر للدخول في الصفقات
    interval_large = "4h"  # إطار زمني أكبر لتحديد الاتجاه العام
    
    df_small = fetch_historical_data(symbol, interval_small)
    df_large = fetch_historical_data(symbol, interval_large)
    
    df_small = create_features(df_small)
    df_large = create_features(df_large)
    
    df_small = generate_signals(df_small, df_large)
    
    df_small["time"] = pd.to_datetime(df_small["time"], unit='ms')  # تحويل الطابع الزمني إلى تنسيق زمني قابل للقراءة
    print(df_small[["time", "close", "signal"]].tail())

    # تحديد سعر الدخول
    close_price = df_small["close"].iloc[-1]
    if df_small["signal"].iloc[-1] == 1:
        entry_price = close_price
    elif df_small["signal"].iloc[-1] == -1:
        entry_price = close_price
    else:
        entry_price = None
    print(f"Entry Price: {entry_price}")

    # تحديد سعر إيقاف الخسارة
    if df_small["signal"].iloc[-1] == 1:
        stop_loss = df_small["low"].min() - (df_small["low"].min() * 0.01)
    elif df_small["signal"].iloc[-1] == -1:
        stop_loss = df_small["high"].max() + (df_small["high"].max() * 0.01)
    else:
        stop_loss = None
    print(f"Stop Loss: {stop_loss}")

    # تحديد سعر أخذ الربح
    if df_small["signal"].iloc[-1] == 1:
        take_profit1 = entry_price + (df_small["high"].max() - entry_price) * 0.33
        take_profit2 = entry_price + (df_small["high"].max() - entry_price) * 0.66
        take_profit3 = df_small["high"].max() - (df_small["high"].max() * 0.01)
    elif df_small["signal"].iloc[-1] == -1:
        take_profit1 = entry_price - (entry_price - df_small["low"].min()) * 0.33
        take_profit2 = entry_price - (entry_price - df_small["low"].min()) * 0.66
        take_profit3 = df_small["low"].min() + (df_small["low"].min() * 0.01)
    else:
        take_profit1 = take_profit2 = take_profit3 = None
    print(f"Take Profit 1: {take_profit1}")
    print(f"Take Profit 2: {take_profit2}")
    print(f"Take Profit 3: {take_profit3}")

if __name__ == "__main__":
    main()
