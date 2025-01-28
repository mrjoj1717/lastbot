import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
import pandas_ta as ta
from datetime import datetime
import aiohttp
import asyncio
import xgboost as xgb  # إضافة مكتبة XGBoost
import lightgbm as lgb  # إضافة مكتبة LightGBM
import os

# تعيين SelectorEventLoop على نظام Windows
if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Fetch historical price data
async def fetch_historical_data(symbol, interval, limit=1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    connector = aiohttp.TCPConnector(ssl=False)  # استخدام TCPConnector
    async with aiohttp.ClientSession(connector=connector) as session:
        async with session.get(url) as response:
            data = await response.json()
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
    df["rsi"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd["MACD_12_26_9"]
    df["macd_signal"] = macd["MACDs_12_26_9"]
    df["macd_hist"] = macd["MACDh_12_26_9"]
    bbands = ta.bbands(df["close"], length=20, std=2)
    df["bollinger_upper"] = bbands["BBU_20_2.0"]
    df["bollinger_middle"] = bbands["BBM_20_2.0"]
    df["bollinger_lower"] = bbands["BBL_20_2.0"]
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    stoch = ta.stoch(df["high"], df["low"], df["close"], fast_k=14, slow_k=3, slow_d=3)
    df["stoch_k"] = stoch["STOCHk_14_3_3"]
    df["stoch_d"] = stoch["STOCHd_14_3_3"]
    df["adx"] = ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["profit_loss_ratio"] = df["close"] / df["close"].shift(1) - 1  # نسبة الربح والخسارة
    df["volume_change"] = df["volume"].pct_change()  # التغير في حجم التداول
    df["ema_diff"] = df["ema_9"] - df["ema_21"]  # الاختلاف بين EMA طويل وقصير المدى
    df["target"] = np.where(df["close"].shift(-1) > df["close"], 1, 0)
    df.dropna(inplace=True)
    return df

# Train machine learning model
def train_model(df):
    features = ["ema_9", "ema_21", "rsi", "macd", "macd_signal", "bollinger_upper", "bollinger_middle", "bollinger_lower", "atr", "stoch_k", "stoch_d", "adx", "profit_loss_ratio", "volume_change", "ema_diff"]
    X = df[features]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # تجربة نموذج XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # تجربة نموذج LightGBM
    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=10, learning_rate=0.05, random_state=42)
    lgb_model.fit(X_train, y_train)
    
    y_pred_xgb = xgb_model.predict(X_test)
    y_pred_lgb = lgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
    print(f"XGBoost Model Accuracy: {accuracy_xgb:.2f}")
    print(f"LightGBM Model Accuracy: {accuracy_lgb:.2f}")
    
    # اختيار النموذج الأفضل بناءً على الدقة
    if accuracy_xgb > accuracy_lgb:
        return xgb_model
    else:
        return lgb_model

# Generate trading signals
def generate_signals(df, model):
    features = ["ema_9", "ema_21", "rsi", "macd", "macd_signal", "bollinger_upper", "bollinger_middle", "bollinger_lower", "atr", "stoch_k", "stoch_d", "adx", "profit_loss_ratio", "volume_change", "ema_diff"]
    df["prediction"] = model.predict(df[features])
    df["signal"] = df["prediction"].diff()
    return df

# Main function
async def main():
    symbol = "BTCUSDT"
    interval = "5m"  # يمكنك تغيير الفريم الزمني هنا
    limit = 50  # يمكنك تغيير الحد هنا
    df = await fetch_historical_data(symbol, interval, limit)
    df = create_features(df)
    model = train_model(df)
    df = generate_signals(df, model)
    df["time"] = pd.to_datetime(df["time"], unit='ms')  # تحويل الطابع الزمني إلى تنسيق زمني قابل للقراءة
    print(df[["time", "close", "signal"]].tail())

    # تحديد سعر الدخول
    close_price = df["close"].iloc[-1]
    if df["signal"].iloc[-1] == 1:
        entry_price = close_price
    elif df["signal"].iloc[-1] == -1:
        entry_price = close_price
    else:
        entry_price = None
    print(f"Entry Price: {entry_price}")

    # تحديد سعر إيقاف الخسارة
    if df["signal"].iloc[-1] == 1:
        stop_loss = df["low"].min() - (df["low"].min() * 0.01)
    elif df["signal"].iloc[-1] == -1:
        stop_loss = df["high"].max() + (df["high"].max() * 0.01)
    else:
        stop_loss = None
    print(f"Stop Loss: {stop_loss}")

    # تحديد سعر أخذ الربح
    if df["signal"].iloc[-1] == 1:
        take_profit1 = entry_price + (df["high"].max() - entry_price) * 0.33
        take_profit2 = entry_price + (df["high"].max() - entry_price) * 0.66
        take_profit3 = df["high"].max() - (df["high"].max() * 0.01)
    elif df["signal"].iloc[-1] == -1:
        take_profit1 = entry_price - (entry_price - df["low"].min()) * 0.33
        take_profit2 = entry_price - (entry_price - df["low"].min()) * 0.66
        take_profit3 = df["low"].min() + (df["low"].min() * 0.01)
    else:
        take_profit1 = take_profit2 = take_profit3 = None
    print(f"Take Profit 1: {take_profit1}")
    print(f"Take Profit 2: {take_profit2}")
    print(f"Take Profit 3: {take_profit3}")

if __name__ == "__main__":
    asyncio.run(main())