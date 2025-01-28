import requests
import pandas as pd
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib
from telegram import Update
from telegram.ext import ContextTypes
import json
from cachetools import TTLCache
from dotenv import load_dotenv
import hmac
import hashlib
import time
import ccxt
import asyncio
import aiohttp

matplotlib.use('Agg')

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Default settings
DEFAULT_INTERVAL = "15m"
DEFAULT_LIMIT = 3
DEFAULT_MODEL = "random_forest"
TRAILING_STOP_PERCENT = 0.01
DEFAULT_LEVERAGE = 5

# Set up cache
cache = TTLCache(maxsize=500, ttl=300)

class CryptoBot:
    def __init__(self):
        self.channel_id = "@Future_Deals"
        self.settings = {
            "interval": DEFAULT_INTERVAL,
            "limit": DEFAULT_LIMIT,
            "symbols": [],
            "model": DEFAULT_MODEL,
            "leverage": DEFAULT_LEVERAGE
        }
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.secret_key = os.getenv("BINANCE_SECRET_KEY")
        self.binance = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'options': {
                'defaultType': 'future',
            },
        })

    def load_user_settings(self, user_id):
        try:
            with open(f"settings_{user_id}.json", "r") as f:
                self.settings = json.load(f)
        except FileNotFoundError:
            logger.warning(f"No settings file found for user {user_id}, using default settings.")

    def fetch_klines(self, symbol, interval, limit=1000):
        cache_key = f"{symbol}_{interval}_{limit}"
        if (cache_key in cache):
            return cache[cache_key]

        klines_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            klines_response = requests.get(klines_url)
            klines_response.raise_for_status()
            klines_data = klines_response.json()
            for kline in klines_data:
                kline[0] = int(kline[0]) // 1000
            cache[cache_key] = klines_data
            return klines_data
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {interval} data for {symbol}: {repr(e)}")
            return None

    def create_features(self, df):
        df["time"] = pd.to_datetime(df["time"], unit='s')  # Ensure time column is in datetime format
        df.set_index("time", inplace=True)  # Set time as index for proper VWAP calculation

        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()
        df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=200, adjust=False).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["rsi_7"] = ta.rsi(df["close"], length=12)  # Change RSI length to 12
        macd = ta.macd(df["close"], fast=12, slow=26, signal=9)  # Use standard MACD periods
        if macd is not None:
            df["macd"] = macd["MACD_12_26_9"]
            df["macd_signal"] = macd["MACDs_12_26_9"]
            df["macd_hist"] = macd["MACDh_12_26_9"]
        else:
            df["macd"] = df["macd_signal"] = df["macd_hist"] = None
        df["volume"] = df["volume"]
        adx = ta.adx(df["high"], df["low"], df["close"])
        if adx is not None:
            df["adx"] = adx["ADX_14"]
        else:
            df["adx"] = None
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)
        df["vwap"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"])  # Add VWAP
        bb = ta.bbands(df["close"], length=14, std=1.5)  # Reduce length to 14 and std to 1.5 for narrower bands
        df["bb_upper"] = bb["BBU_14_1.5"]
        df["bb_lower"] = bb["BBL_14_1.5"]
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
        df.dropna(inplace=True)
        return df

    def train_model(self, df):
        features = ["ema_9", "ema_21", "rsi_7", "macd", "macd_signal", "volume"]
        X = df[features]
        y = df["target"]
        if len(X) < 100 or len(y) < 100:
            logger.error("Not enough data to train the model.")
            return None, None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if len(X_train) == 0 or len(y_train) == 0:
            logger.error("Training set is empty after split.")
            return None, None

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {accuracy:.2f}")
        return model, accuracy

    def is_uptrend(self, ema_9, ema_21, ema_50, ema_200, previous_ema_9):
        return ema_9 > ema_21 and ema_21 > ema_50 and ema_50 > ema_200 and ema_9 > previous_ema_9

    def is_downtrend(self, ema_9, ema_21, ema_50, ema_200, previous_ema_9):
        return ema_9 < ema_21 and ema_21 < ema_50 and ema_50 < ema_200 and ema_9 < previous_ema_9

    def analyze_symbol(self, symbol, interval):
        logger.info(f"Analyzing symbol: {symbol} with interval: {interval}")

        # 1-hour timeframe data to determine the general trend and stop loss price
        klines_data_1h = self.fetch_klines(symbol, "1h")
        if not klines_data_1h or len(klines_data_1h) < 2:
            logger.warning(f"No klines data or not enough data for {symbol} on 1h interval")
            return None

        df_1h = pd.DataFrame(klines_data_1h, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        df_1h["close"] = pd.to_numeric(df_1h["close"], errors='coerce')
        df_1h["open"] = pd.to_numeric(df_1h["open"], errors='coerce')
        df_1h["high"] = pd.to_numeric(df_1h["high"], errors='coerce')
        df_1h["low"] = pd.to_numeric(df_1h["low"], errors='coerce')
        df_1h["volume"] = pd.to_numeric(df_1h["volume"], errors='coerce')

        df_1h = self.create_features(df_1h)
        df_1h.bfill(inplace=True)  # Use bfill() instead of fillna(method='bfill')

        if len(df_1h) < 2:
            logger.warning(f"Not enough data after feature creation for {symbol} on 1h interval")
            return None

        ema_50_1h = df_1h["ema_50"].iloc[-1]
        ema_200_1h = df_1h["ema_200"].iloc[-1]
        adx_1h = df_1h["adx"].iloc[-1]
        atr_1h = df_1h["atr"].iloc[-1]

        is_uptrend_1h = ema_50_1h > ema_200_1h and adx_1h > 25
        is_downtrend_1h = ema_50_1h < ema_200_1h and adx_1h > 25

        # Calculate dynamic stop loss using ATR
        stop_loss_1h = df_1h["close"].iloc[-1] - (atr_1h * 2) if is_uptrend_1h else df_1h["close"].iloc[-1] + (atr_1h * 2)

        # 15-minute timeframe data to determine take profit areas and entry points
        klines_data_15m = self.fetch_klines(symbol, "15m")
        if not klines_data_15m or len(klines_data_15m) < 2:
            logger.warning(f"No klines data or not enough data for {symbol} on 15m interval")
            return None

        df_15m = pd.DataFrame(klines_data_15m, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        df_15m["close"] = pd.to_numeric(df_15m["close"], errors='coerce')
        df_15m["open"] = pd.to_numeric(df_15m["open"], errors='coerce')
        df_15m["high"] = pd.to_numeric(df_15m["high"], errors='coerce')
        df_15m["low"] = pd.to_numeric(df_15m["low"], errors='coerce')
        df_15m["volume"] = pd.to_numeric(df_15m["volume"], errors='coerce')

        df_15m = self.create_features(df_15m)
        df_15m.bfill(inplace=True)  # Use bfill() instead of fillna(method='bfill')

        if len(df_15m) < 2:
            logger.warning(f"Not enough data after feature creation for {symbol} on 15m interval")
            return None

        close_price = df_15m["close"].iloc[-1]
        ema_9 = df_15m["ema_9"].iloc[-1]
        ema_21 = df_15m["ema_21"].iloc[-1]
        ema_50 = df_15m["ema_50"].iloc[-1]
        ema_200 = df_15m["ema_200"].iloc[-1]
        sma_50 = df_15m["sma_50"].iloc[-1]
        macd = df_15m["macd"].iloc[-1]
        macd_signal = df_15m["macd_signal"].iloc[-1]
        macd_hist = df_15m["macd_hist"].iloc[-1]
        rsi_7 = df_15m["rsi_7"].iloc[-1]  # Use RSI 12
        adx = df_15m["adx"].iloc[-1]
        atr = df_15m["atr"].iloc[-1]
        volume = df_15m["volume"].iloc[-1]
        support = df_15m["low"].min()
        resistance = df_15m["high"].max()
        volume_ma_20 = df_15m["volume"].rolling(window=20).mean().iloc[-1]
        previous_ema_9 = df_15m["ema_9"].iloc[-2]
        vwap = df_15m["vwap"].iloc[-1]
        bb_upper = df_15m["bb_upper"].iloc[-1]
        bb_lower = df_15m["bb_lower"].iloc[-1]

        # Calculate Pivot Points
        pivot = (df_15m["high"].iloc[-1] + df_15m["low"].iloc[-1] + df_15m["close"].iloc[-1]) / 3
        r1 = 2 * pivot - df_15m["low"].iloc[-1]
        s1 = 2 * pivot - df_15m["high"].iloc[-1]
        r2 = pivot + (df_15m["high"].iloc[-1] - df_15m["low"].iloc[-1])
        s2 = pivot - (df_15m["high"].iloc[-1] - df_15m["low"].iloc[-1])

        logger.info(f"Signal for {symbol} on 15m interval: Close Price: {close_price}, RSI 12: {rsi_7}, MACD: {macd}, MACD Signal: {macd_signal}, MACD Hist: {macd_hist}, ADX: {adx}, ATR: {atr}, Volume: {volume}, Volume MA 20: {volume_ma_20}, Support: {support}, Resistance: {resistance}, Pivot: {pivot}, R1: {r1}, S1: {s1}, R2: {r2}, S2: {s2}")

        if df_15m.isnull().any().any():
            logger.warning(f"❌ هناك قيم NaN في بيانات {symbol}، يتم تخطيها.")
            return None

        self.change_leverage(symbol, self.settings["leverage"])
        self.change_margin_mode(symbol, "ISOLATED")

        avg_volume = df_15m["volume"].rolling(window=10).mean().iloc[-1]
        if volume < avg_volume * 0.7:
            logger.warning(f"Volume for {symbol} is below 70% of the average volume. Trade not executed.")
            return None

        if volume <= volume_ma_20:
            logger.warning(f"Volume for {symbol} is below the 20-period moving average. Trade not executed.")
            return None

        if (close_price <= support * 1.01) or (close_price >= resistance * 0.99):
            logger.warning(f"Price for {symbol} is too close to support or resistance levels. Trade not executed.")
            return None

        trend_strength = adx

        signal = None
        accuracy = None
        entry = None  # Initialize entry variable
        if (
            (adx > 30) and  # وجود اتجاه قوي
            (rsi_7 >= 65) and  # RSI مرتفع يشير إلى زخم قوي
            (macd > macd_signal) and  # MACD يدعم الزخم
            (close_price > vwap) and  # السعر فوق VWAP
            (ema_9 > ema_21) and  # الاتجاه قصير الأجل صاعد
            (is_uptrend_1h)   # الاتجاه العام صاعد
            #(volume > volume_ma_20)  # حجم تداول أعلى من المتوسط
        ):
            signal = 1
            entry = close_price
            stop_loss = close_price - (atr * 2)  # Stop Loss: 2x ATR
            take_profit = close_price + (atr * 3)  # Take Profit: 3x ATR
            trailing_stop = entry - (atr * 1.5)  # Trailing Stop: 1.5x ATR
            logger.info(f"Scalping Buy signal for {symbol}")

        elif (
            (adx > 25) and  # وجود اتجاه قوي
            (rsi_7 <= 40) and  # RSI منخفض يشير إلى ضعف الزخم
            (macd < macd_signal) and  # MACD يدعم الاتجاه الهابط
            (close_price < vwap) and  # السعر تحت VWAP
            (ema_9 < ema_21) and  # الاتجاه قصير الأجل هابط
            (is_downtrend_1h) and  # الاتجاه العام هابط
            (volume > volume_ma_20)  # حجم تداول أعلى من المتوسط
        ):
            signal = -1
            entry = close_price
            stop_loss = close_price + (atr * 2)  # Stop Loss: 2x ATR
            take_profit = close_price - (atr * 3)  # Take Profit: 3x ATR
            trailing_stop = entry + (atr * 1.5)  # Trailing Stop: 1.5x ATR
            logger.info(f"Scalping Sell signal for {symbol}")


        if signal is not None:
            model, accuracy = self.train_model(df_15m)
            if accuracy is not None and accuracy < 0.20:
                logger.warning(f"Model accuracy for {symbol} is below threshold: {accuracy:.2f}")
                return None
            result = {
                "symbol": symbol,
                "signal": signal,
                "price": close_price,
                "side": "BUY" if signal == 1 else "SELL",
                "entry": round(entry, 4),
                "take_profit": round(take_profit, 4),
                "stop_loss": round(stop_loss, 4),
                "trailing_stop": round(trailing_stop, 4),
                "support": support,
                "resistance": resistance,
                "trend": "Scalping Uptrend" if signal == 1 else "Scalping Downtrend",
                "ema_9": ema_9,
                "ema_21": ema_21,
                "ema_50": ema_50,
                "ema_200": ema_200,
                "sma_50": sma_50,
                "rsi_7": rsi_7,  # Use RSI 12
                "macd": macd,
                "macd_signal": macd_signal,
                "macd_hist": macd_hist,
                "volume": float(volume),
                "adx": adx,
                "atr": atr,
                "accuracy": accuracy if accuracy is not None else 0.0
            }
            logger.debug(f"Generated signal for {symbol}: {result}")
            return result
        logger.info(f"No signal for {symbol}")
        return None

    def fetch_crypto_signals(self, interval=DEFAULT_INTERVAL, limit=3):
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching ticker data: {repr(e)}")
            return []

        data = response.json()
        logger.info(f"Fetched {len(data)} symbols from Binance Futures")

        symbols = [item["symbol"] for item in data if item["symbol"].endswith("USDT") and float(item["volume"]) > 50000
         and abs(float(item["priceChangePercent"])) > 2]
        logger.info(f"Filtered down to {len(symbols)} symbols with high volume and momentum")

        symbols = sorted(symbols, key=lambda x: float(next(item for item in data if item["symbol"] == x)["volume"]), reverse=True)[:500]
        logger.info(f"Selected top {len(symbols)} symbols by volume")

        signals = []
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(self.analyze_symbol, symbol, self.settings["interval"]): symbol for symbol in symbols}
            for future in as_completed(futures):
                result = future.result()
                if not result:
                    #logger.warning(f"No result for {futures[future]}. Skipping...")
                    continue
                if 'signal' not in result:
                    #logger.error(f"Missing 'signal' key in result for {futures[future]}. Full result: {result}")
                    continue
                logger.info(f"Checking signal for {futures[future]}: Signal={result.get('signal', 'None')}, rsi_7={result.get('rsi_7', 'None')}, MACD={result.get('macd', 'None')}, Trend={result.get('trend', 'None')}")
                signals.append(result)

        top_3_signals = sorted(signals, key=lambda x: abs(x['rsi_7']), reverse=True)[:3]  # Select top 3 signals
        logger.info(f"Found {len(top_3_signals)} suitable signals")

        return top_3_signals

    def calculate_trade_fee(self, order_amount, entry_price):
        # Assuming a trading fee of 0.1%
        fee_percentage = 0.001
        trade_fee = order_amount * entry_price * fee_percentage
        return trade_fee

    async def send_signals(self, context: ContextTypes.DEFAULT_TYPE):
        chat_id = context.job.data["chat_id"]
        interval = context.job.data.get("interval", self.settings["interval"])
        limit = context.job.data.get("limit", self.settings["limit"])
        signals = self.fetch_crypto_signals(interval, limit)

        if not signals:
            await context.bot.send_message(chat_id, "No suitable trades found currently. 🚫")
        else:
            for signal in signals:
                if signal["take_profit"] <= signal["entry"] * 1.005:  # Allow smaller profit margins
                    #logger.warning(f"No significant difference between entry and take profit for {signal['symbol']}. Trade not executed.")
                    continue

                try:
                    rsi_value = float(signal.get('rsi', 0.0))
                    formatted_rsi = f"{rsi_value:.5f}"
                except ValueError:
                    formatted_rsi = "N/A"

                general_trend = "Uptrend" if signal["trend"] == "Scalping Uptrend" else "Downtrend"

                # Calculate trade fee
                order_amount = self.calculate_quantity(signal['entry'], 0.50)
                trade_fee = self.calculate_trade_fee(order_amount, signal['entry'])

                # Calculate expected profit/loss
                expected_profit_loss = (signal['take_profit'] - signal['entry']) * order_amount - trade_fee

                reply = (
                    f"🔹 <b>Symbol</b>: {signal['symbol']}\n"
                    f"📈 <b>Side</b>: {signal['side']}\n"
                    f"💰 <b>Current Price</b>: {signal['price']:.5f}\n"
                    f"🚀 <b>Entry Price</b>: {signal['entry']:.5f}\n"
                    f"🎯 <b>Take Profit</b>: {signal['take_profit']:.5f}\n"
                    f"🛑 <b>Stop Loss</b>: {signal['stop_loss']:.5f}\n"
                    f"📉 <b>Support</b>: {signal['support']:.5f}\n"
                    f"📈 <b>Resistance</b>: {signal['resistance']:.5f}\n"
                    f"📈 <b>Trend</b>: {signal['trend']}\n"
                    f"📈 <b>General Trend</b>: {general_trend}\n"
                    f"📊 <b>EMA 9</b>: {signal['ema_9']:.5f}\n"
                    f"📊 <b>EMA 21</b>: {signal['ema_21']:.5f}\n"
                    f"📊 <b>RSI</b>: {formatted_rsi}\n"
                    f"📊 <b>MACD</b>: {signal['macd']:.5f}\n"
                    f"📊 <b>MACD Signal</b>: {signal['macd_signal']:.5f}\n"
                    f"📊 <b>MACD Hist</b>: {signal['macd_hist']:.5f}\n"
                    f"📉 <b>Trailing Stop</b>: {signal['trailing_stop']:.5f}\n"
                    f"📊 <b>Volume</b>: {signal['volume']:.5f}\n"
                    f"📊 <b>Model Accuracy</b>: {signal['accuracy']:.2f}\n"
                    f"💸 <b>Estimated Trade Fee</b>: {trade_fee:.5f} USDT\n"
                    f"📈 <b>Expected Profit/Loss</b>: {expected_profit_loss:.5f} USDT\n\n"
                )

                await context.bot.send_message(chat_id, reply, parse_mode="HTML")

                notional_value = self.calculate_quantity(signal['entry'], 0.50) * signal['entry']
                if notional_value < 5:
                    await context.bot.send_message(chat_id, f"⚠️ الصفقة على {signal['symbol']} أقل من الحد الأدنى 5 USDT. لم يتم تنفيذها.")
                    continue

                # Ensure the trade is executed
                self.execute_trade(
                    signal['symbol'],
                    signal['side'],
                    self.calculate_quantity(signal['entry'], 0.50),
                    signal['stop_loss'],
                    signal['take_profit']
                )

    async def get_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        
        await update.message.reply_text("Fetching data, please wait... ⏳")
        signals = self.fetch_crypto_signals()
        
        if not signals:
            await update.message.reply_text("No suitable trades found currently. 🚫")
        else:
            top_3_signals = sorted(signals, key=lambda x: abs(x['rsi_7']), reverse=True)[:3]  # Select top 3 signals
            available_balance = self.get_available_balance()
            if available_balance * 0.90 > available_balance:
                await update.message.reply_text("⚠️ الرصيد غير كافٍ لتنفيذ 3 صفقات بنسبة 30% لكل صفقة!")
            else:
                for signal in top_3_signals:
                    try:
                        rsi_value = float(signal.get('rsi', 0.0))
                        formatted_rsi = f"{rsi_value:.5f}"
                    except ValueError:
                        formatted_rsi = "N/A"

                    general_trend = "Uptrend" if signal["trend"] == "Scalping Uptrend" else "Downtrend"

                    # Calculate trade fee
                    order_amount = self.calculate_quantity(signal['entry'], 0.50)
                    trade_fee = self.calculate_trade_fee(order_amount, signal['entry'])

                    # Calculate expected profit/loss
                    expected_profit_loss = (signal['take_profit'] - signal['entry']) * order_amount - trade_fee

                    reply = (
                        f"🔹 <b>Symbol</b>: {signal['symbol']}\n"
                        f"📈 <b>Side</b>: {signal['side']}\n"
                        f"💰 <b>Current Price</b>: {signal['price']:.5f}\n"
                        f"🚀 <b>Entry Price</b>: {signal['entry']:.5f}\n"
                        f"🎯 <b>Take Profit</b>: {signal['take_profit']:.5f}\n"
                        f"🛑 <b>Stop Loss</b>: {signal['stop_loss']:.5f}\n"
                        f"📉 <b>Support</b>: {signal['support']:.5f}\n"
                        f"📈 <b>Resistance</b>: {signal['resistance']:.5f}\n"
                        f"📈 <b>Trend</b>: {signal['trend']}\n"
                        f"📈 <b>General Trend</b>: {general_trend}\n"
                        f"📊 <b>EMA 9</b>: {signal['ema_9']:.5f}\n"
                        f"📊 <b>EMA 21</b>: {signal['ema_21']:.5f}\n"
                        f"📊 <b>RSI</b>: {formatted_rsi}\n"
                        f"📊 <b>MACD</b>: {signal['macd']:.5f}\n"
                        f"📊 <b>MACD Signal</b>: {signal['macd_signal']:.5f}\n"
                        f"📊 <b>MACD Hist</b>: {signal['macd_hist']:.5f}\n"
                        f"📉 <b>Trailing Stop</b>: {signal['trailing_stop']:.5f}\n"
                        f"📊 <b>Volume</b>: {signal['volume']:.5f}\n"
                        f"📊 <b>Model Accuracy</b>: {signal['accuracy']:.2f}\n"
                        f"💸 <b>Estimated Trade Fee</b>: {trade_fee:.5f} USDT\n"
                        f"📈 <b>Expected Profit/Loss</b>: {expected_profit_loss:.5f} USDT\n\n"
                    )
                    await update.message.reply_text(reply, parse_mode="HTML")

                    # Ensure the trade is executed
                    self.execute_trade(
                        signal['symbol'],
                        signal['side'],
                        signal['entry'],
                        signal['stop_loss'],
                        signal['take_profit']
                    )

    async def send_message_in_chunks(self, chat_id, text, context, chunk_size=4096):
        for i in range(0, len(text), chunk_size):
            await context.bot.send_message(chat_id, text[i+i+chunk_size], parse_mode="HTML")

    async def start_sending_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
        logger.info(f"Starting to send signals for chat_id: {chat_id}")
        job_queue = context.application.job_queue
        job_queue.run_repeating(self.send_signals, interval=30, first=0, data={"chat_id": chat_id}, name=str(chat_id))  # Ensure data is a dictionary
        await context.bot.send_message(chat_id, "Started sending signals periodically every 60 seconds.")

    async def stop_sending_signals(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.message.chat_id if update.message else update.callback_query.message.chat_id
        logger.info(f"Stopping to send signals for chat_id: {chat_id}")
        current_jobs = context.application.job_queue.get_jobs_by_name(str(chat_id))
        if current_jobs:
            for job in current_jobs:
                job.schedule_removal()
            await context.bot.send_message(chat_id, "Stopped sending signals.")
        else:
            await context.bot.send_message(chat_id, "No scheduled tasks found.")

    def get_open_positions(self):
        url = "https://fapi.binance.com/fapi/v1/positionRisk"
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        signature = hmac.new(self.secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        headers = {
            "X-MBX-APIKEY": self.api_key
        }
        try:
            response = requests.get(f"{url}?{query_string}&signature={signature}", headers=headers)
            response.raise_for_status()
            positions = response.json()
            logger.debug(f"Fetched positions: {positions}")
            return positions
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching open positions: {repr(e)}")
            return None

    def calculate_trade_size(self, available_balance, percentage=1):
        trade_size = available_balance * percentage  # Allocate 35% of the available balance for each trade
        min_notional = 5.5  # Minimum notional value required for trading

        # Ensure the trade size meets the minimum notional value
        if trade_size < min_notional:
            trade_size = min_notional

        return trade_size

    def allocate_balance_for_trades(self, total_balance, num_trades):
        allocated_balance = total_balance * 1  # Allocate 20% of the total balance for trades
        per_trade_balance = allocated_balance / num_trades if num_trades > 0 else 0

        # Ensure each trade has the minimum required balance
        if per_trade_balance < 5.5:
            per_trade_balance = 5.5

        return per_trade_balance

    def execute_trade(self, symbol, side, entry_price, stop_loss, take_profit):
        try:
            # احسب الكمية مع التحقق من الحد الأدنى
            order_amount = self.calculate_quantity(entry_price, percentage=0.50)
            
            if order_amount <= 0:
                logger.error("الكمية غير صالحة!")
                return None
            
            # تحقق من القيمة الاسمية مرة أخرى (لتجنب أخطاء التقريب)
            notional_value = order_amount * entry_price
            if notional_value < 5:
                logger.error(f"القيمة الاسمية لا تزال أقل من 5: {notional_value:.2f}. تم إلغاء الصفقة.")
                return None
            
            # Execute limit order
            order = self.binance.create_order(
                symbol=symbol,
                type='limit',  # Use limit order
                side=side.lower(),
                amount=order_amount,
                price=entry_price  # Set the entry price for limit order
            )
            logger.info(f"Executed limit order: {order}")

            # Place stop loss order
            stop_loss_order = self.binance.create_order(
                symbol=symbol,
                type='stop_market',
                side='sell' if side.lower() == 'buy' else 'buy',
                amount=order_amount,
                params={'stopPrice': stop_loss}
            )
            logger.info(f"Placed stop loss order: {stop_loss_order}")

            # Ensure take profit is valid before placing the order
            if take_profit > 0:
                # Place take profit order
                take_profit_order = self.binance.create_order(
                    symbol=symbol,
                    type='take_profit_market',
                    side='sell' if side.lower() == 'buy' else 'buy',
                    amount=order_amount,
                    params={'stopPrice': take_profit}
                )
                logger.info(f"Placed take profit order: {take_profit_order}")
            else:
                logger.warning(f"Invalid take profit value for {symbol}: {take_profit}")

            # Implement trailing stop logic
            self.manage_trailing_stop(symbol, side, entry_price, order_amount)

            return order, stop_loss_order, take_profit_order
        except ccxt.BaseError as e:
            logger.error(f"Error executing trade: {repr(e)}")
            return None

    def manage_trailing_stop(self, symbol, side, entry_price, order_amount):
        try:
            while True:
                current_price = self.binance.fetch_ticker(symbol)['last']
                if side.lower() == 'buy':
                    if current_price >= entry_price * 1.015:  # Move stop loss to entry point after 1.5% gain
                        new_stop_loss = entry_price
                    elif current_price >= entry_price * 1.03:  # Move stop loss to 1% profit after 3% gain
                        new_stop_loss = entry_price * 1.01
                    else:
                        continue
                else:
                    if current_price <= entry_price * 0.985:  # Move stop loss to entry point after 1.5% gain
                        new_stop_loss = entry_price
                    elif current_price <= entry_price * 0.97:  # Move stop loss to 1% profit after 3% gain
                        new_stop_loss = entry_price * 0.99
                    else:
                        continue

                # Cancel existing stop loss order
                self.cancel_existing_order(symbol, 'stop_market')

                # Update stop loss order
                stop_loss_order = self.binance.create_order(
                    symbol=symbol,
                    type='stop_market',
                    side='sell' if side.lower() == 'buy' else 'buy',
                    amount=order_amount,
                    params={'stopPrice': new_stop_loss}
                )
                logger.info(f"Updated trailing stop loss order: {stop_loss_order}")
                break
        except ccxt.BaseError as e:
            logger.error(f"Error managing trailing stop: {repr(e)}")

    def cancel_existing_order(self, symbol, order_type):
        try:
            open_orders = self.binance.fetch_open_orders(symbol)
            for order in open_orders:
                if order['type'] == order_type:
                    self.binance.cancel_order(order['id'], symbol)
                    logger.info(f"Cancelled existing {order_type} order: {order}")
        except ccxt.BaseError as e:
            logger.error(f"Error cancelling existing order: {repr(e)}")

    def get_available_balance(self):
        if not self.api_key or not self.secret_key:
            logger.error("API key or secret key is missing. Please check your environment variables.")
            return 0

        url = "https://fapi.binance.com/fapi/v2/account"
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        signature = hmac.new(self.secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        headers = {
            "X-MBX-APIKEY": self.api_key
        }
        try:
            response = requests.get(f"{url}?{query_string}&signature={signature}", headers=headers)
            response.raise_for_status()
            account_info = response.json()
            available_balance = float(account_info["availableBalance"])
            logger.info(f"Available Balance: {available_balance}")
            if available_balance < 10:  # إذا كان الرصيد أقل من 10 USDT
                logger.warning("⚠️ الرصيد قريب من الحد الأدنى لتنفيذ الصفقات!")
            return available_balance
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching account balance: {repr(e)}")
            return 0

    def get_total_balance(self):
        if not self.api_key or not self.secret_key:
            logger.error("API key or secret key is missing. Please check your environment variables.")
            return 0

        url = "https://fapi.binance.com/fapi/v2/account"
        timestamp = int(time.time() * 1000)
        query_string = f"timestamp={timestamp}"
        signature = hmac.new(self.secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        headers = {
            "X-MBX-APIKEY": self.api_key
        }
        try:
            response = requests.get(f"{url}?{query_string}&signature={signature}", headers=headers)
            response.raise_for_status()
            account_info = response.json()
            total_balance = float(account_info["totalWalletBalance"])
            logger.info(f"Total Balance: {total_balance}")
            return total_balance
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching total balance: {repr(e)}")
            return 0

    def change_leverage(self, symbol, leverage):
        url = "https://fapi.binance.com/fapi/v1/leverage"
        timestamp = int(time.time() * 1000)
        leverage_data = {
            "symbol": symbol,
            "leverage": leverage,
            "timestamp": timestamp
        }
        query_string = "&".join([f"{key}={value}" for key, value in leverage_data.items()])
        signature = hmac.new(self.secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()
        headers = {
            "X-MBX-APIKEY": self.api_key
        }
        try:
            response = requests.post(f"{url}?{query_string}&signature={signature}", headers=headers)
            response.raise_for_status()
            leverage_response = response.json()
            logger.info(f"Changed leverage: {leverage_response}")
            return leverage_response
        except requests.exceptions.RequestException as e:
            logger.error(f"Error changing leverage: {repr(e)}")
            return None

    unsupported_margin_symbols = set()

    def change_margin_mode(self, symbol, margin_type):
        if symbol in self.unsupported_margin_symbols:
            ##logger.warning(f"Skipping margin mode change for {symbol}: Not supported by Binance.")
            return None

        try:
            url = "https://fapi.binance.com/fapi/v1/marginType"
            timestamp = int(time.time() * 1000)
            margin_data = {
                "symbol": symbol,
                "marginType": margin_type,
                "timestamp": timestamp
            }
            query_string = "&".join([f"{key}={value}" for key, value in margin_data.items()])
            signature = hmac.new(self.secret_key.encode(), query_string.encode(), hashlib.sha256).hexdigest()
            headers = {"X-MBX-APIKEY": self.api_key}

            response = requests.post(f"{url}?{query_string}&signature={signature}", headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if "400" in str(e):
                #logger.warning(f"Margin mode change not supported for {symbol}. Adding to ignore list.")
                self.unsupported_margin_symbols.add(symbol)  # Add symbol to the list to prevent future attempts
                return None
            else:
                logger.error(f"Error changing margin mode: {repr(e)}")
            return None

    def calculate_quantity(self, entry_price, percentage=0.50):
        available_balance = self.get_available_balance()
        if available_balance <= 0:
            return 0
        
        # حساب الكمية بناءً على النسبة والرافعة
        trade_size = (available_balance * percentage) * self.settings["leverage"]
        order_amount = trade_size / entry_price
        
        # التحقق من الحد الأدنى للقيمة الاسمية (5 USDT)
        min_notional = 5.0
        notional_value = order_amount * entry_price
        
        if (notional_value < min_notional):
            # إذا كانت القيمة أقل من 5، زيادة الكمية لتلبية الحد الأدنى
            required_amount = (min_notional / entry_price)
            logger.warning(f"الكمية المعدلة لتلبية الحد الأدنى: {required_amount:.5f}")
            return required_amount
        
        return order_amount

    def backtest_strategy(self, symbol, interval, days=30):
        klines = self.fetch_klines(symbol, interval, limit=days*24)  # افتراض 24 شمعة يومياً
        if not klines:
            logger.error(f"Failed to fetch klines for backtesting {symbol} on {interval} interval.")
            return

        df = pd.DataFrame(klines, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        df["close"] = pd.to_numeric(df["close"], errors='coerce')
        df["open"] = pd.to_numeric(df["open"], errors='coerce')
        df["high"] = pd.to_numeric(df["high"], errors='coerce')
        df["low"] = pd.to_numeric(df["low"], errors='coerce')
        df["volume"] = pd.to_numeric(df["volume"], errors='coerce')

        df = self.create_features(df)
        df.bfill(inplace=True)  # Use bfill() instead of fillna(method='bfill')

        if len(df) < 2:
            logger.warning(f"Not enough data after feature creation for {symbol} on {interval} interval")
            return

        signals = self.generate_historical_signals(df)
        profit = self.calculate_profit(signals)
        logger.info(f"العائد التاريخي لـ {symbol} على فترة {days} يوم: {profit:.2f}%")

    def generate_historical_signals(self, df):
        signals = []
        for i in range(1, len(df)):
            if self.is_uptrend(df["ema_9"].iloc[i], df["ema_21"].iloc[i], df["ema_50"].iloc[i], df["ema_200"].iloc[i], df["ema_9"].iloc[i-1]):
                signals.append({"type": "buy", "price": df["close"].iloc[i]})
            elif self.is_downtrend(df["ema_9"].iloc[i], df["ema_21"].iloc[i], df["ema_50"].iloc[i], df["ema_200"].iloc[i], df["ema_9"].iloc[i-1]):
                signals.append({"type": "sell", "price": df["close"].iloc[i]})
        return signals

    def calculate_profit(self, signals):
        profit = 0
        position = None
        for signal in signals:
            if signal["type"] == "buy" and position is None:
                position = signal["price"]
            elif signal["type"] == "sell" and position is not None:
                profit += (signal["price"] - position) / position * 100
                position = None
        return profit