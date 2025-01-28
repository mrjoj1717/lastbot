import requests
import pandas as pd
import pandas_ta as ta
import backtrader as bt
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler
import logging

# إعداد سجل الأخطاء
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# توكن البوت
BOT_TOKEN = "8139823264:AAGK947IH6riOFNti4QOEklBLgoxXzNDcXQ"

# معرف القناة الفريد
CHANNEL_ID = "@Future_Deals"

# تعريف استراتيجية التداول باستخدام Backtrader
class TradingStrategy(bt.Strategy):
    params = (
        ('ema_fast', 9),
        ('ema_slow', 21),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
    )

    def __init__(self):
        self.ema_fast = bt.indicators.EMA(self.data.close, period=self.params.ema_fast)
        self.ema_slow = bt.indicators.EMA(self.data.close, period=self.params.ema_slow)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

    def next(self):
        if self.ema_fast > self.ema_slow and self.rsi < self.params.rsi_oversold:
            self.buy()
        elif self.ema_fast < self.ema_slow and self.rsi > self.params.rsi_overbought:
            self.sell()

# دالة لإجراء الاختبار التاريخي
def backtest_strategy(dataframe, strategy):
    cerebro = bt.Cerebro()
    dataframe["datetime"] = pd.to_datetime(dataframe["time"], unit='ms')
    dataframe.set_index("datetime", inplace=True)
    data = bt.feeds.PandasData(dataname=dataframe)
    cerebro.adddata(data)
    cerebro.addstrategy(strategy)
    cerebro.run()
    cerebro.plot()

# دالة لتحليل البيانات باستخدام المؤشرات الفنية
def analyze_symbol(symbol):
    logger.info(f"Analyzing symbol: {symbol}")

    klines_url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=100"
    try:
        response = requests.get(klines_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

    klines_data = response.json()
    if not klines_data:
        logger.warning(f"No data returned for {symbol}")
        return None

    df = pd.DataFrame(klines_data, columns=[
        "time", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore"
    ])
    df["close"] = pd.to_numeric(df["close"])
    df["open"] = pd.to_numeric(df["open"])
    df["high"] = pd.to_numeric(df["high"])
    df["low"] = pd.to_numeric(df["low"])
    df["volume"] = pd.to_numeric(df["volume"])

    # إجراء اختبار Backtrader
    logger.info(f"Running backtest for {symbol}")
    backtest_strategy(df, TradingStrategy)

    # إضافة مؤشرات فنية
    df["ema_9"] = ta.ema(df["close"], length=9)
    df["ema_21"] = ta.ema(df["close"], length=21)
    df["rsi"] = ta.rsi(df["close"], length=14)

    # إشارات التداول
    close_price = df["close"].iloc[-1]
    if df["ema_9"].iloc[-1] > df["ema_21"].iloc[-1] and df["rsi"].iloc[-1] < 55:
        return {"symbol": symbol, "price": close_price, "side": "شراء 🟢"}
    elif df["ema_9"].iloc[-1] < df["ema_21"].iloc[-1] and df["rsi"].iloc[-1] > 45:
        return {"symbol": symbol, "price": close_price, "side": "بيع 🔴"}
    return None

# دالة لجلب إشارات التداول
def fetch_crypto_signals():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching ticker data: {e}")
        return []

    data = response.json()
    symbols = [item["symbol"] for item in data if item["symbol"].endswith("USDT")]

    signals = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(analyze_symbol, symbols)
        for result in results:
            if result:
                signals.append(result)

    return signals[:5]

# دوال Telegram Bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("بدء إرسال التوصيات", callback_data='start_sending_signals'),
            InlineKeyboardButton("إيقاف إرسال التوصيات", callback_data='stop_sending_signals'),
        ],
        [InlineKeyboardButton("جلب التوصيات الآن", callback_data='get_signals')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("مرحبًا! استخدم الأزرار أدناه للتحكم في البوت:", reply_markup=reply_markup)

async def get_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("جاري جلب البيانات، الرجاء الانتظار... ⏳")
    signals = fetch_crypto_signals()
    if not signals:
        await update.message.reply_text("لم أجد أي صفقات مناسبة حاليًا. 🚫")
    else:
        reply = "📊 أفضل التوصيات:\n\n"
        for signal in signals:
            reply += f"🔹 العملة: {signal['symbol']} | الجانب: {signal['side']} | السعر: {signal['price']:.4f}\n"
        await update.message.reply_text(reply)

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == 'get_signals':
        await get_signals(query, context)

# تشغيل البوت
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(button))
app.run_polling()
