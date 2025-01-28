from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler, JobQueue
import requests
import pandas as pd
import pandas_ta as ta
from concurrent.futures import ThreadPoolExecutor
import time
import matplotlib.pyplot as plt
import logging

# إعداد سجل الأخطاء
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levellevel)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# توكن البوت
BOT_TOKEN = "8139823264:AAGK947IH6riOFNti4QOEklBLgoxXzNDcXQ"

# معرف القناة الفريد
CHANNEL_ID = "@Future_Deals"

# دالة لجلب بيانات الشموع وتحليلها لكل زوج
def analyze_symbol(symbol):
    logger.info(f"Analyzing symbol: {symbol}")

    # جلب بيانات فريم 15 دقيقة لاكتشاف الصفقات وتحديد نقاط الدخول والخروج
    klines_url_15m = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=15m&limit=100"
    try:
        klines_response_15m = requests.get(klines_url_15m)
        klines_response_15m.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching 15m data for {symbol}: {e}")
        return None

    klines_data_15m = klines_response_15m.json()
    if not klines_data_15m:
        logger.warning(f"No data returned for {symbol}")
        return None

    df_15m = pd.DataFrame(klines_data_15m, columns=["time", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
    df_15m["close"] = pd.to_numeric(df_15m["close"])
    df_15m["high"] = pd.to_numeric(df_15m["high"])
    df_15m["low"] = pd.to_numeric(df_15m["low"])

    # إضافة المؤشرات الفنية
    df_15m["ema_9"] = ta.ema(df_15m["close"], length=9)
    df_15m["ema_21"] = ta.ema(df_15m["close"], length=21)
    df_15m["rsi"] = ta.rsi(df_15m["close"], length=14)
    df_15m["volume"] = df_15m["volume"]
    
    # حساب مؤشر بولينجر باندز
    bbands = ta.bbands(df_15m["close"], length=20, std=2)
    df_15m["bollinger_upper"] = bbands["BBU_20_2.0"]
    df_15m["bollinger_middle"] = bbands["BBM_20_2.0"]
    df_15m["bollinger_lower"] = bbands["BBL_20_2.0"]

    # حساب مؤشر MACD
    macd = ta.macd(df_15m["close"], fast=12, slow=26, signal=9)
    df_15m["macd"] = macd["MACD_12_26_9"]
    df_15m["macd_signal"] = macd["MACDs_12_26_9"]
    df_15m["macd_hist"] = macd["MACDh_12_26_9"]

    close_price = df_15m["close"].iloc[-1]

    # طباعة القيم المستخدمة في التحليل لتصحيح الأخطاء
    logger.info(f"Symbol: {symbol}, EMA9: {df_15m['ema_9'].iloc[-1]}, EMA21: {df_15m['ema_21'].iloc[-1]}, RSI: {df_15m['rsi'].iloc[-1]}, Close: {close_price}, BB Lower: {df_15m['bollinger_lower'].iloc[-1]}, BB Upper: {df_15m['bollinger_upper'].iloc[-1]}")

    # تحديد مستويات الدعم والمقاومة على فريم 15 دقيقة
    support_15m = df_15m["low"].min()
    resistance_15m = df_15m["high"].max()

    # تحديد الصفقات بناءً على المؤشرات
    if df_15m["ema_9"].iloc[-1] > df_15m["ema_21"].iloc[-1] and df_15m["rsi"].iloc[-1] < 55 and close_price <= df_15m["bollinger_lower"].iloc[-1]:
        entry = close_price
        stop_loss = support_15m - (support_15m * 0.01)  # إيقاف الخسارة بعد الدعم
        take_profit1 = entry + (resistance_15m - entry) * 0.33  # الهدف الأول
        take_profit2 = entry + (resistance_15m - entry) * 0.66  # الهدف الثاني
        take_profit3 = resistance_15m - (resistance_15m * 0.01)  # الهدف الثالث
        logger.info(f"Buy signal for {symbol}")
        return {
            "symbol": symbol,
            "price": close_price,
            "side": "شراء 🟢",
            "entry": round(entry, 4),
            "take_profit1": round(take_profit1, 4),
            "take_profit2": round(take_profit2, 4),
            "take_profit3": round(take_profit3, 4),
            "stop_loss": round(stop_loss, 4),
            "support": support_15m,
            "resistance": resistance_15m,
            "trend": "صاعد 📈"
        }
    elif df_15m["ema_9"].iloc[-1] < df_15m["ema_21"].iloc[-1] and df_15m["rsi"].iloc[-1] > 45 and close_price >= df_15m["bollinger_upper"].iloc[-1]:
        entry = close_price
        stop_loss = resistance_15m + (resistance_15m * 0.01)  # إيقاف الخسارة بعد المقاومة
        take_profit1 = entry - (entry - support_15m) * 0.33  # الهدف الأول
        take_profit2 = entry - (entry - support_15m) * 0.66  # الهدف الثاني
        take_profit3 = support_15m + (support_15m * 0.01)  # الهدف الثالث
        logger.info(f"Sell signal for {symbol}")
        return {
            "symbol": symbol,
            "price": close_price,
            "side": "بيع 🔴",
            "entry": round(entry, 4),
            "take_profit1": round(take_profit1, 4),
            "take_profit2": round(take_profit2, 4),
            "take_profit3": round(take_profit3, 4),
            "stop_loss": round(stop_loss, 4),
            "support": support_15m,
            "resistance": resistance_15m,
            "trend": "هابط 📉"
        }
    logger.info(f"No signal for {symbol}")
    return None

# دالة لجلب بيانات الشموع وتحليلها لجميع الأزواج
def fetch_crypto_signals():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching ticker data: {e}")
        return []

    data = response.json()

    symbols = [item["symbol"] for item in data if item["symbol"].endswith("USDT") and float(item["volume"]) > 100000 and abs(float(item["priceChangePercent"])) > 2]

    signals = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(analyze_symbol, symbols)
        for result in results:
            if result:
                signals.append(result)

    # اختيار أفضل 5 صفقات
    signals = signals[:5]

    return signals

# دالة لإرسال التوصيات بشكل دوري
async def send_signals(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.data
    signals = fetch_crypto_signals()

    if not signals:
        await context.bot.send_message(chat_id, "لم أجد أي صفقات مناسبة حاليًا. 🚫")
    else:
        reply = "📊 أفضل التوصيات:\n\n"
        for signal in signals:
            reply += (
                f"🔹 <b>العملة</b>: {signal['symbol']}\n"
                f"📈 <b>الجانب</b>: {signal['side']}\n"
                f"💰 <b>السعر الحالي</b>: {signal['price']:.4f}\n"
                f"🚀 <b>سعر الدخول</b>: {signal['entry']:.4f}\n"
                f"🎯 <b>أخذ الربح 1</b>: {signal['take_profit1']:.4f}\n"
                f"🎯 <b>أخذ الربح 2</b>: {signal['take_profit2']:.4f}\n"
                f"🎯 <b>أخذ الربح 3</b>: {signal['take_profit3']:.4f}\n"
                f"🛑 <b>إيقاف الخسارة</b>: {signal['stop_loss']:.4f}\n"
                f"📉 <b>الدعم</b>: {signal['support']:.4f}\n"
                f"📈 <b>المقاومة</b>: {signal['resistance']:.4f}\n"
                f"📈 <b>الاتجاه العام</b>: {signal['trend']}\n\n"
            )
            # إنشاء الرسم البياني وحفظه كصورة
            plt.figure(figsize=(10, 5))
            plt.plot(signal['price'], label='Price')
            plt.axhline(y=signal['entry'], color='g', linestyle='--', label='Entry')
            plt.axhline(y=signal['take_profit1'], color='b', linestyle='--', label='Take Profit 1')
            plt.axhline(y=signal['take_profit2'], color='b', linestyle='--', label='Take Profit 2')
            plt.axhline(y=signal['take_profit3'], color='b', linestyle='--', label='Take Profit 3')
            plt.axhline(y=signal['stop_loss'], color='r', linestyle='--', label='Stop Loss')
            plt.legend()
            plt.title(f"{signal['symbol']} - {signal['side']}")
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.grid(True)
            plt.savefig(f"{signal['symbol']}.png")
            plt.close()

            # إرسال الصورة عبر Telegram
            with open(f"{signal['symbol']}.png", 'rb') as photo:
                await context.bot.send_photo(chat_id, photo)

        await send_message_in_chunks(chat_id, reply, context)

# دالة الترحيب
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

# دالة جلب التوصيات
async def get_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("جاري جلب البيانات، الرجاء الانتظار... ⏳")
    signals = fetch_crypto_signals()

    if not signals:
        await update.message.reply_text("لم أجد أي صفقات مناسبة حاليًا. 🚫")
    else:
        reply = "📊 أفضل التوصيات:\n\n"
        for signal in signals:
            reply += (
                f"🔹 <b>العملة</b>: {signal['symbol']}\n"
                f"📈 <b>الجانب</b>: {signal['side']}\n"
                f"💰 <b>السعر الحالي</b>: {signal['price']:.4f}\n"
                f"🚀 <b>سعر الدخول</b>: {signal['entry']:.4f}\n"
                f"🎯 <b>أخذ الربح 1</b>: {signal['take_profit1']:.4f}\n"
                f"🎯 <b>أخذ الربح 2</b>: {signal['take_profit2']:.4f}\n"
                f"🎯 <b>أخذ الربح 3</b>: {signal['take_profit3']:.4f}\n"
                f"🛑 <b>إيقاف الخسارة</b>: {signal['stop_loss']:.4f}\n"
                f"📉 <b>الدعم</b>: {signal['support']:.4f}\n"
                f"📈 <b>المقاومة</b>: {signal['resistance']:.4f}\n"
                f"📈 <b>الاتجاه العام</b>: {signal['trend']}\n\n"
            )
            # إنشاء الرسم البياني وحفظه كصورة
            plt.figure(figsize=(10, 5))
            plt.plot(signal['price'], label='Price')
            plt.axhline(y=signal['entry'], color='g', linestyle='--', label='Entry')
            plt.axhline(y=signal['take_profit1'], color='b', linestyle='--', label='Take Profit 1')
            plt.axhline(y=signal['take_profit2'], color='b', linestyle='--', label='Take Profit 2')
            plt.axhline(y=signal['take_profit3'], color='b', linestyle='--', label='Take Profit 3')
            plt.axhline(y=signal['stop_loss'], color='r', linestyle='--', label='Stop Loss')
            plt.legend()
            plt.title(f"{signal['symbol']} - {signal['side']}")
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.grid(True)
            plt.savefig(f"{signal['symbol']}.png")
            plt.close()

            # إرسال الصورة عبر Telegram
            with open(f"{signal['symbol']}.png", 'rb') as photo:
                await context.bot.send_photo(update.message.chat_id, photo)

        await send_message_in_chunks(update.message.chat_id, reply, context)

        # طباعة الصفقات في التيرمينال
        logger.info("📊 أفضل التوصيات:\n")
        for signal in signals:
            logger.info(
                f"🔹 <b>العملة</b>: {signal['symbol']}\n"
                f"📈 <b>الجانب</b>: {signal['side']}\n"
                f"💰 <b>السعر الحالي</b>: {signal['price']:.4f}\n"
                f"🚀 <b>سعر الدخول</b>: {signal['entry']:.4f}\n"
                f"🎯 <b>أخذ الربح 1</b>: {signal['take_profit1']:.4f}\n"
                f"🎯 <b>أخذ الربح 2</b>: {signal['take_profit2']:.4f}\n"
                f"🎯 <b>أخذ الربح 3</b>: {signal['take_profit3']:.4f}\n"
                f"🛑 <b>إيقاف الخسارة</b>: {signal['stop_loss']:.4f}\n"
                f"📉 <b>الدعم</b>: {signal['support']:.4f}\n"
                f"📈 <b>المقاومة</b>: {signal['resistance']}\n"
                f"📈 <b>الاتجاه العام</b>: {signal['trend']}\n"
            )

# دالة لإرسال الرسائل على دفعات
async def send_message_in_chunks(chat_id, text, context, chunk_size=4096):
    for i in range(0, len(text), chunk_size):
        await context.bot.send_message(chat_id, text[i:i+chunk_size], parse_mode="HTML")

# دالة لبدء الجدولة
async def start_sending_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    job_queue = context.application.job_queue
    job_queue.run_repeating(send_signals, interval=30, first=0, data=chat_id, name=str(chat_id))  # كل 15 ثانية
    await update.message.reply_text("تم بدء إرسال التوصيات بشكل دوري كل 30 ثانية.")

# دالة لإيقاف الجدولة
async def stop_sending_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    current_jobs = context.application.job_queue.get_jobs_by_name(str(update.message.chat_id))
    if current_jobs:
        for job in current_jobs:
            job.schedule_removal()
        await update.message.reply_text("تم إيقاف إرسال التوصيات.")
    else:
        await update.message.reply_text("لا توجد مهام مجدولة.")

# دالة لمعالجة الأزرار التفاعلية
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == 'start_sending_signals':
        await start_sending_signals(query, context)
    elif query.data == 'stop_sending_signals':
        await stop_sending_signals(query, context)
    elif query.data == 'get_signals':
        await get_signals(query, context)

# تشغيل البوت
app = ApplicationBuilder().token(BOT_TOKEN).build()
app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(button))
app.add_handler(CommandHandler("get_signals", get_signals))
app.add_handler(CommandHandler("start_sending_signals", start_sending_signals))
app.add_handler(CommandHandler("stop_sending_signals", stop_sending_signals))

app.run_polling()