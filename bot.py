from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, CallbackQueryHandler, JobQueue, ConversationHandler, MessageHandler, filters
import logging
import os
from dotenv import load_dotenv
import telegram
from bot_logic import CryptoBot
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# تحميل المتغيرات من ملف .env
load_dotenv()

# إعداد سجل الأخطاء
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levellevel)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# توكن البوت
BOT_TOKEN = os.getenv("BOT_TOKEN")

# تشغيل البوت باستخدام ngrok
NGROK_URL = os.getenv("NGROK_URL")

# قائمة المستخدمين المصرح لهم
AUTHORIZED_USERS = os.getenv("AUTHORIZED_USERS", "").split(',')

# إنشاء كائن البوت
crypto_bot = CryptoBot()

# Default settings
DEFAULT_INTERVAL = "15m"  # تقليل الفاصل الزمني إلى دقيقة واحدة
DEFAULT_LIMIT = 5  # زيادة عدد الصفقات المفتوحة إلى 5
DEFAULT_MODEL = "random_forest"
TRAILING_STOP_PERCENT = 0.01  # تقليل نسبة وقف الخسارة المتحرك إلى 1%
DEFAULT_LEVERAGE = 1  # زيادة الرافعة المالية إلى 10

# دالة التحقق من المستخدم
def is_authorized(user_id):
    return str(user_id) in AUTHORIZED_USERS

# دالة الترحيب
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.message.from_user.id):
        await update.message.reply_text("عذرًا، ليس لديك إذن لاستخدام هذا البوت.")
        return

    keyboard = [
        [
            InlineKeyboardButton("بدء إرسال التوصيات", callback_data='start_sending_signals'),
            InlineKeyboardButton("إيقاف إرسال التوصيات", callback_data='stop_sending_signals'),
        ],
        [InlineKeyboardButton("جلب التوصيات الآن", callback_data='get_signals')],
        [InlineKeyboardButton("تغيير الإعدادات", callback_data='change_settings')],
        [InlineKeyboardButton("عرض الصفقات المفتوحة", callback_data='show_open_positions')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("مرحبًا! استخدم الأزرار أدناه للتحكم في البوت:", reply_markup=reply_markup)

# دالة لمعالجة الأزرار التفاعلية
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    try:
        await query.answer()
    except telegram.error.BadRequest as e:
        if "query is too old" in str(e):
            await query.message.reply_text("عذرًا، انتهت صلاحية هذا الطلب. يرجى المحاولة مرة أخرى.")
        else:
            logger.error(f"Failed to answer callback query: {e}")
        return

    if not is_authorized(query.from_user.id):
        await query.message.reply_text("عذرًا، ليس لديك إذن لاستخدام هذا البوت.")
        return

    if query.data == 'start_sending_signals':
        await crypto_bot.start_sending_signals(query, context)
    elif query.data == 'stop_sending_signals':
        await crypto_bot.stop_sending_signals(query, context)
    elif query.data == 'get_signals':
        await crypto_bot.get_signals(query, context)
    elif query.data == 'change_settings':
        await change_settings(query, context)
    elif query.data == 'show_open_positions':
        await show_open_positions(update, context)
    elif query.data == 'main_menu':
        await start(query, context)

# دالة لتغيير الإعدادات
async def change_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.callback_query.from_user.id):
        await update.callback_query.message.reply_text("عذرًا، ليس لديك إذن لاستخدام هذا البوت.")
        return

    keyboard = [
        [InlineKeyboardButton("تغيير الفاصل الزمني", callback_data='change_interval')],
        [InlineKeyboardButton("تغيير الحد", callback_data='change_limit')],
        [InlineKeyboardButton("تغيير الرموز", callback_data='change_symbols')],
        [InlineKeyboardButton("العودة إلى القائمة الرئيسية", callback_data='main_menu')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.reply_text("اختر الإعداد الذي تريد تغييره:", reply_markup=reply_markup)

# دالة لتغيير الفاصل الزمني
async def change_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["setting_type"] = "INTERVAL"
    await update.callback_query.message.reply_text("أدخل الفاصل الزمني الجديد (مثل 15m، 1h، 4h):")
    return "INTERVAL"

# دالة لتغيير الحد
async def change_limit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["setting_type"] = "LIMIT"
    await update.callback_query.message.reply_text("أدخل الحد الجديد:")
    return "LIMIT"

# دالة لتغيير الرموز
async def change_symbols(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["setting_type"] = "SYMBOLS"
    await update.callback_query.message.reply_text("أدخل الرموز الجديدة مفصولة بفواصل (مثل BTCUSDT، ETHUSDT):")
    return "SYMBOLS"

# دالة لحفظ الإعدادات
async def save_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if not is_authorized(user_id):
        await update.message.reply_text("عذرًا، ليس لديك إذن لاستخدام هذا البوت.")
        return ConversationHandler.END

    setting_type = context.user_data.get("setting_type")
    value = update.message.text

    if setting_type == "INTERVAL":
        crypto_bot.settings["interval"] = value
    elif setting_type == "LIMIT":
        crypto_bot.settings["limit"] = int(value)
    elif setting_type == "SYMBOLS":
        crypto_bot.settings["symbols"] = value.split(',')

    with open(f"settings_{user_id}.json", "w") as f:
        json.dump(crypto_bot.settings, f)

    await update.message.reply_text("تم حفظ الإعدادات بنجاح.")
    return ConversationHandler.END

# دالة لعرض الصفقات المفتوحة
async def show_open_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update.callback_query.from_user.id):
        await update.callback_query.message.reply_text("عذرًا، ليس لديك إذن لاستخدام هذا البوت.")
        return

    open_positions = crypto_bot.get_open_positions()
    if open_positions is None:
        await update.callback_query.message.reply_text("حدث خطأ أثناء جلب الصفقات المفتوحة.")
        return

    if not open_positions:
        await update.callback_query.message.reply_text("لا توجد صفقات مفتوحة حاليًا.")
    else:
        reply = "📊 الصفقات المفتوحة:\n\n"
        for position in open_positions:
            reply += (
                f"🔹 <b>العملة</b>: {position['symbol']}\n"
                f"📈 <b>الجانب</b>: {position['side']}\n"
                f"💰 <b>السعر الحالي</b>: {position['price']:.5f}\n"
                f"📉 <b>الربح/الخسارة</b>: {position['pnl']:.5f}\n\n"
            )
        await update.callback_query.message.reply_text(reply, parse_mode="HTML")

# دالة لمعالجة الأخطاء
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    if update and update.effective_message:
        await update.effective_message.reply_text("حدث خطأ غير متوقع. يرجى المحاولة مرة أخرى لاحقًا.")
    if update:
        logger.error(f"Update: {update}")
    if context.error:
        logger.error(f"Context Error: {context.error}")
    else:
        logger.error("Context Error: No error information available")

app = ApplicationBuilder().token(BOT_TOKEN).build()

# إعداد محادثة لتغيير الإعدادات
conv_handler = ConversationHandler(
    entry_points=[CallbackQueryHandler(change_interval, pattern='^change_interval$'),
                  CallbackQueryHandler(change_limit, pattern='^change_limit$'),
                  CallbackQueryHandler(change_symbols, pattern='^change_symbols$')],
    states={
        "INTERVAL": [MessageHandler(filters.TEXT & ~filters.COMMAND, save_settings)],
        "LIMIT": [MessageHandler(filters.TEXT & ~filters.COMMAND, save_settings)],
        "SYMBOLS": [MessageHandler(filters.TEXT & ~filters.COMMAND, save_settings)],
    },
    fallbacks=[],
)

app.add_handler(CommandHandler("start", start))
app.add_handler(CallbackQueryHandler(button))
app.add_handler(CommandHandler("get_signals", crypto_bot.get_signals))
app.add_handler(CommandHandler("start_sending_signals", crypto_bot.start_sending_signals))
app.add_handler(CommandHandler("stop_sending_signals", crypto_bot.stop_sending_signals))
app.add_handler(conv_handler)
app.add_error_handler(error_handler)

app.run_webhook(
    listen="0.0.0.0",
    port=8443,
    url_path="webhook",
    webhook_url=f"{NGROK_URL}/webhook"
)

