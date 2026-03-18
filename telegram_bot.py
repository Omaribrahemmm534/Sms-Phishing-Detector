"""
SMS Phishing Detection — Telegram Bot
======================================
بوت تيليجرام يحلل أي رسالة بيبعتها المستخدم.

المتطلبات:
    pip install python-telegram-bot requests

إعداد:
    1. ابعت message لـ @BotFather على تيليجرام وعمل بوت جديد
    2. خد الـ Token وحطه في TELEGRAM_TOKEN أدناه
    3. تأكد الـ API شغّال على localhost:8000
    4. شغّل: python telegram_bot.py
"""

import os
import requests
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    CallbackQueryHandler, filters, ContextTypes
)

# ─── إعداد ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8668299543:AAEwlM-iGq_xD0cZSqnHMEtlxnfCfWiKWzQ")
API_URL        = os.getenv("API_URL", "http://localhost:8000/predict")

RISK_EMOJI = {
    "high":   "🔴",
    "medium": "🟡",
    "low":    "🟢",
}

# ─── Handlers ────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """رسالة الترحيب."""
    await update.message.reply_text(
        "👋 مرحباً! أنا بوت كشف رسائل التصيد الاحتيالي 🛡\n\n"
        "📩 ابعتلي أي رسالة مشبوهة وهقولك هي:\n"
        "  🟢 آمنة — أو\n"
        "  🔴 احتيالية (Phishing)\n\n"
        "الأوامر المتاحة:\n"
        "/start — رسالة الترحيب\n"
        "/help  — شرح الاستخدام\n"
        "/stats — إحصائيات جلستك\n"
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 كيفية الاستخدام:\n\n"
        "1️⃣ انسخ الرسالة المشبوهة\n"
        "2️⃣ ابعتها هنا مباشرة\n"
        "3️⃣ هيرجعلك تقرير فوري\n\n"
        "⚠️ تنبيه: النتائج إرشادية، استخدم حكمك دايماً."
    )

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """إحصائيات المستخدم في الجلسة الحالية."""
    data = context.user_data
    total   = data.get("total", 0)
    phish   = data.get("phishing", 0)
    safe    = data.get("safe", 0)
    await update.message.reply_text(
        f"📊 إحصائياتك:\n"
        f"  الرسائل الكلية : {total}\n"
        f"  احتيالية       : {phish} 🔴\n"
        f"  آمنة           : {safe} 🟢"
    )

async def analyze_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يحلل الرسالة المرسلة."""
    text = update.message.text
    if not text:
        return

    # أرسل "جارٍ التحليل..."
    thinking = await update.message.reply_text("🔍 جارٍ تحليل الرسالة...")

    try:
        resp = requests.post(
            API_URL,
            json={"text": text, "source": "telegram"},
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()

        emoji    = RISK_EMOJI.get(result["risk_level"], "⚪")
        label    = "احتيالية ⚠️" if result["is_phishing"] else "آمنة ✅"
        conf_pct = int(result["confidence"] * 100)
        feats    = result.get("features", {})

        # بناء الرسالة
        reply = (
            f"{emoji} *نتيجة التحليل*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"📌 التصنيف   : *{label}*\n"
            f"📊 الثقة     : *{conf_pct}%*\n"
            f"⚠️ مستوى الخطر: *{result['risk_level'].upper()}*\n"
            f"━━━━━━━━━━━━━━━━━━━\n"
            f"🔗 يحتوي URL  : {'نعم' if feats.get('has_url') else 'لا'}\n"
            f"📞 يحتوي رقم  : {'نعم' if feats.get('has_phone') else 'لا'}\n"
            f"🚨 كلمات مشبوهة: {feats.get('keyword_count', 0)}\n"
        )

        # أزرار
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ صحيح", callback_data=f"fb_correct"),
            InlineKeyboardButton("❌ خطأ",  callback_data=f"fb_wrong"),
        ]])

        await thinking.edit_text(reply, parse_mode="Markdown",
                                 reply_markup=keyboard)

        # تحديث الإحصائيات
        context.user_data["total"]    = context.user_data.get("total", 0) + 1
        if result["is_phishing"]:
            context.user_data["phishing"] = context.user_data.get("phishing", 0) + 1
        else:
            context.user_data["safe"]     = context.user_data.get("safe", 0) + 1

    except requests.exceptions.ConnectionError:
        await thinking.edit_text(
            "❌ تعذر الاتصال بالـ API.\n"
            "تأكد إن الـ Backend شغّال على port 8000."
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        await thinking.edit_text("❌ حصل خطأ أثناء التحليل، جرب تاني.")


async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """يستقبل ردود فعل المستخدم على النتائج."""
    query = update.callback_query
    await query.answer()
    if query.data == "fb_correct":
        await query.edit_message_reply_markup(None)
        await query.message.reply_text("✅ شكراً على التأكيد! هيساعدنا نحسّن النموذج.")
    else:
        await query.edit_message_reply_markup(None)
        await query.message.reply_text(
            "🙏 شكراً على التصحيح!\n"
            "لو عندك وقت، ابعتلي النص الصح يدويًا."
        )


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if TELEGRAM_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("[!] حط الـ Telegram Token في TELEGRAM_TOKEN أو في متغير البيئة")
        return

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start",  start))
    app.add_handler(CommandHandler("help",   help_cmd))
    app.add_handler(CommandHandler("stats",  stats))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_message))
    app.add_handler(CallbackQueryHandler(feedback_callback))

    print("[✓] البوت شغّال... (Ctrl+C للإيقاف)")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
