import os
import json
import logging

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in environment")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

client = OpenAI(api_key=OPENAI_API_KEY)

START_TEXT = "Привет. Я помогаю отличать инсайты от наблюдений, фактов и просто красиво звучащих мыслей. Пришли одну мысль — я скажу, что это и почему."

SYSTEM_PROMPT = """
Ты — строгий классификатор текстов.

Твоя главная задача: определить, является ли текст инсайтом или нет.

ОТВЕЧАЙ ТОЛЬКО В ДВУХ ПОЛЯХ:
1) is_insight: true или false
2) why: короткое объяснение на русском языке

ПРАВИЛА:
- Инсайт — это только такой текст, где есть явный внутренний человеческий конфликт, парадокс, скрытый мотив или неожиданное открытие.
- Если фраза просто описывает предпочтение, тенденцию, удобство, экономию, выбор, стиль жизни или краткое ценностное утверждение — это НЕ инсайт.
- Короткие формулировки без явного конфликта не считать инсайтом, даже если они звучат глубоко.
- Инсайт нельзя выводить только из противопоставления двух вариантов, если в тексте нет явного внутреннего напряжения.

ОСОБЫЕ ПРАВИЛА ДЛЯ СПОРНЫХ СЛУЧАЕВ:
- "Ограниченность предложения усиливает желание купить" — НЕ инсайт.
- "Маленькие радости вместо больших трат" — НЕ инсайт.
- "Возврат к локальным брендам" — НЕ инсайт.
- "Потребитель часто выбирает не самый дешёвый вариант, а самый удобный" — НЕ инсайт.
- Если мысль выглядит как лозунг, слоган, наблюдение или обобщение без внутреннего конфликта, отвечай false.

ФОРМАТ ОТВЕТА:
{
  "is_insight": true или false,
  "why": "короткое объяснение"
}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "is_insight": {"type": "boolean"},
        "why": {"type": "string"},
    },
    "required": ["is_insight", "why"],
    "additionalProperties": False,
}


def call_model(user_text: str) -> dict:
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "insight_eval",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )
    raw = response.output_text
    return json.loads(raw)


def humanize(result: dict) -> str:
    is_insight = result.get("is_insight")
    why = (result.get("why") or "").strip()
    return ("Это инсайт. " if is_insight else "Это не инсайт. ") + why


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(START_TEXT)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_text = update.message.text.strip()
    if not user_text:
        await update.message.reply_text("Пришли одну мысль или один абзац.")
        return
    try:
        result = call_model(user_text)
        reply = humanize(result)
        await update.message.reply_text(reply)
    except Exception:
        logger.exception("Model or bot error")
        return


if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(drop_pending_updates=True)
