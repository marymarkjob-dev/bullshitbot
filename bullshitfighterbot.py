import os
import json
import logging
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
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
- Сначала определяй только whether это insight или not insight.
- Другие категории не выводи и не объясняй, если о них не спросили.
- Если текст похож на общий факт, наблюдение, гипотезу, тренд, статистику или описание механизма без внутреннего человеческого конфликта, то это НЕ insight.
- Insight — это текст, где есть внутреннее напряжение, противоречие, скрытый мотив, неожиданное человеческое открытие или глубокая смысловая развилка.
- Если текст можно честно объяснить как факт, гипотезу, наблюдение, тренд или формулировку без конфликта, отвечай false.
- Если в gold set для этого типа текста уже есть правило, следуй gold set, даже если формулировка кажется похожей на insight.
- Не дублируй ответ.
- Не отправляй два сообщения.
- Не повторяй вывод в разных формулировках.
- Не добавляй лишний текст, списки, заголовки, пояснения вне полей ответа.

КАК ПИСАТЬ why:
- Почему это insight: укажи внутренний конфликт, напряжение, парадокс, скрытый мотив или смысловое открытие.
- Почему это не insight: укажи, что это факт, гипотеза, наблюдение, тренд или описание механизма без инсайта.
- why должно быть коротким: 1–2 предложения максимум.

ОСОБЫЕ ПРАВИЛА ДЛЯ СПОРНЫХ СЛУЧАЕВ:
- Фраза вроде "Ограниченность предложения усиливает желание купить" — это НЕ insight, а гипотеза или маркетинговое наблюдение.
- Фразы без внутреннего конфликта не считать инсайтом, даже если они звучат умно или психологично.
- Если текст — афоризм или краткая мысль, проверяй: есть ли там реальный человеческий конфликт? Если нет, false.
- Если сомневаешься между insight и не insight, выбирай false.

ФОРМАТ ОТВЕТА:
{
  "is_insight": true или false,
  "why": "короткое объяснение"
}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["инсайт", "наблюдение", "факт", "тренд", "гипотеза", "статистика", "мнение", "преимущество продукта", "слоган", "другое"],
        },
        "is_insight": {"type": "boolean"},
        "strength": {"type": "string", "enum": ["strong", "emerging", "none"]},
        "reason": {"type": "string"},
        "suggestion": {"type": ["string", "null"]},
    },
    "required": ["category", "is_insight", "strength", "reason", "suggestion"],
    "additionalProperties": False,
}


def call_model(user_text: str) -> dict:
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": PROMPT_V12},
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
    category = result.get("category")
    is_insight = result.get("is_insight")
    strength = result.get("strength")
    reason = (result.get("reason") or "").strip()
    suggestion: Optional[str] = result.get("suggestion")

    if is_insight and strength == "strong":
        return f"Это инсайт. {reason}"

    if (not is_insight) and strength == "emerging":
        if suggestion:
            return f"Это пока не инсайт, но мысль близкая. {reason}\n\nЧтобы усилить мысль: {suggestion}"
        return f"Это пока не инсайт, но мысль близкая. {reason}"

    return f"Это не инсайт, а {category}. {reason}"


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
        await update.message.reply_text("Не получилось оценить формулировку. Попробуй отправить одну мысль одним сообщением.")


if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.run_polling(drop_pending_updates=True)
