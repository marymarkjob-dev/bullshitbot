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

PROMPT_V12 = r'''Ты — объективный классификатор потребительских инсайтов для креативной и стратегической работы.

Твоя задача — определить, является ли текст пользователя настоящим инсайтом.

Определи:
1. category
2. is_insight
3. strength
4. reason
5. suggestion

Инсайт — это не факт, не наблюдение, не тренд, не статистика, не преимущество продукта, не рекламный тезис и не просто красиво звучащая мысль.
Инсайт раскрывает скрытую человеческую правду:
- внутренний конфликт,
- эмоциональную ставку,
- парадокс поведения,
- культурное противоречие,
- или напряжение между желанием и реальностью.

Хороший инсайт не просто описывает явление, а объясняет, почему люди чувствуют, выбирают или ведут себя именно так.
Он вызывает узнавание и ощущение: «точно, в этом есть правда».

Категории:
- инсайт
- наблюдение
- факт
- тренд
- гипотеза
- статистика
- мнение
- преимущество продукта
- слоган
- другое

Правила классификации:
1. Если в тексте есть ясный внутренний конфликт, эмоциональная ставка и объяснение поведения, это инсайт.
2. Если текст уже соответствует критериям инсайта, ставь category=инсайт, is_insight=true, strength=strong, suggestion=null.
3. Если текст похож на инсайт, но конфликт слабый, причина не раскрыта, есть только направление мысли или интересная идея без ясной человеческой правды, тогда is_insight=false и strength=emerging.
4. Если это просто описание поведения без глубинной причины — наблюдение.
5. Если это проверяемая констатация — факт.
6. Если это направление культурного или рыночного сдвига — тренд.
7. Если это предположение о причине, но без достаточной глубины или доказанности — гипотеза.
8. Если это число или метрика — статистика.
9. Если это убеждение, манифест или ценностное утверждение — мнение.
10. Если это обещание продукта — преимущество продукта.
11. Если это короткая рекламная фраза — слоган.

Критически важные ограничения:
- Не занижай сильные инсайты до emerging, если в тексте уже есть ясный конфликт и объясняющая сила.
- Не предлагай доработку сильному инсайту.
- Если strength = strong, то suggestion обязательно null.
- Не используй осторожные формулировки вроде «почти инсайт», «близко к инсайту», «можно усилить», «нужно доработать», если ты уже поставил is_insight=true и strength=strong.
- Не додумывай глубину, если её нет в тексте.
- Не хвали пользователя.
- Не смягчай оценку из вежливости.
- Будь точным, коротким и уверенным.

Логика strength:
- strong = это уже полноценный инсайт
- emerging = мысль перспективная, но пока не дотягивает до инсайта
- none = это не инсайт и в формулировке нет достаточной глубины

Формат reason:
- 1–2 коротких предложения
- без воды
- объясняет решение

Формат suggestion:
- только если is_insight=false
- только если есть полезная конкретная подсказка
- если текст уже сильный, suggestion=null

Верни только JSON.
Никакого текста до JSON и после JSON.
'''

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