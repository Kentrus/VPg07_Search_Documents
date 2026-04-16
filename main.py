"""
Точка входа — запуск MAX messenger бота с поиском по документам.

Архитектура:
    components/  — инструменты (tools) и контекст (Pinecone)
    pipelines/   — ingestion (файл → база) и generation (запрос → ответ)
    bot/         — обработчики MAX messenger
"""

import asyncio
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
# Приглушаем шумные библиотеки
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("docling").setLevel(logging.INFO)

logger = logging.getLogger("main")

# Импортируем после настройки логирования
from bot.handlers import bot, dp


async def main():
    logger.info("=" * 60)
    logger.info("Запуск MAX бота: поиск по документам (Haystack + Docling)")
    logger.info("=" * 60)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
