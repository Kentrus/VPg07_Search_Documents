"""
Обработчики MAX messenger — текстовые сообщения и файлы.

Логика:
  - Текст: контекст из Pinecone → Haystack Agent → ответ
  - Файл: скачивание → Docling → чанки → Pinecone → резюме пользователю
  - Команды: /start, /help, /clear
"""

import asyncio
import logging
import os
import tempfile

from maxapi import Bot, Dispatcher, F
from maxapi.types import BotStarted, Command, MessageCreated

from components.context import ContextManager
from pipelines.generation import GenerationPipeline
from pipelines.ingestion import IngestionPipeline
from pinecone_manager import PineconeManager

logger = logging.getLogger(__name__)

# ── Инициализация ────────────────────────────────────────────────────

bot = Bot(token=os.getenv("MAX_BOT_TOKEN"))
dp = Dispatcher()

memory = PineconeManager()
ctx = ContextManager(memory)
generation = GenerationPipeline()
ingestion = IngestionPipeline(memory)

# Поддерживаемые расширения документов
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".pptx", ".html", ".md", ".txt"}


# ── Команды ──────────────────────────────────────────────────────────

@dp.bot_started()
async def on_bot_started(event: BotStarted):
    await event.bot.send_message(
        chat_id=event.chat_id,
        text=(
            "Привет! Я персональный ассистент с поиском по документам.\n\n"
            "Что я умею:\n"
            "- Запоминаю контекст наших разговоров\n"
            "- Анализирую PDF, DOCX и другие документы\n"
            "- Отвечаю на вопросы по загруженным файлам\n"
            "- Факты о кошках, картинки собак, погода\n\n"
            "Отправь мне файл или просто напиши!"
        ),
    )


@dp.message_created(Command("start"))
async def on_start(event: MessageCreated):
    await event.message.answer(
        "Привет! Я ассистент с памятью и поиском по документам.\n"
        "Отправь PDF/DOCX — я изучу и смогу отвечать на вопросы по содержимому.\n"
        "Также спроси о погоде, кошках или собаках!"
    )


@dp.message_created(Command("help"))
async def on_help(event: MessageCreated):
    await event.message.answer(
        "Возможности:\n"
        "- Отправь документ (PDF, DOCX) — я изучу его содержимое\n"
        "- Задай вопрос по документу — я найду ответ\n"
        "- Факт о кошках: «расскажи факт о кошках»\n"
        "- Картинка собаки: «покажи собаку»\n"
        "- Погода: «какая погода в Москве?»\n"
        "- Память: я запоминаю наши разговоры\n\n"
        "/clear — очистить историю и документы"
    )


@dp.message_created(Command("clear"))
async def on_clear(event: MessageCreated):
    sender = event.message.sender
    user_id = str(getattr(sender, "user_id", "unknown")) if sender else "unknown"
    try:
        memory.delete_by_filter({"user_id": user_id})
        await event.message.answer("История и документы очищены! Начинаем с чистого листа.")
    except Exception as e:
        logger.error("Ошибка очистки: %s", e)
        await event.message.answer(f"Не удалось очистить: {e}")


# ── Обработка файлов ─────────────────────────────────────────────────

@dp.message_created(F.message.body.attachments)
async def on_file(event: MessageCreated):
    """Обработка загруженных файлов через Docling → Pinecone."""
    sender = event.message.sender
    user_id = str(getattr(sender, "user_id", "unknown")) if sender else "unknown"

    attachments = event.message.body.attachments or []
    if not attachments:
        return

    for attachment in attachments:
        # Проверяем, что это файл с поддерживаемым расширением
        filename = getattr(attachment, "filename", None)
        if not filename:
            # Пробуем получить из payload
            payload = getattr(attachment, "payload", None)
            if payload:
                filename = getattr(payload, "filename", None)

        if not filename:
            logger.debug("[ФАЙЛ] Вложение без имени файла, пропускаем")
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            await event.message.answer(
                f"Формат {ext} не поддерживается. Поддерживаются: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
            continue

        # Получаем URL для скачивания
        file_url = None
        payload = getattr(attachment, "payload", None)
        if payload:
            file_url = getattr(payload, "url", None)

        if not file_url:
            logger.warning("[ФАЙЛ] Не удалось получить URL файла %s", filename)
            await event.message.answer(f"Не удалось получить ссылку на файл {filename}.")
            continue

        logger.info("[ФАЙЛ] Получен файл: %s (ext=%s)", filename, ext)
        await event.message.answer(
            f"Файл «{filename}» получен. Запускаю анализ и сохранение. "
            "Это может занять немного времени..."
        )

        # Скачиваем файл во временную директорию
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                local_path = await bot.download_file(file_url, tmp_dir)
                logger.info("[ФАЙЛ] Скачан в: %s", local_path)

                # Запускаем Ingestion Pipeline в отдельном потоке
                result = await asyncio.to_thread(
                    ingestion.process_file,
                    str(local_path),
                    user_id,
                    filename,
                )

            chunks_count = result["chunks_count"]
            summary = result["summary"]

            await event.message.answer(
                f"Готово! Файл «{filename}» изучен ({chunks_count} фрагментов сохранено).\n\n"
                f"Резюме: {summary}\n\n"
                "Теперь можешь задавать вопросы по содержимому!"
            )
            logger.info("[ФАЙЛ] Обработка завершена: %s, %d чанков", filename, chunks_count)

        except Exception as e:
            logger.exception("[ФАЙЛ] Ошибка обработки %s", filename)
            await event.message.answer(f"Ошибка при обработке файла «{filename}»: {e}")


# ── Обработка текстовых сообщений ────────────────────────────────────

@dp.message_created(F.message.body.text)
async def on_message(event: MessageCreated):
    """Текстовое сообщение → контекст из Pinecone → Agent → ответ."""
    sender = event.message.sender
    user_text = (event.message.body.text or "").strip() if event.message.body else ""
    if not user_text:
        return

    user_id = str(getattr(sender, "user_id", "unknown")) if sender else "unknown"
    logger.info("=" * 60)
    logger.info("[СООБЩЕНИЕ] От user=%s: '%s'", user_id, user_text)

    # 1. Контекст из Pinecone (сообщения + документы)
    memory_context = ctx.fetch_context(user_id, user_text)

    # 2. Generation Pipeline (Agent)
    reply = await asyncio.to_thread(generation.run, user_text, memory_context)

    # 3. Сохраняем ТОЛЬКО сообщение пользователя
    await asyncio.to_thread(ctx.save_user_message, user_id, user_text, sender)

    # 4. Ответ
    logger.info("[ОТВЕТ] Отправляем (%d символов)", len(reply))
    await event.message.answer(reply)
