"""
Ingestion Pipeline — загрузка документов В базу Pinecone.

Поток данных:
    Файл (PDF/DOCX) → Docling (парсинг → Markdown) → Разбивка на чанки
    → OpenAI Embeddings → Pinecone (сохранение с метаданными)
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from docling.document_converter import DocumentConverter
from openai import OpenAI
from dotenv import load_dotenv

from pinecone_manager import PineconeManager

load_dotenv()
logger = logging.getLogger(__name__)

# Размер чанка (символов). ~500 слов ≈ ~2500 символов
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


class IngestionPipeline:
    """Конвейер загрузки документов: файл → чанки → эмбеддинги → Pinecone."""

    def __init__(self, memory: PineconeManager):
        self.memory = memory
        self.converter = DocumentConverter()
        self.llm_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )

    def process_file(self, file_path: str, user_id: str, filename: str) -> dict:
        """
        Полный цикл обработки файла.

        Returns:
            {
                "chunks_count": int,
                "summary": str,
                "filename": str,
            }
        """
        logger.info("[INGESTION] Начинаем обработку файла: %s", filename)

        # 1. Docling: файл → Markdown текст
        logger.info("[INGESTION] Шаг 1: Docling парсит файл...")
        text = self._convert_to_text(file_path)
        logger.info("[INGESTION] Docling готов, текст: %d символов", len(text))

        if not text.strip():
            logger.warning("[INGESTION] Docling вернул пустой текст")
            return {"chunks_count": 0, "summary": "Не удалось извлечь текст из файла.", "filename": filename}

        # 2. Разбиваем на чанки
        logger.info("[INGESTION] Шаг 2: Разбиваем на чанки...")
        chunks = self._split_into_chunks(text)
        logger.info("[INGESTION] Получено %d чанков", len(chunks))

        # 3. Сохраняем каждый чанк в Pinecone
        logger.info("[INGESTION] Шаг 3: Сохраняем в Pinecone...")
        self._save_chunks_to_pinecone(chunks, user_id, filename)
        logger.info("[INGESTION] Все чанки сохранены в Pinecone")

        # 4. Генерируем резюме
        logger.info("[INGESTION] Шаг 4: Генерируем резюме...")
        summary = self._generate_summary(text, filename)
        logger.info("[INGESTION] Резюме: %s", summary[:100])

        return {
            "chunks_count": len(chunks),
            "summary": summary,
            "filename": filename,
        }

    def _convert_to_text(self, file_path: str) -> str:
        """Docling конвертирует файл в Markdown-текст."""
        try:
            result = self.converter.convert(file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            logger.error("[INGESTION] Ошибка Docling: %s", e)
            return ""

    def _split_into_chunks(self, text: str) -> list[str]:
        """Разбиваем текст на чанки с перекрытием."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - CHUNK_OVERLAP
        return chunks

    def _save_chunks_to_pinecone(self, chunks: list[str], user_id: str, filename: str) -> None:
        """Сохраняем каждый чанк как отдельный вектор в Pinecone."""
        timestamp = datetime.now(timezone.utc).isoformat()
        for i, chunk in enumerate(chunks):
            vector_id = f"{user_id}-doc-{filename}-chunk-{i}-{int(datetime.now(timezone.utc).timestamp() * 1000)}"
            metadata = {
                "user_id": user_id,
                "source": "document",
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": timestamp,
            }
            try:
                result = self.memory.upsert_document(
                    doc_id=vector_id,
                    text=chunk,
                    metadata=metadata,
                    check_similarity=False,
                )
                logger.info(
                    "[INGESTION] Чанк %d/%d сохранён (id=%s)",
                    i + 1, len(chunks), vector_id[:50],
                )
            except Exception as e:
                logger.error("[INGESTION] Ошибка сохранения чанка %d: %s", i, e)

    def _generate_summary(self, text: str, filename: str) -> str:
        """LLM генерирует краткое резюме документа."""
        # Берём первые ~3000 символов для резюме (экономим токены)
        preview = text[:3000]
        try:
            resp = self.llm_client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
                messages=[
                    {
                        "role": "system",
                        "content": "Ты анализируешь документы. Дай краткое резюме в 2-3 предложения на русском языке.",
                    },
                    {
                        "role": "user",
                        "content": f"Файл: {filename}\n\nСодержимое:\n{preview}",
                    },
                ],
                max_tokens=200,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error("[INGESTION] Ошибка генерации резюме: %s", e)
            return f"Файл {filename} обработан ({len(text)} символов), но резюме создать не удалось."
