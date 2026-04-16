"""
Модуль работы с контекстом — чтение и запись в Pinecone.

Отвечает за:
  - Поиск релевантного контекста из памяти (сообщения пользователя + документы)
  - Сохранение сообщений пользователя в Pinecone
"""

import logging
from datetime import datetime, timezone

from pinecone_manager import PineconeManager

logger = logging.getLogger(__name__)


class ContextManager:
    """Управляет контекстом пользователя в Pinecone."""

    def __init__(self, memory: PineconeManager, score_threshold: float = 0.2):
        self.memory = memory
        self.score_threshold = score_threshold

    def fetch_context(self, user_id: str, query: str, top_k: int = 5) -> str:
        """Ищем релевантный контекст: сообщения пользователя + чанки документов."""
        logger.info("[КОНТЕКСТ] Ищем для user=%s, запрос='%s'", user_id, query[:80])
        try:
            filter_ = {"user_id": user_id} if user_id != "unknown" else None
            results = self.memory.query_by_text(text=query, top_k=top_k, filter=filter_)
        except Exception as e:
            logger.warning("[КОНТЕКСТ] Не удалось прочитать: %s", e)
            return ""

        matches = (
            results.get("matches", []) if isinstance(results, dict) else results.matches
        )
        chunks = []
        for m in matches:
            meta = (m.get("metadata") if isinstance(m, dict) else m.metadata) or {}
            text = meta.get("text")
            score = m.get("score") if isinstance(m, dict) else m.score
            source = meta.get("source", "memory")

            if text and score is not None and score >= self.score_threshold:
                prefix = f"[doc:{meta.get('filename', '?')}]" if source == "document" else "[msg]"
                logger.info("[КОНТЕКСТ] Найдено: score=%.3f %s text='%s'", score, prefix, text[:60])
                chunks.append(f"- {prefix} {text}")
            elif text and score is not None:
                logger.debug("[КОНТЕКСТ] Пропущено (score=%.3f): '%s'", score, text[:60])

        if not chunks:
            logger.info("[КОНТЕКСТ] Релевантный контекст не найден")
            return ""

        logger.info("[КОНТЕКСТ] Найдено %d фрагментов", len(chunks))
        return "Контекст из памяти и документов:\n" + "\n".join(chunks)

    def save_user_message(self, user_id: str, text: str, sender=None) -> None:
        """Сохраняем ТОЛЬКО текст пользователя (не ответы бота, не чанки документов)."""
        metadata = self._build_metadata(user_id, sender)
        metadata["source"] = "user_message"
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
        vector_id = f"{user_id}-msg-{int(datetime.now(timezone.utc).timestamp() * 1000)}"

        logger.info("[КОНТЕКСТ] Сохраняем сообщение: '%s'", text[:80])
        try:
            result = self.memory.upsert_document(
                doc_id=vector_id,
                text=text,
                metadata=metadata,
            )
            score = result.get("similarity_score")
            score_str = f"{score:.3f}" if score is not None else "—"
            logger.info(
                "[КОНТЕКСТ] Память: action=%s score=%s",
                result.get("action"), score_str,
            )
        except Exception as e:
            logger.warning("[КОНТЕКСТ] Не удалось записать: %s", e)

    @staticmethod
    def _build_metadata(user_id: str, sender=None) -> dict:
        metadata = {"user_id": user_id}
        if sender is None:
            return metadata
        for field in ("first_name", "last_name", "username"):
            value = getattr(sender, field, None)
            if value:
                metadata[field] = value
        return metadata
