"""
PineconeManager — модуль для работы с векторной памятью Pinecone.

Возможности:
  - создание эмбеддингов через OpenAI (в т.ч. через прокси по OPENAI_BASE_URL)
  - запись/чтение векторов и документов
  - автоматическая фильтрация дубликатов по косинусному сходству
"""

import os
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
logger = logging.getLogger(__name__)

# Глобальный порог косинусного сходства.
# score >= threshold → дубликат (обновляем существующий слот)
# score <  threshold → новая информация (создаём новую запись)
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))


class PineconeManager:
    """Класс для управления операциями с векторной базой данных Pinecone."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        index_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        openai_model: str = "text-embedding-3-small",
        embedding_dim: int = 1536,
        similarity_threshold: Optional[float] = None,
    ):
        load_dotenv()

        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.region = region or os.getenv("PINECONE_REGION", "us-east-1")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.embedding_dim = embedding_dim
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else SIMILARITY_THRESHOLD
        )

        if not self.api_key:
            raise ValueError("PINECONE_API_KEY не найден. Укажите в параметрах или в .env файле.")
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME не найден. Укажите в параметрах или в .env файле.")

        self.pc = Pinecone(api_key=self.api_key)
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
        self.openai_model = openai_model
        self.openai_client: Optional[OpenAI] = None
        if self.openai_api_key:
            client_kwargs: Dict[str, Any] = {"api_key": self.openai_api_key}
            if self.openai_base_url:
                client_kwargs["base_url"] = self.openai_base_url
            self.openai_client = OpenAI(**client_kwargs)

    def _ensure_index_exists(self) -> None:
        existing = [idx["name"] for idx in self.pc.list_indexes()]
        if self.index_name not in existing:
            logger.info("Индекс %s не найден — создаём новый.", self.index_name)
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.region),
            )

    # ------------------------------------------------------------------
    # Эмбеддинги
    # ------------------------------------------------------------------

    def create_embedding(self, text: str) -> List[float]:
        if self.openai_client is None:
            raise RuntimeError("OpenAI клиент не инициализирован — нужен OPENAI_API_KEY.")
        response = self.openai_client.embeddings.create(
            model=self.openai_model,
            input=text,
        )
        return response.data[0].embedding

    # ------------------------------------------------------------------
    # Проверка сходства
    # ------------------------------------------------------------------

    def _check_similarity(
        self,
        vector: List[float],
        filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Возвращает лучший найденный вектор, если его score >= threshold, иначе None."""
        try:
            result = self.index.query(
                vector=vector,
                top_k=1,
                include_metadata=True,
                filter=filter,
            )
        except Exception as e:
            logger.warning("Ошибка при поиске сходства: %s", e)
            return None

        matches = result.get("matches", []) if isinstance(result, dict) else result.matches
        if not matches:
            return None

        best = matches[0]
        score = best["score"] if isinstance(best, dict) else best.score
        best_id = best["id"] if isinstance(best, dict) else best.id
        best_meta = (best.get("metadata") if isinstance(best, dict) else best.metadata) or {}

        if score >= self.similarity_threshold:
            return {"id": best_id, "score": score, "metadata": best_meta}
        return None

    # ------------------------------------------------------------------
    # Запись векторов
    # ------------------------------------------------------------------

    def upsert_vector(
        self,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        check_similarity: bool = True,
    ) -> Dict[str, Any]:
        """
        Записывает вектор с предварительной проверкой косинусного сходства.

        Returns:
            {
              "action": "inserted" | "updated",
              "similarity_score": float | None,
              "existing_id": str | None,
            }
        """
        result: Dict[str, Any] = {
            "action": "inserted",
            "similarity_score": None,
            "existing_id": None,
        }

        if check_similarity:
            filter_ = None
            if metadata and "user_id" in metadata:
                filter_ = {"user_id": metadata["user_id"]}
            similar = self._check_similarity(vector, filter=filter_)
            if similar:
                existing_id = similar["id"]
                result.update(
                    action="updated",
                    similarity_score=similar["score"],
                    existing_id=existing_id,
                )
                self.index.upsert(
                    vectors=[
                        {
                            "id": existing_id,
                            "values": vector,
                            "metadata": metadata or {},
                        }
                    ]
                )
                return result

        self.index.upsert(
            vectors=[
                {
                    "id": vector_id,
                    "values": vector,
                    "metadata": metadata or {},
                }
            ]
        )
        return result

    def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]],
        check_similarity: bool = True,
    ) -> List[Dict[str, Any]]:
        return [
            self.upsert_vector(
                vector_id=v["id"],
                vector=v["values"],
                metadata=v.get("metadata"),
                check_similarity=check_similarity,
            )
            for v in vectors
        ]

    # ------------------------------------------------------------------
    # Запись документов (текст → эмбеддинг → вектор)
    # ------------------------------------------------------------------

    def upsert_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        check_similarity: bool = True,
    ) -> Dict[str, Any]:
        embedding = self.create_embedding(text)
        full_metadata = {**(metadata or {}), "text": text}
        return self.upsert_vector(
            vector_id=doc_id,
            vector=embedding,
            metadata=full_metadata,
            check_similarity=check_similarity,
        )

    def upsert_documents(
        self,
        documents: List[Dict[str, Any]],
        check_similarity: bool = True,
    ) -> List[Dict[str, Any]]:
        return [
            self.upsert_document(
                doc_id=doc["id"],
                text=doc["text"],
                metadata=doc.get("metadata"),
                check_similarity=check_similarity,
            )
            for doc in documents
        ]

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------

    def query_by_vector(
        self,
        vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        return self.index.query(
            vector=vector,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata,
        )

    def query_by_text(
        self,
        text: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        vector = self.create_embedding(text)
        return self.query_by_vector(
            vector,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata,
        )

    def fetch_vectors(self, ids: List[str]) -> Dict[str, Any]:
        return self.index.fetch(ids=ids)

    # ------------------------------------------------------------------
    # Удаление
    # ------------------------------------------------------------------

    def delete(self, ids: List[str]):
        return self.index.delete(ids=ids)

    def delete_by_filter(self, filter: Dict[str, Any]):
        return self.index.delete(filter=filter)

    def delete_all(self):
        return self.index.delete(delete_all=True)

    # ------------------------------------------------------------------
    # Служебное
    # ------------------------------------------------------------------

    def describe_index_stats(self) -> Dict[str, Any]:
        return self.index.describe_index_stats()

    def update_metadata(self, vector_id: str, metadata: Dict[str, Any]):
        return self.index.update(id=vector_id, set_metadata=metadata)


# ----------------------------------------------------------------------
# Ручной тест — запускается командой: python pinecone_manager.py
#
# Важно про serverless Pinecone:
#   - describe_index_stats() имеет кэш и может показывать устаревшие счётчики.
#     Для точного подсчёта используем index.list().
#   - После upsert новая запись становится searchable не сразу — нужна пауза,
#     иначе similarity-check для следующей фразы не увидит только что вставленное.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import time

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8")
        except OSError:
            pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=== Ручной тест PineconeManager ===")
    manager = PineconeManager()
    print(f"Индекс: {manager.index_name}")
    print(f"Порог сходства: {manager.similarity_threshold}")

    # Чистый старт
    print("\nОчищаем индекс перед тестом...")
    manager.delete_all()
    time.sleep(3)

    test_user = "test-user-001"
    phrases = [
        ("1", "Я люблю пиццу с пепперони"),
        ("2", "Моё любимое блюдо — пицца пепперони"),   # смысловой дубликат → updated
        ("3", "Я программирую на Python"),               # новое → inserted
        ("4", "В свободное время пишу код на Python"),   # похоже → updated
        ("5", "Я родился в Москве"),                     # новое → inserted
    ]

    print("\n--- Запись фраз (с паузой 2с на индексацию) ---")
    for suffix, text in phrases:
        doc_id = f"{test_user}-{suffix}"
        result = manager.upsert_document(
            doc_id=doc_id,
            text=text,
            metadata={"user_id": test_user},
        )
        score = result["similarity_score"]
        score_str = f"{score:.3f}" if score is not None else "—"
        print(f"  [{suffix}] '{text}' → action={result['action']}, score={score_str}")
        time.sleep(2)

    print("\n--- Реальное состояние индекса (через list) ---")
    all_ids = []
    for page in manager.index.list():
        all_ids.extend(page)
    print(f"  IDs: {all_ids}")
    print(f"  Реальное количество записей: {len(all_ids)}")
    print(f"  Ожидаем: 3 (дубликаты схлопнулись в слоты оригиналов)")

    print("\n--- Содержимое записей ---")
    if all_ids:
        fetched = manager.fetch_vectors(all_ids)
        vectors = fetched.vectors if hasattr(fetched, "vectors") else fetched.get("vectors", {})
        for vid, v in vectors.items():
            meta = v.metadata if hasattr(v, "metadata") else v.get("metadata", {})
            print(f"  {vid} → {meta.get('text')}")

    print("\n--- Поиск: 'что я люблю есть?' ---")
    results = manager.query_by_text(
        text="что я люблю есть?",
        top_k=3,
        filter={"user_id": test_user},
    )
    matches = results.get("matches", []) if isinstance(results, dict) else results.matches
    for m in matches:
        score = m["score"] if isinstance(m, dict) else m.score
        meta = (m.get("metadata") if isinstance(m, dict) else m.metadata) or {}
        print(f"  score={score:.3f} | text={meta.get('text')}")

    print("\n=== Тест завершён ===")
