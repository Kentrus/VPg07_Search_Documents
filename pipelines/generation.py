"""
Generation Pipeline — генерация ответов ИЗ базы Pinecone.

Поток данных:
    Запрос пользователя → Поиск контекста в Pinecone
    → Haystack Agent (LLM + инструменты) → Ответ
"""

import logging
import os

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
from dotenv import load_dotenv

from components.tools import get_cat_fact, get_dog_image_with_analysis, get_weather

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты — умный персональный помощник в мессенджере MAX с долговременной памятью. "
    "Ты запоминаешь контекст разговора и можешь отвечать на вопросы по загруженным документам.\n\n"
    "У тебя есть три инструмента:\n"
    "1. get_cat_fact — получить случайный факт о кошках.\n"
    "2. get_dog_image_with_analysis — получить картинку собаки и анализ породы.\n"
    "3. get_weather — получить текущую погоду в городе.\n\n"
    "Если в контексте есть фрагменты из документов (помечены [doc:...]) — "
    "используй их для ответа. Если есть сообщения пользователя (помечены [msg]) — "
    "учитывай их для персонализации.\n"
    "Отвечай на русском языке, дружелюбно и по делу."
)


class GenerationPipeline:
    """Конвейер генерации ответов: запрос → контекст → Agent → ответ."""

    def __init__(self):
        chat_generator = OpenAIChatGenerator(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=Secret.from_token(os.getenv("OPENAI_API_KEY")),
            api_base_url=os.getenv("OPENAI_BASE_URL") or None,
        )

        self.agent = Agent(
            chat_generator=chat_generator,
            system_prompt=SYSTEM_PROMPT,
            tools=[get_cat_fact, get_dog_image_with_analysis, get_weather],
            exit_conditions=["text"],
            max_agent_steps=5,
        )

    def run(self, user_text: str, memory_context: str) -> str:
        """Запускаем Haystack Agent с контекстом из Pinecone."""
        messages = []
        if memory_context:
            logger.info("[GENERATION] Передаём контекст (%d символов)", len(memory_context))
            messages.append(ChatMessage.from_system(memory_context))
        else:
            logger.info("[GENERATION] Контекст отсутствует")

        messages.append(ChatMessage.from_user(user_text))
        logger.info("[GENERATION] Запускаем Agent с %d сообщениями...", len(messages))

        try:
            result = self.agent.run(messages=messages)
            agent_messages = result.get("messages", [])

            # Логируем шаги агента
            for i, msg in enumerate(agent_messages):
                role = msg.role if hasattr(msg, "role") else "?"
                text_preview = (msg.text or "")[:100] if hasattr(msg, "text") else str(msg)[:100]
                tool_calls = getattr(msg, "tool_calls", None)
                tool_call_result = getattr(msg, "tool_call_result", None)

                if tool_calls:
                    for tc in tool_calls:
                        logger.info("[GENERATION] Шаг %d: ВЫЗОВ ИНСТРУМЕНТА: %s(%s)",
                                    i, tc.tool_name, str(tc.arguments)[:80])
                elif tool_call_result:
                    logger.info("[GENERATION] Шаг %d: РЕЗУЛЬТАТ ИНСТРУМЕНТА: %s",
                                i, str(tool_call_result.result)[:120])
                else:
                    logger.info("[GENERATION] Шаг %d: role=%s text='%s'", i, role, text_preview)

            if agent_messages:
                final_text = agent_messages[-1].text or "Не удалось сформировать ответ."
                logger.info("[GENERATION] Ответ (%d символов): '%s'",
                            len(final_text), final_text[:120])
                return final_text
            return "Агент не вернул ответ."

        except Exception as e:
            logger.exception("[GENERATION] Ошибка Agent")
            return f"Ошибка агента: {e}"
