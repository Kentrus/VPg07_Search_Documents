"""
Инструменты (Tools) для Haystack-агента.

Каждый инструмент — обычная Python-функция с декоратором @tool.
Haystack автоматически генерирует JSON-схему из типов и Annotated-описаний,
а Agent (через OpenAI function calling) решает, когда какой вызвать.
"""

import os
import logging
from typing import Annotated

import requests
from openai import OpenAI
from dotenv import load_dotenv
from haystack.tools import tool

load_dotenv()
logger = logging.getLogger(__name__)


# ── Факт о кошках ─────────────────────────────────────────────────────

@tool
def get_cat_fact() -> str:
    """Получает случайный интересный факт о кошках из API catfact.ninja.
    Используй этот инструмент, когда пользователь просит факт о кошках."""
    try:
        resp = requests.get("https://catfact.ninja/fact", timeout=10)
        resp.raise_for_status()
        fact = resp.json().get("fact", "Факт не найден.")
        logger.info("CatFact получен: %s", fact[:80])
        return fact
    except Exception as e:
        logger.error("Ошибка CatFact API: %s", e)
        return f"Не удалось получить факт о кошках: {e}"


# ── Картинка собаки + анализ породы ───────────────────────────────────

@tool
def get_dog_image_with_analysis() -> str:
    """Получает случайную картинку собаки и анализирует породу через OpenAI Vision.
    Возвращает URL картинки и описание породы.
    Используй этот инструмент, когда пользователь просит картинку собаки
    или хочет узнать о породе собаки."""
    try:
        resp = requests.get("https://dog.ceo/api/breeds/image/random", timeout=10)
        resp.raise_for_status()
        image_url = resp.json().get("message", "")
        if not image_url:
            return "Не удалось получить картинку собаки."
        logger.info("DogImage URL: %s", image_url)

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        vision_resp = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Определи породу собаки на фото. "
                                "Дай краткое описание породы: характер, "
                                "история происхождения, интересные факты. "
                                "Отвечай на русском языке."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        analysis = vision_resp.choices[0].message.content.strip()
        logger.info("DogImage анализ получен, длина: %d", len(analysis))
        return f"Картинка: {image_url}\n\nАнализ породы:\n{analysis}"

    except Exception as e:
        logger.error("Ошибка DogImage: %s", e)
        return f"Не удалось получить/проанализировать картинку собаки: {e}"


# ── Погода ────────────────────────────────────────────────────────────

@tool
def get_weather(
    city: Annotated[str, "Название города, например Moscow или Париж"] = "Moscow",
) -> str:
    """Получает текущую погоду в указанном городе через бесплатный API wttr.in.
    Используй этот инструмент, когда пользователь спрашивает о погоде."""
    try:
        resp = requests.get(
            f"https://wttr.in/{city}?format=j1",
            timeout=10,
            headers={"Accept-Language": "ru"},
        )
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "?")
        feels_like = current.get("FeelsLikeC", "?")
        humidity = current.get("humidity", "?")
        desc_list = current.get("weatherDesc", [{}])
        desc = desc_list[0].get("value", "нет данных") if desc_list else "нет данных"
        wind_kmph = current.get("windspeedKmph", "?")

        result = (
            f"Погода в {city}:\n"
            f"- Температура: {temp_c}°C (ощущается как {feels_like}°C)\n"
            f"- Состояние: {desc}\n"
            f"- Влажность: {humidity}%\n"
            f"- Ветер: {wind_kmph} км/ч"
        )
        logger.info("Weather для %s: %s°C, %s", city, temp_c, desc)
        return result

    except Exception as e:
        logger.error("Ошибка Weather API: %s", e)
        return f"Не удалось получить погоду для {city}: {e}"
