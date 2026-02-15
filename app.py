import os
import time
import threading
from hashlib import sha256
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Загружаем .env локально
load_dotenv()

app = Flask(__name__)

# --- Настройки (можно менять без переписывания логики)
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Таймаут на запрос к модели (миллисекунды)
GENAI_TIMEOUT_MS = int(os.getenv("GENAI_TIMEOUT_MS", "25000"))

# Rate limit: N запросов за WINDOW секунд на один IP
RATE_LIMIT_N = int(os.getenv("RATE_LIMIT_N", "10"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

# Кэш: сколько хранить ответ (секунды)
CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "21600"))  # 6 часов

# max_output_tokens по уровням
MAX_TOKENS_BY_LEVEL = {
    "Кратко": 320,
    "Средне": 800,
    "Подробно": 1600,
}

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Не задан GEMINI_API_KEY. Локально: .env, на Render: Environment Variables.")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1", timeout=GENAI_TIMEOUT_MS)
)

# --- In-memory structures ---
_lock = threading.Lock()

# rate limiting: ip -> [timestamps]
_ip_hits: dict[str, list[float]] = {}

# cache: key -> (answer, expires_at)
_cache: dict[str, tuple[str, float]] = {}


def _get_ip() -> str:
    """
    На Render запрос может идти через прокси.
    X-Forwarded-For обычно содержит реальный IP клиента.
    """
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_limit_check(ip: str) -> bool:
    """
    True = можно, False = лимит превышен.
    """
    now = time.time()
    with _lock:
        hits = _ip_hits.get(ip, [])
        # оставляем только события в окне
        hits = [t for t in hits if now - t < RATE_LIMIT_WINDOW_SEC]
        if len(hits) >= RATE_LIMIT_N:
            _ip_hits[ip] = hits
            return False
        hits.append(now)
        _ip_hits[ip] = hits
        return True


def _make_cache_key(grade: str, subject: str, level: str, style: str, topic: str) -> str:
    raw = f"{grade}|{subject}|{level}|{style}|{topic.strip().lower()}"
    return sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> str | None:
    now = time.time()
    with _lock:
        item = _cache.get(key)
        if not item:
            return None
        answer, expires_at = item
        if now >= expires_at:
            _cache.pop(key, None)
            return None
        return answer


def _cache_set(key: str, answer: str) -> None:
    expires_at = time.time() + CACHE_TTL_SEC
    with _lock:
        _cache[key] = (answer, expires_at)


def _friendly_error_message(err: Exception) -> tuple[str, int]:
    s = str(err)

    # 429 / quota / resource exhausted
    if "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower():
        return ("Слишком много запросов или закончился лимит. Подожди минуту и попробуй снова.", 429)

    # timeout / disconnected
    if "timeout" in s.lower() or "Server disconnected" in s or "RemoteProtocolError" in s:
        return ("Сервис отвечал слишком долго. Попробуй ещё раз (или выбери «Кратко»).", 504)

    # прочие ошибки модели/сервера
    return ("Произошла ошибка на сервере. Попробуй повторить запрос чуть позже.", 500)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    ip = _get_ip()

    if not _rate_limit_check(ip):
        return jsonify(error="Слишком много запросов. Подожди немного и попробуй снова."), 429

    data = request.get_json(silent=True)
    if not data:
        return jsonify(error="Неверный запрос. Попробуй обновить страницу."), 400

    grade = (data.get("grade") or "").strip()
    subject = (data.get("subject") or "").strip()
    level = (data.get("level") or "").strip()
    style = (data.get("style") or "").strip()
    topic = (data.get("topic") or "").strip()

    if not topic:
        return jsonify(error="Введите тему."), 400

    # --- Кэш ---
    cache_key = _make_cache_key(grade, subject, level, style, topic)
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(answer=cached, cached=True)

    # --- Настройка длины ---
    max_out = MAX_TOKENS_BY_LEVEL.get(level, 800)

    # --- Промпт (пока базовый; позже усилим на этапе 1) ---
    prompt = f"""
Ты — помощник для школьника.
Класс: {grade}
Предмет: {subject}
Тема: {topic}
Стиль: {style}

Сделай ответ структурированным:
1) Определение
2) Пояснение
3) 1–2 примера
4) Краткий вывод (3 тезиса)
5) Проверь себя: 2 вопроса
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_out
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            return jsonify(error="Модель вернула пустой ответ. Попробуй ещё раз."), 500

        _cache_set(cache_key, answer)
        return jsonify(answer=answer, cached=False)

    except Exception as e:
        msg, code = _friendly_error_message(e)
        return jsonify(error=msg), code


if __name__ == "__main__":
    app.run(debug=True)
