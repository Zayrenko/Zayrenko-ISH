import os
import time
import threading
from hashlib import sha256
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

app = Flask(__name__)

# --- Настройки ---
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GENAI_TIMEOUT_MS = int(os.getenv("GENAI_TIMEOUT_MS", "25000"))

RATE_LIMIT_N = int(os.getenv("RATE_LIMIT_N", "10"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))

CACHE_TTL_SEC = int(os.getenv("CACHE_TTL_SEC", "21600"))  # 6 часов

MAX_TOKENS_BY_LEVEL = {
    "Кратко": 800,
    "Средне": 2000,
    "Подробно": 4000,
}

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Не задан GEMINI_API_KEY.")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1", timeout=GENAI_TIMEOUT_MS)
)

_lock = threading.Lock()
_ip_hits = {}
_cache = {}


def _get_ip():
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _rate_limit_check(ip):
    now = time.time()
    with _lock:
        hits = _ip_hits.get(ip, [])
        hits = [t for t in hits if now - t < RATE_LIMIT_WINDOW_SEC]
        if len(hits) >= RATE_LIMIT_N:
            _ip_hits[ip] = hits
            return False
        hits.append(now)
        _ip_hits[ip] = hits
        return True


def _make_cache_key(grade, subject, level, style, topic):
    raw = f"{grade}|{subject}|{level}|{style}|{topic.strip().lower()}"
    return sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key):
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


def _cache_set(key, answer):
    expires_at = time.time() + CACHE_TTL_SEC
    with _lock:
        _cache[key] = (answer, expires_at)


def _friendly_error_message(err):
    s = str(err)

    if "429" in s or "RESOURCE_EXHAUSTED" in s or "quota" in s.lower():
        return ("Слишком много запросов или закончился лимит. Подожди минуту и попробуй снова.", 429)

    if "timeout" in s.lower():
        return ("Сервис отвечал слишком долго. Попробуй ещё раз.", 504)

    return ("Произошла ошибка на сервере. Попробуй позже.", 500)


# --- Правила по классам ---
GRADE_RULES = {
    "1-4": """
Уровень: младшая школа.

Объясняй максимально просто.
Короткие предложения.
Минимум терминов (0–1). Если используешь термин — объясни его сразу.
Не используй сложные формулы.
Пример должен быть бытовым и понятным.
Без сложных научных формулировок.
""",

    "5-6": """
Уровень: средние классы (базовый).

Можно использовать 1–2 термина с обязательным пояснением.
Объяснение должно быть логичным и последовательным.
Допустимы простые формулы или правила.
Без избыточной теории.
""",

    "7-9": """
Уровень: основная школа.

Допустимо 1–3 термина (каждый кратко пояснить).
Разрешены формулы и учебные определения.
Объяснение должно быть структурированным.
Можно указать 1 типичную ошибку (одним предложением).
""",

    "10-11": """
Уровень: старшая школа.

Можно использовать точные формулировки.
Допустимы формулы и краткие обоснования.
Без вузовской глубины.
Структура должна быть строгой и логичной.
"""
}

LEVEL_RULES = {
    "Кратко": """
Формат: сжато и чётко.
Каждый пункт структуры обязателен.
Минимум 2 предложения в каждом разделе.
Пример должен быть коротким, но завершённым.
Если нужно сократить — сокращай формулировки, но не убирай пункты.
""",

    "Средне": """
Формат: развернутое школьное объяснение.
Каждый пункт структуры обязателен.
Пункт 2 (Пошаговое объяснение) — минимум 3 шага.
Пример — с полноценным разбором.
Нельзя завершать ответ до заполнения всех пунктов.
""",

    "Подробно": """
Формат: максимально полное школьное объяснение.
Каждый пункт структуры обязателен.
Пункт 2 — 4–6 логичных шагов.
Пункт 3 — 1–2 примера с разбором.
Если не хватает объёма — сокращай детали, но не убирай структуру.
Ответ считается завершённым только после пункта 5.
"""
}

STYLE_RULES = {
    "Простым языком": """
Используй простые слова.
Если встречается сложный термин — объясни его простыми словами.
Без сложных конструкций.
""",

    "Как учитель": """
Объясняй спокойно и структурировано.
Можно использовать фразы типа "Важно:" или "Запомни:".
Без приветствий и разговорных вступлений.
""",

    "Научно": """
Используй точные формулировки.
Допустимы термины.
Без разговорного стиля.
Без лишней воды.
"""
}


def build_prompt(grade, subject, level, style, topic):
    grade_rule = GRADE_RULES.get(grade, "")
    level_rule = LEVEL_RULES.get(level, "")
    style_rule = STYLE_RULES.get(style, "")

    return f"""
Ты — образовательная система AI School Helper.

СТРОГИЕ ПРАВИЛА (обязательны):

1. Не использовать приветствия.
2. Не повторять входные параметры (предмет, класс, стиль).
3. Не писать вступлений.
4. Начинать сразу с "1. Определение".
5. Заполнить ВСЕ пункты структуры.
6. Ответ не считается завершённым, пока не заполнен пункт 5.
7. Если тема не относится к предмету "{subject}" — кратко сообщи об этом и остановись.
8. Не завершать ответ досрочно.
9. Не добавлять лишние разделы.

Предмет: {subject}
Класс: {grade}
Стиль: {style}

Правила уровня:
{grade_rule}

Правила объема:
{level_rule}

Стиль подачи:
{style_rule}

Структура ответа (строго соблюдать порядок):

1. Определение
2. Пошаговое объяснение
3. Пример
4. Краткий вывод (2–3 тезиса)
5. Проверь себя (1 вопрос)

Тема: {topic}
"""


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/app")
def app_page():
    return render_template("app.html")

@app.route("/about")
def about_page():
    return render_template("about.html")

@app.route("/empty")
def empty_page():
    return render_template("empty.html")



@app.route("/generate", methods=["POST"])
def generate():
    ip = _get_ip()

    if not _rate_limit_check(ip):
        return jsonify(error="Слишком много запросов. Подожди немного."), 429

    data = request.get_json(silent=True)
    if not data:
        return jsonify(error="Неверный запрос."), 400

    grade = (data.get("grade") or "").strip()
    subject = (data.get("subject") or "").strip()
    level = (data.get("level") or "").strip()
    style = (data.get("style") or "").strip()
    topic = (data.get("topic") or "").strip()

    if not topic:
        return jsonify(error="Введите тему."), 400

    cache_key = _make_cache_key(grade, subject, level, style, topic)
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(answer=cached, cached=True)

    max_out = MAX_TOKENS_BY_LEVEL.get(level, 800)

    prompt = build_prompt(grade, subject, level, style, topic)

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
            return jsonify(error="Пустой ответ модели."), 500

        _cache_set(cache_key, answer)
        return jsonify(answer=answer, cached=False)

    except Exception as e:
        msg, code = _friendly_error_message(e)
        return jsonify(error=msg), code


if __name__ == "__main__":
    app.run(debug=True)
