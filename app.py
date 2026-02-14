from flask import Flask, render_template, request, jsonify
from google import genai

app = Flask(__name__)

import os
from google import genai

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1"}
)

MODEL_NAME = "gemini-2.5-flash"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json(silent=True)
    if not data:
        return jsonify(error="Запрос без JSON."), 400

    grade = (data.get("grade") or "").strip()
    subject = (data.get("subject") or "").strip()
    level = (data.get("level") or "").strip()
    style = (data.get("style") or "").strip()
    topic = (data.get("topic") or "").strip()

    if not topic:
        return jsonify(error="Введите тему."), 400

    length_map = {
        "Кратко": "8–12 предложений",
        "Средне": "2–3 абзаца",
        "Подробно": "4–6 абзацев с примерами"
    }
    length_hint = length_map.get(level, "2–3 абзаца")

    style_map = {
        "Простым языком": "максимально просто, без сложных терминов",
        "Как учитель": "как учитель в классе, структурировано",
        "Научно": "более научно, но понятно школьнику"
    }
    style_hint = style_map.get(style, "понятно школьнику")

    prompt = f"""
Ты — AI School Helper.
Класс: {grade}
Предмет: {subject}
Тема: {topic}

Требования:
- Объём: {length_hint}
- Стиль: {style_hint}
- Начни с определения.
- Добавь 1–2 примера.
- В конце задай 2 вопроса для самопроверки.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        answer = response.text.strip() if response.text else ""
        if not answer:
            return jsonify(error="Пустой ответ от модели."), 500

        return jsonify(answer=answer)

    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True)
