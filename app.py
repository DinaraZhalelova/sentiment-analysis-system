import os, sqlite3, uuid
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session

ROOT = Path(__file__).resolve().parent
DB = ROOT / "history.db"
MODEL_NAME = os.getenv("SENTIMENT_MODEL", "cointegrated/rubert-tiny-sentiment-balanced")
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "local-development-secret-change-me")
_pipeline = None
_model_checked = False

def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS analyses (
          id INTEGER PRIMARY KEY AUTOINCREMENT, sid TEXT NOT NULL, text TEXT NOT NULL,
          sentiment TEXT NOT NULL, confidence REAL NOT NULL, method TEXT NOT NULL, created TEXT NOT NULL)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_sid ON analyses(sid, id DESC)")

def sid():
    if "sid" not in session: session["sid"] = str(uuid.uuid4())
    return session["sid"]

def model():
    global _pipeline, _model_checked
    if not _model_checked:
        _model_checked = True
        try:
            from transformers import pipeline
            _pipeline = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME, device=-1)
        except Exception:
            _pipeline = None
    return _pipeline

POS = "хорошо хороший хорошая хорошее отлично отличный отличная прекрасный прекрасная люблю нравится доволен довольна счастлив рад рада классно супер замечательно спасибо благодарю красиво великолепно приятно лучший лучшая рекомендую удобно качественный качественная понравился понравилась".split()
NEG = "плохо плохой плохая ужасно ужасный ненавижу разочарован разочарована кошмар отвратительно ошибка проблема проблемы сломалось неудобно медленно дорого обман ужас худший худшая бесполезно злой злая провал неудачно".split()

def fallback(text):
    t = text.lower().replace("ё", "е")
    p = sum(t.count(w) for w in POS)
    n = sum(t.count(w) for w in NEG)
    if "не нравится" in t or "не рекомендую" in t or "не хорошо" in t: n += 1
    if p > n: return "Положительная", min(.60 + .08*(p-n), .92), "Резервный словарный анализ"
    if n > p: return "Отрицательная", min(.60 + .08*(n-p), .92), "Резервный словарный анализ"
    return "Нейтральная", .55, "Резервный словарный анализ"

def analyze(text):
    m = model()
    if m:
        try:
            r = m(text[:4000], truncation=True, max_length=512)[0]
            label = str(r.get("label", "")).upper()
            if "POS" in label or label in ("LABEL_2", "2"): sentiment = "Положительная"
            elif "NEG" in label or label in ("LABEL_0", "0"): sentiment = "Отрицательная"
            else: sentiment = "Нейтральная"
            return sentiment, float(r.get("score", 0)), "Модель " + MODEL_NAME
        except Exception: pass
    return fallback(text)

@app.get("/")
def index():
    sid()
    return render_template("index.html")

@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str): return jsonify(error="Введите текст строкой."), 400
    text = text.strip()
    if not text: return jsonify(error="Введите текст для анализа."), 400
    if len(text) > 4000: return jsonify(error="Максимальная длина текста — 4000 символов."), 400
    sentiment, confidence, method = analyze(text)
    created = datetime.now().astimezone().strftime("%d.%m.%Y %H:%M")
    with db() as c:
        cur = c.execute("INSERT INTO analyses(sid,text,sentiment,confidence,method,created) VALUES(?,?,?,?,?,?)",
                        (sid(), text, sentiment, confidence, method, created))
        aid = cur.lastrowid
    return jsonify(id=aid, text=text, sentiment=sentiment, confidence=round(confidence*100,1), method=method, created=created)

@app.get("/api/history")
def history():
    with db() as c:
        rows = c.execute("SELECT * FROM analyses WHERE sid=? ORDER BY id DESC LIMIT 30", (sid(),)).fetchall()
    return jsonify([dict(id=r["id"], text=r["text"], sentiment=r["sentiment"], confidence=round(r["confidence"]*100,1),
                         method=r["method"], created=r["created"]) for r in rows])

@app.delete("/api/history")
def clear_history():
    with db() as c: c.execute("DELETE FROM analyses WHERE sid=?", (sid(),))
    return jsonify(ok=True)

@app.get("/api/status")
def status():
    return jsonify(mode="model" if model() else "fallback", model=MODEL_NAME)

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False)
