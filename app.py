from flask import Flask, request, jsonify, send_from_directory
from rag_engine import MedicalRAGEngine
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(BASE_DIR, "static") if os.path.isdir(os.path.join(BASE_DIR, "static")) else BASE_DIR

app = Flask(__name__, static_folder=static_dir)
engine = MedicalRAGEngine()


@app.route("/")
def index():
    return send_from_directory(static_dir, "index.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"type": "error", "message": "Please enter a question."}), 400
    result = engine.answer(query)
    return jsonify(result)


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify({
        "total_qa_pairs": len(engine.df),
        "sources": engine.df["source"].value_counts().to_dict(),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
