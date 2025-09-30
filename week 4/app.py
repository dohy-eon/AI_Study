from flask import Flask, request, jsonify, send_from_directory
import ollama
from flask_cors import CORS
import os
import re

app = Flask(__name__)
CORS(app)

model_name = "exaone3.5:2.4b"
chat_history = []
history_limit = 10

@app.route('/')
def index():
    return send_from_directory(os.path.dirname(__file__), 'chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    data = request.get_json()
    question = data.get('question', '')
    story = data.get('story', '')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    # Ollama에 퀘스트 스토리와 질문을 함께 전달
    user_content = f"퀘스트: {story}\n질문: {question}"
    chat_history.append({"role": "user", "content": user_content})
    if len(chat_history) > history_limit:
        chat_history = chat_history[-history_limit:]
    response = ollama.chat(
        model=model_name,
        messages=chat_history
    )
    answer = response["message"]["content"].strip()
    # 강조문자, 별표, 마크다운 등 제거
    def clean_text(text):
        # **강조** 제거
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        # *강조* 제거
        text = re.sub(r"\*(.*?)\*", r"\1", text)
        # __강조__ 제거
        text = re.sub(r"__([^_]+)__", r"\1", text)
        # _강조_ 제거
        text = re.sub(r"_([^_]+)_", r"\1", text)
        # 마크다운 헤더/리스트/기호 등 단순화
        text = re.sub(r"^#+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[-*] ", "", text, flags=re.MULTILINE)
        # 여러 개의 빈 줄을 하나로
        text = re.sub(r"\n{2,}", "\n\n", text)
        # 문장 끝에 줄바꿈 추가 (마침표, 느낌표, 물음표)
        text = re.sub(r"([.!?]) +", r"\1\n", text)
        return text.strip()
    answer = clean_text(answer)
    chat_history.append({"role": "assistant", "content": answer})
    if len(chat_history) > history_limit:
        chat_history = chat_history[-history_limit:]
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
