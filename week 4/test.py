import ollama

# 사용할 모델 이름
model_name = "exaone3.5:2.4b"

# 대화 히스토리 (리스트로 관리)
chat_history = []
history_limit = 10  # 최대 10개만 유지

def ask(question):
    global chat_history

    # 사용자 메시지 추가
    chat_history.append({"role": "user", "content": question})

    # 히스토리 길이 제한
    if len(chat_history) > history_limit:
        chat_history = chat_history[-history_limit:]

    # Ollama 호출
    response = ollama.chat(
        model=model_name,
        messages=chat_history
    )

    answer = response["message"]["content"].strip()

    # 어시스턴트 답변도 히스토리에 추가
    chat_history.append({"role": "assistant", "content": answer})

    # 히스토리 길이 제한
    if len(chat_history) > history_limit:
        chat_history = chat_history[-history_limit:]

    return answer

if __name__ == "__main__":
    print("💬 Ollama 대화 시작 (종료하려면 'exit' 입력)")
    while True:
        q = input("\n나: ")
        if q.lower() in ["exit", "quit", "종료"]:
            print("👋 대화를 종료합니다.")
            break
        answer = ask(q)
        print(f"AI: {answer}")


# import ollama

# # 사용할 모델 이름
# model_name = "exaone3.5:2.4b"

# def ask(question):
#     response = ollama.chat(
#         model=model_name,
#         messages=[{"role": "user", "content": question}]
#     )
#     return response["message"]["content"].strip()

# if __name__ == "__main__":
#     q = input("질문을 입력하세요: ")
#     answer = ask(q)
#     print("\n=== 답변 ===\n")
#     print(answer)
