from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from datetime import datetime

# API 키 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


# 날짜 형식 변환 함수
def convert_date_format(date_str, current_format, target_format):
    try:
        date_obj = datetime.strptime(date_str, current_format)
        return date_obj.strftime(target_format)
    except Exception as e:
        return f"날짜 변환 오류: {e}"


# 숫자 덧셈 함수
def add_numbers(x, y):
    try:
        return float(x) + float(y)
    except Exception as e:
        return f"덧셈 오류: {e}"


# tools 정의
tools = [
    {
        "type": "function",
        "function": {
            "name": "convert_date_format",
            "description": "날짜 형식을 변환합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_str": {"type": "string"},
                    "current_format": {"type": "string"},
                    "target_format": {"type": "string"},
                },
                "required": ["date_str", "current_format", "target_format"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "두 숫자를 더합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                },
                "required": ["x", "y"]
            }
        }
    }
]


# OpenAIAgent 클래스 정의
class OpenAIAgent:
    def __init__(self):
        self.messages = []

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})

        # GPT 첫 호출 (함수 호출 여부 판단)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=self.messages,
            tools=tools,
            tool_choice="auto"
        )

        message = response.choices[0].message
        # 함수 호출이 포함된 경우
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # 사용자 정의 함수 호출
            if func_name == "convert_date_format":
                result = convert_date_format(**args)
            elif func_name == "add_numbers":
                result = add_numbers(**args)
            else:
                result = "알 수 없는 함수 호출입니다."

            # 결과를 messages에 추가
            self.messages.append({"role": "assistant", "tool_calls": [tool_call]})
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

            # 최종 응답 생성
            self.messages.append({
                "role": "system",
                 # 간결한 응답 유도
                "content": "함수 결과를 최대한 그대로 사용하고, 불필요한 부연 설명은 하지 말고 간결하게 답변하세요."
            })

            final_response = client.chat.completions.create(
                model="gpt-4",
                messages=self.messages
            )
            final_message = final_response.choices[0].message.content
            print("\n함수 응답:", final_message)
            print()
        # 함수 호출이 없으면 일반 GPT 응답 출력
        else:
            print("\nGPT 응답:", message.content)
            print()
            self.messages.append({"role": "assistant", "content": message.content})


# 메인 실행 루프
def main():
    agent = OpenAIAgent()
    print("종료하려면 'exit' 입력\n")

    while True:
        user_input = input("질문을 입력하세요: ")
        if user_input.lower() == "exit":
            print("프로그램 종료")
            break
        agent.chat(user_input)


if __name__ == "__main__":
    main()