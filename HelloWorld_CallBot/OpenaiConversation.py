class OpenaiConversation:
    def __init__(self, openai_client,user_speech):
        self.conversations = {}
        self.openai_client = openai_client
        self.user_input = user_speech

    def get_response(self, caller_id):
        # 해당 발신자의 대화 기록이 없으면 새로 생성
        if caller_id not in self.conversations:
            self.conversations[caller_id] = []

        # 사용자 입력을 대화 기록에 추가
        self.conversations[caller_id].append({"role": "user", "content": self.user_input})

        # OpenAI API 호출
        try:
            response = self.openai_client.create(
                model="gpt-3.5-turbo-0125",  # 또는 다른 적절한 모델
                messages=[
                    {"role": "system", "content": "당신은 외국인 근로자를 위한 상담 AI입니다. 한국어로 응답해주세요."},
                    *self.conversations[caller_id]
                ]
            )

            # AI 응답 추출
            ai_response = response['choices'][0]['message']['content']

            # AI 응답을 대화 기록에 추가
            self.conversations[caller_id].append({"role": "assistant", "content": ai_response})

            return ai_response

        except Exception as e:
            print(f"OpenAI API 호출 중 오류 발생: {str(e)}")
            return "죄송합니다. 현재 서비스에 문제가 있습니다. 잠시 후 다시 시도해 주세요."

    def end_conversation(self, caller_id):
        # 대화 종료 시 해당 발신자의 대화 기록 삭제
        if caller_id in self.conversations:
            del self.conversations[caller_id]
        return "대화가 종료되었습니다. 이용해 주셔서 감사합니다."