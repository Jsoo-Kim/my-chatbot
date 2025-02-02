class CustomMemoryChatHistory:
    def __init__(self):
        self.chat_history = []

    def add_message(self, role, content):
        """대화 기록에 메시지 추가"""
        self.chat_history.append({"role": role, "content": content})

    def get_messages(self):
        """대화 기록 반환"""
        return self.chat_history