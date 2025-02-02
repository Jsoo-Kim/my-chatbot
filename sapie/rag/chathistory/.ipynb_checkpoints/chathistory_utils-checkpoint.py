from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,trim_messages
from transformers import AutoTokenizer
from typing import List


class ChathistoryUtil:
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("tokenizer_path", None))
        self.chat_history_max_token = config.get("chat_history_max_token", 1500)

    def get_trimmer(self):
        return trim_messages(
                # 토큰 수를 계산하는 함수 지정
                token_counter=self.custom_token_counter,
                
                # "last": 가장 최근 메시지부터 유지
                # "first": 가장 오래된 메시지부터 유지
                strategy="last",  
                
                # 유지할 최대 토큰 수
                max_tokens=self.chat_history_max_token,
                
                # 어떤 역할의 메시지부터 시작할지 지정
                # "human": 사용자 메시지부터 시작
                # "ai": AI 메시지부터 시작
                start_on="human",
                
                # 어떤 역할의 메시지에서 끝낼지 지정
                # "ai": AI 메시지에서 종료
                # "human": 사용자 메시지에서 종료
                end_on=("ai"),
                
                # 시스템 프롬프트 포함 여부
                # True: 시스템 프롬프트 유지
                # False: 시스템 프롬프트 제외
                include_system=True,
        )


    def custom_token_counter(self, messages: List[BaseMessage]) -> int:
        # 사용중인 LLM의 토크나이저 로드
        tokenizer = self.tokenizer
        num_tokens = 0
        for msg in messages:
            # 메시지 타입에 따른 역할 구분
            if isinstance(msg, HumanMessage):
                prefix = "Human: "
            elif isinstance(msg, AIMessage):
                prefix = "Assistant: "
            elif isinstance(msg, SystemMessage):
                prefix = "System: "
            elif isinstance(msg, ToolMessage):
                prefix = "Tool: "
            else:
                prefix = ""
                
            # 전체 메시지 구성 (prefix + content)
            full_message = prefix + str(msg.content)
            
            # 해당 LLM의 토크나이저로 토큰 수 계산
            tokens = len(tokenizer.encode(full_message))
            num_tokens += tokens
            # print(f"token수는 {num_tokens}")
        # print(f"총 chat_history의 토큰수는 {num_tokens}")    
        return num_tokens
