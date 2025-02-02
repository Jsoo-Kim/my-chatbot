import openai
from sapie.models.llm.base_llm import BaseLLM

class OpenAIService(BaseLLM):
    def __init__(self, base_url: str, api_key: str, model_path: str, streaming: bool):
        super().__init__(base_url, api_key, model_path, streaming)
        self.client = openai.Client(base_url=base_url, api_key=api_key)

    def call_api(self, messages: list):
        """
        OpenAI-Compatible API 호출
        """
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=0.0,
            max_tokens=10000,
            top_p=0.7,
            presence_penalty=0.2,
            stream=self.streaming,
        )
        if self.streaming:
            # 스트리밍 데이터 생성 (제너레이터)
            for chunk in response:
                if chunk.choices[0].delta.content:
                    # print(f"openai_service 쪽 청크: {chunk.choices[0].delta.content}")
                    yield chunk.choices[0].delta.content
        else:
            # 단일 응답 처리
            return response.choices[0].message.content
