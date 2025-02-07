from abc import ABC, abstractmethod
import openai

class OpenAIService():
    def __init__(self, base_url: str, api_key: str, model_path: str, streaming: bool):
        # super().__init__(base_url, api_key, model_path, streaming)
        self.base_url = base_url
        self.api_key = api_key
        self.model_path = model_path
        self.streaming = streaming
        self.client = openai.Client(base_url=base_url, api_key=api_key)

    @abstractmethod
    def call_api(self, messages: list):
        """
        OpenAI-Compatible API 호출 메서드 (추상 메서드)
        """
        pass

