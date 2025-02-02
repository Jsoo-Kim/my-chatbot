import requests
from sapie.models.llm.base_llm import BaseLLM

class OllamaService(BaseLLM):  ### 임시 작성! 테스트 필요 
    """
    Ollama API를 사용하여 LLM 호출을 처리하는 클래스
    """
    def __init__(self, base_url: str, api_key: str, model_path: str, streaming: bool):
        super().__init__(base_url, api_key, model_path, streaming)
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def call_api(self, messages: list):
        """
        Ollama API 호출 메서드
        Args:
            messages (list): 사용자 입력 메시지
        Returns:
            generator or str: Ollama 응답 (스트리밍 또는 단일 응답)
        """
        # Ollama는 OpenAI와 달리 prompt 형식이 단순하므로 조정 필요
        prompt = self._build_prompt(messages)
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model_path,
            "prompt": prompt
        }

        if self.streaming:
            # 스트리밍 방식 요청
            response = requests.post(url, json=payload, headers=self.headers, stream=True)
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        # 스트리밍된 JSON 데이터 파싱
                        chunk = line.decode("utf-8")
                        yield chunk
            else:
                raise RuntimeError(f"Streaming failed with status code {response.status_code}")
        else:
            # 단일 응답 방식 요청
            response = requests.post(url, json=payload, headers=self.headers)
            if response.status_code == 200:
                return response.json().get("content", "")
            else:
                raise RuntimeError(f"Request failed with status code {response.status_code}")

    def _build_prompt(self, messages: list) -> str:
        """
        Ollama API에 맞는 프롬프트 빌더
        Args:
            messages (list): 사용자 메시지 리스트
        Returns:
            str: Ollama 호환 프롬프트
        """
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        return prompt.strip()
