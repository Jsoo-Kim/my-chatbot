from sapie.models.llm.load_config import LoadConfig
from sapie.models.llm.openai_service import OpenAIService
from sapie.models.llm.vllm_service import VLLMService
from sapie.models.llm.sglang_service import SGLangService
from sapie.models.llm.ollama_service import OllamaService

class LLMLoader:
    def __init__(self, config_file: str):
        """
        LLM 로더 초기화
        Args:
        - inference_type (str): 사용할 LLM 타입 (vllm, sglang, ollama, etc.)
        - config_file (str): 설정 파일 경로
        """
        config = LoadConfig.load_config(config_file)
        self.type = config["type"]
        self.base_url = config[self.type]["base_url"]
        self.api_key = config[self.type]["api_key"]
        self.model_path = config[self.type]["default_model"]
        self.streaming = config[self.type]["streaming"]

    def get_llm_instance(self):
        """
        LLM 인스턴스 생성 및 반환
        """
        if self.type == "openai":
            return OpenAIService(self.base_url, self.api_key, self.model_path, self.streaming)
        elif self.type == "vllm":
            return VLLMService(self.base_url, self.api_key, self.model_path, self.streaming)
        elif self.type == "sglang":
            return SGLangService(self.base_url, self.api_key, self.model_path, self.streaming)
        elif self.type == "ollama":
            return OllamaService(self.base_url, self.api_key, self.model_path, self.streaming)
        else:
            raise ValueError(f"Unsupported Inference type: {self.type}")
