import asyncio
from sapie.models.embeddings.embedding_initializers import (
    get_huggingface_embeddings,
    get_bedrock_embeddings,
    get_openai_embeddings
)
from sapie.models.embeddings.load_config import LoadConfig
from typing import List


class EmbeddingLoader:
    """
    다양한 임베딩 엔진(HuggingFace, Bedrock, OpenAI, Custom API)을 초기화하는 클래스.
    
    사용 예시:
        loader = EmbeddingLoader(config_file="config.json")
        embeddings = loader.embedding_model  # 초기화된 임베딩 객체를 사용

    Args:
    - config_file (str): JSON 형식의 설정 파일 경로. 설정 파일 예시는 다음과 같음:
        {
            "type": "huggingface",  // "bedrock", "openai", "custom_api"
            "model_name": "dragonkue/BGE-m3-ko",
            "device": "cuda",  // HuggingFace
            "region": "us-west-2",  // Bedrock
            "api_url": "http://localhost:5407",  // OpenAI or Custom API
            "api_key": "SaltwareSapie"  // OpenAI
        }
    """
    def __init__(self, config_file: str):
        self.config = LoadConfig.load_config(config_file)
        self.type = self.config["type"]
        # 임베딩 초기화
        self.embedding_model = self.get_embedding_model()

    def get_embedding_model(self):
        """
        설정 파일을 기반으로 적절한 임베딩 엔진 초기화

        Returns:
        - 초기화된 임베딩 객체
        """
        if self.type == "huggingface":
            huggingface_config = self.config.get("huggingface", {})
            return get_huggingface_embeddings(
                model_name=huggingface_config.get("model_name", "default_model"),
                device=huggingface_config.get("device", "cpu"),
            )
        elif self.type == "bedrock":
            bedrock_config = self.config.get("bedrock", {})
            return get_bedrock_embeddings(
                model_id=bedrock_config.get("model_id", "default_model_id"),
                region=bedrock_config.get("region", "us-west-2"),
            )
        elif self.type == "openai":
            openai_config = self.config.get("openai", {})
            return get_openai_embeddings(
                api_url=openai_config.get("api_url", "http://localhost:8000"),
                api_key=openai_config.get("api_key", "default_api_key"),
                model_name=openai_config.get("model_name", "default_model_name"),
                local_path=openai_config.get("local_path", ""),
            )
        else:
            raise ValueError(f"Unsupported embedding type: {self.type}")
