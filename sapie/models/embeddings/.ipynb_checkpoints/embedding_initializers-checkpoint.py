from langchain_huggingface import HuggingFaceEmbeddings
from langchain_aws import BedrockEmbeddings
from langchain_openai.embeddings.base import OpenAIEmbeddings
import boto3


def get_huggingface_embeddings(model_name="dragonkue/BGE-m3-ko", device="cpu"):
    """
    HuggingFace 임베딩 초기화

    Args:
    - model_name (str): 사용할 모델 이름
    - device (str): 실행 디바이스 ("cpu" 또는 "cuda")

    Returns:
    - HuggingFaceEmbeddings: 초기화된 임베딩 객체
    """
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_bedrock_embeddings(model_id="amazon.titan-embed-text-v1", region="us-west-2"):
    """
    Bedrock Embeddings 초기화

    Args:
    - model_id (str): 사용할 Bedrock 모델 ID.
    - region (str): AWS 리전 (기본값: "us-west-2").

    Returns:
    - BedrockEmbeddings: 초기화된 임베딩 객체
    """
    bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region)
    return BedrockEmbeddings(model_id=model_id, client=bedrock_runtime)


def get_openai_embeddings(
    api_url="http://172.20.1.243:5407/v1",
    api_key="SaltwareSapie",
    model_name="bge-m3-ko",
    local_path="/home/jskim/data_js/test_241226/models/local_models/bge-m3-ko",
):
    """
    OpenAI 방식 임베딩 초기화

    Args:
    - api_url (str): OpenAI API URL (로컬 또는 원격)
    - api_key (str): OpenAI API 키
    - model_name (str): 사용할 모델 이름
    - local_path (str): tiktoken 모델 경로

    Returns:
    - OpenAIEmbeddings: 초기화된 임베딩 객체
    """
    return OpenAIEmbeddings(
        openai_api_base=api_url,
        api_key=api_key,
        model=model_name,
        # tiktoken_enabled=bool(local_path),
        tiktoken_enabled=False,
        tiktoken_model_name=local_path,
    )


# async def get_custom_api_embeddings(api_url: str, model_name: str, texts: List[str]) -> List[List[float]]:
#     """
#     Custom API를 통해 임베딩 생성

#     Args:
#     - api_url (str): Custom API 엔드포인트 URL
#     - model_name (str): 사용할 모델 이름
#     - texts (List[str]): 입력 텍스트 리스트

#     Returns:
#     - List[List[float]]: 생성된 임베딩 벡터 리스트
#     """
#     async with httpx.AsyncClient() as client:
#         response = await client.post(
#             f"{api_url}/v1/embeddings",
#             json={"input": texts, "model": model_name},
#         )
#         if response.status_code != 200:
#             raise ValueError(f"API 요청 실패: {response.json()}")
#         return [item["embedding"] for item in response.json()["data"]]