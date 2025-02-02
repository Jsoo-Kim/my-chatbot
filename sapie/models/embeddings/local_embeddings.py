import boto3
from langchain_aws import BedrockEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

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
    - BedrockEmbeddings: 초기화된 임베딩 객체.
    """
    bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=region)
    return BedrockEmbeddings(model_id=model_id, client=bedrock_runtime)
