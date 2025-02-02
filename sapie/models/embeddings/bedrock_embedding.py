# bedrock_embedding.py : Bedrock Embeddings 초기화

import boto3
from langchain_aws import BedrockEmbeddings

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
