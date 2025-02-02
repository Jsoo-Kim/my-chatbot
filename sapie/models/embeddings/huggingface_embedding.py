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