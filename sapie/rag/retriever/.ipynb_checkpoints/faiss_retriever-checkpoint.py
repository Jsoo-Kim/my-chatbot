from langchain_community.vectorstores import FAISS

def get_faiss_db(embeddings, faiss_path):
    """
    FAISS 리트리버 초기화

    Args:
    - embeddings: 임베딩 객체 (HuggingFace 또는 Bedrock)
    - faiss_path (str): FAISS 인덱스 파일 경로

    Returns:
    - FAISS Retriever: 초기화된 리트리버 객체
    """
    db = FAISS.load_local(
        folder_path=faiss_path,
        index_name="index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return db
