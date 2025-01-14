from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer
from embedding.huggingface import get_huggingface_embeddings
from retriever.faiss_retriever import get_faiss_db
from langchain_core.runnables import RunnableLambda
from operator import itemgetter


class FaissRetrieverChain:
    def __init__(self, config):
        """
            config: {
                "embedding_model_path": "/home/jskim/data_js/vllm/code/models/bge-m3-ko",
                "faiss_path": "/home/jskim/data_js/vllm/code/faiss_db/faiss_BGE-m3-ko_text_512_table_4090", 
            }
        """
        # 필수 설정 체크
        required_fields = ["embedding_model_path", "faiss_path"]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"(faiss_retriever_chain.py) Missing required configuration fields: {', '.join(missing_fields)}")
        
        self.config = config
        self.search_k = 3  # 검색할 문서 수
        self.fetch_k = 10   # MMR을 위해 가져올 후보 문서 수

    def format_docs(self, docs):
        """
        검색된 문서 결과를 포맷
        """
        return "\n".join(f"제목: {doc.metadata['source']}\n내용: {doc.page_content}" for doc in docs)
    
    def create_retrieval_chain(self):
        """
        FAISS 검색 체인 생성
        """
        try:
            # 임베딩 모델 로드
            embeddings = get_huggingface_embeddings(model_name=self.config["embedding_model_path"])
            
            # FAISS 벡터 스토어 로드
            vectorstore = get_faiss_db(embeddings, self.config["faiss_path"])
            
            # Retriever 설정
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": self.search_k,
                    "fetch_k": self.fetch_k
                }
            )

            # 검색 체인 생성
            retrieval_chain = itemgetter("query") | retriever | self.format_docs

            return retrieval_chain

        except Exception as e:
            print(f"Error creating retrieval chain: {e}")
            raise
        
# if __name__ == "__main__":
  
#     faiss_retriever_chain_config = {
#         "embedding_model_path": "dragonkue/BGE-m3-ko",
#         "faiss_path": "/data_sm/codeset/github/sapie_project2024_backend_sglang/data/text_512_table_4096", 
#         "tokenizer_path":"/data_sm/models/Qwen/Qwen2.5-32B-Instruct-AWQ",
#         }
#     faiss = FaissRetrieverChain(faiss_retriever_chain_config)
#     print("==========")
#     chain = faiss.create_retrieval_chain()
#     result = chain.invoke({"query":"담당자"})
#     print(result)