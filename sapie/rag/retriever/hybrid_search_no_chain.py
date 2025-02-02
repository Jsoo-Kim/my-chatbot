import jsonlines
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sapie.models.embeddings.load_embeddings import EmbeddingLoader
from sapie.rag.retriever.faiss_retriever import get_faiss_db
# from sapie.rag.retriever.custom_bm25_retriever import BM25Retriever
from sapie.rag.retriever.nochain_bm25_retriever import BM25Retriever
# from sapie.rag.retriever.custom_faiss_retriever import FaissRetriever
from sapie.rag.retriever.nochain_faiss_retriever import FaissRetriever
from kiwipiepy import Kiwi


class HybridRetrieverChain:
    def __init__(self, config, save_csv=False):
        """
        Hybrid Retriever Chain (BM25 + FAISS)

        Args:
            config: 설정 정보 (임베딩 모델, FAISS 경로, 토크나이저 경로, JSONL 경로 포함)
            save_csv (bool): CSV 저장 여부 (기본값: False)
        """
        self._validate_config(config)
        self.config = config
        self.save_csv = save_csv  # 🔹 CSV 저장 옵션

        self.bm25_config = {"k1": 1.5, "b": 0.75, "epsilon": 0.5}  # BM25 설정
        self.bm25_search_k = 5
        self.vectorstore_search_k = 5
        self.hybrid_search_k = 3  # 최종 검색할 문서 개수
        self.context_max_length = 8000  # 최종 컨텍스트 길이 제한
        self.rrf_score = 60  # RRF 설정값

        self.kiwi = Kiwi()  # 형태소 분석기
        self.embedding_loader = EmbeddingLoader(config_file="sapie/configs/config.json")

        # BM25 및 FAISS 초기화
        self._initialize_retrievers()

    def _validate_config(self, config):
        """설정 파일에서 필수 필드 확인"""
        required_fields = ["embedding_model_path", "faiss_path", "tokenizer_path", "jsonl_path"]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"(hybrid_search_chain.py) Missing required configuration fields: {', '.join(missing_fields)}")

    def _initialize_retrievers(self):
        """BM25 및 FAISS 검색기 초기화"""
        train_data = self.read_jsonl(self.config["jsonl_path"])
        corpus = [data["combined_metadata"] for data in train_data]

        # BM25 검색기 초기화
        self.bm25_retriever = BM25Retriever(
            corpus,
            k=self.bm25_search_k,
            preprocess_func=self.kiwi_tokenize,
            bm25_params=self.bm25_config,
        )

        # FAISS 검색기 초기화
        embeddings = self.embedding_loader.get_embedding_model()
        vectorstore = get_faiss_db(embeddings, self.config["faiss_path"])

        self.faiss_retriever = FaissRetriever(
            vectorstore, 
            k=self.vectorstore_search_k
        )

    def _save_to_csv(self, df, filename):
        """CSV 저장 (save_csv=True일 때만 실행)"""
        if not self.save_csv:
            return
        try:
            save_path = f"/home/jskim/data_js/test_241226/sapie/data/csv_results/{filename}"
            df.to_csv(save_path, index=False, encoding="utf-8")
            print(f"✅ CSV 저장 완료: {save_path}")
        except Exception as e:
            print(f"🚨 CSV 저장 오류: {str(e)}")

    def kiwi_tokenize(self, text):
        """형태소 분석을 사용한 토큰화"""
        return [token.form for token in self.kiwi.tokenize(text)]

    def read_jsonl(self, path):
        """JSONL 파일 읽기"""
        with jsonlines.open(path) as f:
            return [line for line in f]

    def calculate_rrf_scores(self, bm25_results, faiss_results):
        """BM25 & FAISS 결과를 RRF 방식으로 결합"""
        bm25_df = bm25_results.sort_values("score", ascending=False).reset_index(drop=True)
        faiss_df = faiss_results.sort_values("score", ascending=True).reset_index(drop=True)

        # RRF 점수 계산
        bm25_df["new_scores_RRF"] = 1 / ((bm25_df.index + 5) + self.rrf_score)
        faiss_df["new_scores_RRF"] = 1 / ((faiss_df.index + 1) + self.rrf_score)

        combined_df = pd.concat([bm25_df, faiss_df]).sort_values("new_scores_RRF", ascending=False).reset_index(drop=True)
        self._save_to_csv(combined_df, "combined_results.csv")  # 🔹 RRF 결과 CSV 저장

        top_n_results = combined_df.drop_duplicates(subset="text").sort_values("new_scores_RRF", ascending=False).reset_index(drop=True)

        # 검색 길이 제한
        total_text_length = top_n_results.head(self.hybrid_search_k)["text_length"].sum()
        print(f"📏 총 문자열 길이: {total_text_length}")

        cumulative_sum = np.cumsum(top_n_results["text_length"])
        first_n = (cumulative_sum >= self.context_max_length).argmax() + 1
        if first_n == 0 or first_n == 1:
            first_n = len(cumulative_sum)
        print(f"📌 선택된 문서 개수: {first_n}")
        top_n_results = top_n_results.head(first_n)

        # 컨텍스트 생성
        result_str = ""
        current_length = 0
        used_indices = []
        for i in range(len(top_n_results)):
            document = top_n_results.loc[i]["text"]
            score = top_n_results.loc[i]["new_scores_RRF"]

            document_str = f"{document} 관련성: {score}\n\n"
            new_length = current_length + len(document_str)
            if new_length > self.context_max_length:
                break

            result_str += document_str
            current_length = new_length
            used_indices.append(i)

        # 사용된 결과 CSV 저장
        used_results = top_n_results.iloc[used_indices]
        self._save_to_csv(used_results, "used_results.csv")

        return result_str

    def retrieve_hybrid_results(self, query):
        """BM25 + FAISS 하이브리드 검색 실행"""
        bm25_results = self.bm25_retriever.retrieve(query)
        faiss_results = self.faiss_retriever.retrieve(query)

        # RRF 기반 최적 결과 계산
        final_context = self.calculate_rrf_scores(bm25_results, faiss_results)
        return final_context
