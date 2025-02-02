import pandas as pd
from rank_bm25 import BM25Okapi

class BM25Retriever:
    """BM25 검색기 (LangChain 제거, vectorizer 내부 생성)"""

    def __init__(self, corpus, k=4, preprocess_func=None, bm25_params=None):
        """
        BM25 검색기 초기화

        Args:
            corpus (List[str]): 검색할 문서 리스트
            k (int): 검색할 상위 문서 개수
            preprocess_func (callable): 텍스트 전처리 함수
            bm25_params (dict): BM25 파라미터 설정
        """
        self.corpus = corpus
        self.k = k
        self.preprocess_func = preprocess_func if preprocess_func else lambda x: x.split()
        
        # 🔹 텍스트 전처리
        processed_corpus = [self.preprocess_func(text) for text in corpus]

        # 🔹 BM25Okapi 초기화 (vectorizer 직접 생성)
        bm25_params = bm25_params or {}
        self.vectorizer = BM25Okapi(processed_corpus, **bm25_params)

    def retrieve(self, query: str) -> pd.DataFrame:
        """BM25 검색 실행 (LangChain 제거)"""
        processed_query = self.preprocess_func(query)
        doc_scores = self.vectorizer.get_scores(processed_query)

        result_bm25 = pd.DataFrame({
            'type': 'bm25',
            'text': self.corpus,
            'score': doc_scores,
            'text_length': [len(text) for text in self.corpus]
        }).sort_values('score', ascending=False).reset_index(drop=True)

        try:
            save_path = './sapie/data/csv_results/sapie_bm25_results.csv'
            result_bm25.to_csv(save_path, index=False, encoding='utf-8')  # 한글 깨짐 방지        
        except Exception as e:
            print(f"{save_path} 파일 저장 중 오류가 발생했습니다: {str(e)}")

        return result_bm25.loc[result_bm25.score > 0].head(self.k)
