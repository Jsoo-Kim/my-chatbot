from __future__ import annotations
<<<<<<< HEAD
from typing import Any, Callable, Dict, Iterable, List, Optional
=======

from typing import Any, Callable, Dict, Iterable, List, Optional

>>>>>>> main
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from pydantic import Field  
from langchain_core.retrievers import BaseRetriever
import numpy as np
import pandas as pd
import os
## 원본 코드
#https://api.python.langchain.com/en/latest/_modules/langchain_community/retrievers/bm25.html#BM25Retriever


def default_preprocessing_func(text: str) -> List[str]:
    return text.split()


<<<<<<< HEAD
=======

>>>>>>> main
class BM25Retriever(BaseRetriever):
    """`BM25` retriever without Elasticsearch."""

    vectorizer: Any
    """ BM25 vectorizer."""
    docs: List[Document] = Field(repr=False)
    """ List of documents."""
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], List[str]] = default_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    ## 추가 변수:원본 문서
    corpus : List[str]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BM25Retriever:
        """
        Create a BM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )
        
        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        
        docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, corpus=texts, **kwargs
        )

    ## 현재 사용안함 

    # @classmethod
    # def from_documents(
    #     cls,
    #     documents: Iterable[Document],
    #     *,
    #     bm25_params: Optional[Dict[str, Any]] = None,
    #     preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
    #     **kwargs: Any,
    # ) -> BM25Retriever:
    #     """
    #     Create a BM25Retriever from a list of Documents.
    #     Args:
    #         documents: A list of Documents to vectorize.
    #         bm25_params: Parameters to pass to the BM25 vectorizer.
    #         preprocess_func: A function to preprocess each text before vectorization.
    #         **kwargs: Any other arguments to pass to the retriever.

    #     Returns:
    #         A BM25Retriever instance.
    #     """
    #     texts, metadatas = zip(*((d.page_content, d.metadata) for d in documents))
    #     return cls.from_texts(
    #         texts=texts,
    #         bm25_params=bm25_params,
    #         metadatas=metadatas,
    #         preprocess_func=preprocess_func,
    #         **kwargs,
    #     )

    ## Chain invoke시 호출되는 함수
    ## score반환 하도록 수정 
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> pd.DataFrame: # 반환 타입을 DataFrame
        processed_query = self.preprocess_func(query)
        # get_scores로 점수 획득
        doc_scores_bm25 = self.vectorizer.get_scores(processed_query)

        # 재윤님 코드
        result_bm25 = pd.DataFrame({
            'type' :'bm25',
            'text': self.corpus,
            'score': doc_scores_bm25,
            'text_length': [len(text) for text in self.corpus]  # text 길이 컬럼 추가
        })
        # ### 내림차순
        result_bm25 = result_bm25.sort_values(['score'], ascending=False).reset_index(drop=True)
        result_bm25 = result_bm25.loc[result_bm25.score>0]
        # print(result_bm25)

        #상위 k개 자르기
        # result_bm25 = pd.DataFrame({
        #     'type' :'bm25',
        #     'text': self.corpus,
        #     'score': doc_scores_bm25,
        #     'text_length': [len(text) for text in self.corpus]  # text 길이 컬럼 추가
        # }).sort_values('score', ascending=False).head(self.k)

        # CSV 저장(확인용)
        # output_dir = '/data/data_sm/csv/jsm/jsm_bm25_results.csv'
        # # os.makedirs(output_dir, exist_ok=True) 
        # result_bm25.to_csv(output_dir, index=False, encoding='utf-8')  # 한글 깨짐 방지
        # result_bm25.to_csv('/home/jskim/data_js/sapie_project2024_backend_sglang/combined_results.csv', index=False, encoding='utf-8')  # 한글 깨짐 방지        
        
        
        try:
            save_path = './sapie/data/csv_results/sapie_bm25_results.csv'
            result_bm25.to_csv(save_path, index=False, encoding='utf-8')  # 한글 깨짐 방지        
        except Exception as e:
            print(f"{save_path}파일 저장 중 오류가 발생했습니다: {str(e)}")

<<<<<<< HEAD
        return result_bm25
=======
        return result_bm25
    
    
>>>>>>> main
