from typing import List, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import pandas as pd

class FaissRetriever(BaseRetriever):
    vectorstore: Any  # Any:어떤 타입이든 허용
    k: int = 5  # 기본 검색 개수

    @classmethod
    def from_db(cls, vectorstore, **kwargs):
        """
        기존 FAISS vectorstore로부터 FaissRetriever를 생성합니다.

        Args:
            vectorstore: FAISS vectorstore 인스턴스
            **kwargs: Retriever에 전달할 추가 인자들

        Returns:
            FaissRetriever 인스턴스

        예시:
            db = FAISS.load_local(...)
            retriever = FaissRetriever.from_db(db, k=5)
        """
        return cls(
            vectorstore=vectorstore,
            **kwargs
        )


   ## score반환 하도록 수정 
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> pd.DataFrame:
        doc_list = []
        doc_scores_list = []
        
        # similarity_search_with_score 사용하여 문서와 점수 획득
        search_results  = self.vectorstore.similarity_search_with_score(query, k=self.k)
        # search_results[0] => (Document(metadata={'source':'출처','type':'table','page':'1'},page_content='문서내용'),score)
        
        for document, score in search_results:
            # print(i)
            tmp = ''
            page_content = document.page_content.strip()
            metadata = document.metadata  
            # print(metadata_tmp)
            metadata_source = metadata['source'].strip()
            tmp += metadata_source+'\n' 
        
            if metadata['type'] == 'table':
                tmp +=  metadata['before_table_text'].strip()+'\n'
            tmp += '내용: ' + page_content
            
            doc_scores_list.append(score)
            doc_list.append(tmp)
        
        result_vectors = pd.DataFrame({
            'type':'vectors',
            'text': doc_list,
            'score': doc_scores_list,
        })
        result_vectors['text_length'] = result_vectors['text'].str.len() # text 길이 컬럼 추가

        # 정렬 및 타입 추가
        result_vectors = result_vectors.sort_values('score', ascending=True).reset_index(drop=True)

        # 질문하기
        result_vectors = result_vectors.loc[result_vectors.score>0]

        # CSV 저장(확인용)
        # result_vectors.to_csv('/home/jskim/data_js/sapie_project2024_backend_sglang/combined_results.csv', index=False, encoding='utf-8')  # 한글 깨짐 방지        
        try:
            save_path = './sapie/data/csv_results/sapie_faiss_results.csv'
            result_vectors.to_csv(save_path, index=False, encoding='utf-8')  # 한글 깨짐 방지        
        except Exception as e:
            print(f"{save_path}파일 저장 중 오류가 발생했습니다: {str(e)}")
        
        return result_vectors
