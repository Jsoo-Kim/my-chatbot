import pandas as pd


class FaissRetriever:
    """FAISS 검색기 (LangChain 제거)"""

    def __init__(self, vectorstore, k=5):
        self.vectorstore = vectorstore
        self.k = k

    def retrieve(self, query: str) -> pd.DataFrame:
        """FAISS 검색 실행 (LangChain 제거)"""
        doc_list = []
        doc_scores_list = []

        search_results = self.vectorstore.similarity_search_with_score(query, k=self.k)

        for document, score in search_results:
            tmp = ''
            page_content = document.page_content.strip()
            metadata = document.metadata  
            metadata_source = metadata['source'].strip()
            tmp += metadata_source+'\n' 
        
            if metadata['type'] == 'table':
                tmp +=  metadata['before_table_text'].strip()+'\n'
            tmp += '내용: ' + page_content
            
            doc_scores_list.append(score)
            doc_list.append(tmp)

        result_vectors = pd.DataFrame({
            'type': 'vectors',
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