import jsonlines
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
from transformers import AutoTokenizer
from operator import itemgetter
from sapie.models.embeddings.load_embeddings import EmbeddingLoader
from sapie.models.embeddings.embedding_initializers import get_huggingface_embeddings
from sapie.rag.retriever.faiss_retriever import get_faiss_db
from sapie.rag.retriever.custom_bm25_retriever import BM25Retriever
from sapie.rag.retriever.custom_faiss_retriever import FaissRetriever

# os.environ["TOKENIZERS_PARALLELISM"] = "false"
from kiwipiepy import Kiwi

class HybridRetrieverChain:
    def __init__(self, config):
        """
            config: {
                "embedding_model_path": "dragonkue/BGE-m3-ko",
                "faiss_path": "/home/jskim/data_js/sapie_project2024_backend_sglang/data/text_512_table_4096", 
                "tokenizer_path":"/home/jskim/data_js/vllm/code/models/Qwen2.5-32B-Instruct-AWQ",
                "jsonl_path": "/home/jskim/data_js/sapie_project2024_backend_sglang/data/text_512_table_4096_2.jsonl"
            }
        """

        ## 에러 처리 추후에 함수화 시킬예정
        required_fields = ["embedding_model_path", "faiss_path", "tokenizer_path", "jsonl_path"]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"(hybrid_search_chain.py) Missing required configuration fields: {', '.join(missing_fields)}")

        self.config = config
        
        self.bm25_config = {"k1":1.5, "b":0.75, "epsilon":0.5} ## 설정옵션 재윤님 코드참조
        self.bm25_search_k = 5 # bm25에서 찾을 갯수 현재는 사용안함 기존코드와 동일하게 하기 위해
        self.vectorstore_search_k = 5
        self.hybrid_search_k = 3 # 최종검색할 문서 갯수
        self.context_max_length = 8000 # 최종검색할 길이
        self.rrf_score = 60 ##재윤님 코드참조

        self.kiwi = Kiwi()# 형태소 분석기 라이브러리
        self.embedding_loader = EmbeddingLoader(config_file="sapie/configs/config.json")
    
    
    ## 형태소 형태로 자르기
    def kiwi_tokenize(self, text):
        return [token.form for token in self.kiwi.tokenize(text)]
    
    def read_jsonl(self, path):
        data = list()
        with jsonlines.open(path) as f:
            for line in f:
                data.append(line)
        return data

    ## rrf 알고리즘 
    def calculate_rrf_scores(self, input_dict):
        bm25_df = input_dict['bm25_results']
        bm25_df = bm25_df.sort_values(['score'], ascending=False).reset_index(drop=True)

        faiss_df = input_dict['faiss_results']
        faiss_df = faiss_df.sort_values('score', ascending=True).reset_index(drop=True)

        # RRF(Reciprocal Rank Fusion): 각 검색기의 순위(index)를 사용해 역순위 점수를 계산
        faiss_df['new_scores_RRF'] = 1 / ((faiss_df.index + 1) + self.rrf_score)
        bm25_df['new_scores_RRF'] = 1 / ((bm25_df.index + 5) + self.rrf_score)
        
        combined_df_RRF = pd.concat([bm25_df, faiss_df]).sort_values('new_scores_RRF', ascending=False).reset_index(drop=True)
        # print(combined_df_RRF)
        # combined_df_RRF.to_csv('/home/jskim/data_js/sapie_project2024_backend_sglang/combined_results.csv', index=False, encoding='utf-8')  # 한글 깨짐 방지       

        top_n_RFF = combined_df_RRF.drop_duplicates(subset='text').sort_values('new_scores_RRF', ascending=False).reset_index(drop=True)

        first_n = self.hybrid_search_k
        # until_len = 25000
        until_len = 5000
        
        # text_length 기준 head(n) 계산
        total_text_length = top_n_RFF.head(first_n)['text_length'].sum()
        print('총 문자열 길이:', total_text_length)
        
        # 결과 길이 제한 설정
        cumulative_sum = np.cumsum(top_n_RFF['text_length'])
        first_n = (cumulative_sum >= until_len).argmax()+1  # +1은 head()에 적합하도록 조정  
        if first_n==0 or  first_n==1:
            first_n=len(cumulative_sum)
        print('first_n', first_n)
        top_n_RFF = top_n_RFF.head(first_n)
        
        
        # top_n_RFF = combined_df_RRF.drop_duplicates(subset='text') \
        #     .sort_values(by='new_scores_RRF', ascending=False) \
        #     .head(first_n) \
        #     .reset_index(drop=True)
        print(top_n_RFF)
        print(top_n_RFF.text, len(top_n_RFF))

        # CSV 저장(확인용)
        # combined_df_RRF.to_csv('/data/data_sm/csv/jsm/jsm_combined_results.csv', index=False, encoding='utf-8')  # 한글 깨짐 방지       
        # top_n_RFF.to_csv('/home/jskim/data_js/sapie_project2024_backend_sglang/total_results.csv', index=False, encoding='utf-8')  # 한글 깨짐 방지       

        
        # 실제 사용된 인덱스만 따로 저장
        used_indices = []
        current_length  = 0
        result_str =''
        
        ## 임시로 max길이만큼 설정
        for i in range(len(top_n_RFF)):
            document = top_n_RFF.loc[i]['text']
            score = top_n_RFF.loc[i]['new_scores_RRF']
            
            document_str = f"{document} 관련성: {score}\n\n"
            print(f"현재길이는 {current_length} 추가할 길이는 {len(document_str)}")
            
            new_length = current_length + len(document_str)
            if new_length > until_len:
                break
            result_str += document_str
            current_length = new_length

            used_indices.append(i)  # 실제 사용된 인덱스 저장
    
        
        # print(f"총 Context의 길이 : {current_length}")     
        # 실제 사용된 결과만 CSV로 저장
        used_results = top_n_RFF.iloc[used_indices]

        try:
            save_path = './data/csv/sapie_used_results.csv'
            used_results.to_csv(save_path, index=False, encoding='utf-8')  # 한글 깨짐 방지        
        except Exception as e:
            print(f"{save_path}파일 저장 중 오류가 발생했습니다: {str(e)}")

        # used_results.to_csv('/home/jskim/data_js/sapie_project2024_backend_sglang/used_results.csv', index=False, encoding='utf-8')

        return result_str  


    def create_retrieval_chain(self):
        train_data = self.read_jsonl(self.config["jsonl_path"])
        corpus = [data['combined_metadata'] for data in train_data]
    
        bm25_retriever = BM25Retriever.from_texts(
            corpus,
            bm25_params=self.bm25_config,
            preprocess_func=self.kiwi_tokenize,
            k=self.bm25_search_k
        )
        
        # embeddings = get_huggingface_embeddings(model_name = self.config["embedding_model_path"])
        embeddings = self.embedding_loader.get_embedding_model()
        vectorstore = get_faiss_db(embeddings, self.config["faiss_path"])

        faiss_retriever = FaissRetriever.from_db(vectorstore, k=self.vectorstore_search_k)
             
        hybrid_chain = {
            "bm25_results": itemgetter("query") | bm25_retriever,
            "faiss_results": itemgetter("query") | faiss_retriever
        } | RunnableLambda(self.calculate_rrf_scores)
            
        return hybrid_chain
