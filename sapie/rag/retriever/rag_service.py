from sapie.rag.retriever.hybrid_search_chain import HybridRetrieverChain
# from sapie.rag.retriever.hybrid_search_no_chain import HybridRetrieverChain


class RagService:
    def __init__(self):
        hybrid_retriever_chain_config = {
            "embedding_model_path": "dragonkue/BGE-m3-ko",
            # "tokenizer_path":"/home/jskim/data_js/test_241226/sapie/models/local_models/Qwen2.5-32B-Instruct-AWQ",
            "tokenizer_path":"/home/jskim/data_js/vllm/code/models/Qwen2.5-14B-Instruct-AWQ",
            "faiss_path": "/home/jskim/data_js/test_241226/sapie/data/vector_db/faiss/text_1024_table_8192_deidentified", 
            "jsonl_path": "/home/jskim/data_js/test_241226/sapie/data/vector_db/jsonl/text_1024_table_8192_deidentified_2.jsonl"
        }
        hybrid_chain = HybridRetrieverChain(hybrid_retriever_chain_config)
        self.retriever = hybrid_chain.create_retrieval_chain()

        # self.nochain_retriever = HybridRetrieverChain(hybrid_retriever_chain_config)
    
    
    def get_context(self, query):
        """
        사용자의 쿼리를 기반으로 FAISS에서 문서 컨텍스트를 검색합니다.
    
        Args:
            query (str): 사용자의 입력 쿼리
    
        Returns:
            str: 검색된 문서 컨텍스트
        """
        try:
            result = self.retriever.invoke({"query": query})
            print(f"get_context에서 반환된 컨텍스트 길이: {len(result)}")
            return result
            # return self.retriever
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""

    # def get_context(self, query):
    #     """
    #     사용자의 쿼리를 기반으로 FAISS에서 문서 컨텍스트를 검색합니다.
    
    #     Args:
    #         query (str): 사용자의 입력 쿼리
    
    #     Returns:
    #         str: 검색된 문서 컨텍스트
    #     """
    #     try:
    #         result = self.nochain_retriever.retrieve_hybrid_results(query=query)
    #         print(f"get_context에서 반환된 컨텍스트 길이: {len(result)}")
    #         return result
    #         # return self.retriever
    #     except Exception as e:
    #         print(f"Error retrieving context: {e}")
    #         return ""
        
    
    def generate_chat_prompt_hybrid(self, query: str, context: str, chat_history: str):
        """
        하이브리드 프롬프트 생성 함수
    
        Args:
            query (str): 사용자의 쿼리
            context (str): 검색된 컨텍스트
            chat_history (str): 대화 기록
    
        Returns:
            str: 완성된 프롬프트
        """
        prompt = """
        당신은 솔트웨어㈜에서 제공하는 기업 맞춤형 인공지능 챗봇 솔루션 Sapie입니다.
        다음은 솔트웨어 사내 업무와 관련된 직원의 문의입니다.
        Generate the final answer for the given passage. 
        
        질문: {query}

        Chat history:
        {chat_history}
    
        검색된 정보 (HTML 데이터 포함):
        {context}
        
        아래의 지침을 따라 응답하세요:
        1. 검색된 정보와 HTML 데이터를 분석하여 유의미한 정보를 식별하세요.
        2. 표가 있다면 표 근처에 요약된 정보를 바탕으로 내부적으로 추론하여 질문에 대한 최종 답변을 도출하세요.
        3. HTML 태그는 사용하지 말고, 깔끔한 텍스트 형식으로 출력하세요.
        4. 필요한 경우, 데이터의 불완전한 부분은 "알 수 없음" 또는 "정보 부족"으로 표시하여 직원에게 명확히 전달하세요.
        5. 직급 또는 직책에 관한 정보가 주어질 때 참고해서 작성하세요.
        6. 중국어는 사용하지 말고 최대한 한국어로 대답해주세요.
        """
    
        # 플레이스홀더에 데이터를 채움
        filled_prompt = prompt.format(
            query=query,
            context=context,
            chat_history=chat_history
        )
    
        return [{"role": "user", "content": filled_prompt}]