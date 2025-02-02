import asyncio
from langchain_core.messages import BaseMessage
# from pydantic_models.chat_models import ChatReqeust, ChatResponse
from sapie.rag.chathistory.mongo.custom_mongo_chat import initialize_chat_history
from sapie.rag.chathistory.chathistory_utils import ChathistoryUtil
from sapie.rag.retriever.rag_service import RagService
from sapie.models.llm.load_llm import LLMLoader


ragService = RagService()

llm_loader = LLMLoader(config_file="sapie/configs/config.json")
llm_instance = llm_loader.get_llm_instance()
# print(f"llm 모델: {llm_instance.model_path}")


class SapieService:
    def __init__(self):      
        config = {
            # "llm_model_path":"/home/jskim/data_js/test_241226/sapie/models/local_models/Qwen2.5-32B-Instruct-AWQ",
            "llm_model_path":"/home/jskim/data_js/vllm/code/models/Qwen2.5-14B-Instruct-AWQ",
            # "tokenizer_path":"/home/jskim/data_js/test_241226/sapie/models/local_models/Qwen2.5-32B-Instruct-AWQ",
            "tokenizer_path":"/home/jskim/data_js/vllm/code/models/Qwen2.5-14B-Instruct-AWQ",
        }
        chat_history_util = ChathistoryUtil(config)
        self.trimmer = chat_history_util.get_trimmer()


    def get_session_history(self, session_id:str):
        chat_history = initialize_chat_history(session_id=session_id, database_name="saltware", collection_name="chat_histories")

        # # 영구 DB
        # permanat_history = initialize_chat_history(session_id=session_id, database_name="saltware", collection_name="permanent_chat_histories")

        messages = chat_history.messages
        if messages:
            trimmed_messages = self.trimmer.invoke(messages)
            # 기존 세션 기록 삭제
            chat_history.clear()
            
            ## trim된 메시지들을 다시 저장
            for message in trimmed_messages:
                chat_history.add_message(message)

        return chat_history
    
    
    async def process_chat(self, session_id, query):
        try: 
            print(f"procees_chat에서 받은 session_id: {session_id} / query: {query}")
            # 1. 대화 히스토리 초기화 및 가져오기
            chat_history = self.get_session_history(session_id)
            print(f"넣어줄 챗히스토리: {chat_history}")

            # 2. 컨텍스트 검색
            context = ragService.get_context(query)
            print(f"컨텍스트 총 문자열 길이: {len(context)}")

            # 3. 프롬프트 생성
            messages = ragService.generate_chat_prompt_hybrid(query=query, context=context, chat_history=chat_history)
            print(f"프롬프트: {messages}")
            print(f"프롬프트 총 문자열 길이: {len(str(messages[0]))}")

            # 4. OpenAI API 호출
            full_response = ''
            for chunk in llm_instance.call_api(messages=messages):
                # print(f"sapie_service 쪽 청크: {chunk}")  # 각 청크 확인
                chunk_replaced = chunk.replace('\n', '🖐️')
                full_response += chunk
                yield f"data: {chunk_replaced}\n\n"  # SSE 포맷에 맞게 데이터 청크 전송 
                await asyncio.sleep(0.01)  # 0.01초 지연
                # asyncio.sleep이나 다른 비동기 호출로 이벤트 루프를 "양보"하지 않으면, 현재 작업이 이벤트 루프를 독점하게 됨 
                # 이 경우 다른 비동기 작업(예: 스트리밍 데이터 전송)이 지연될 수 있음
                # 위 코드에서 yield를 사용하면 청크를 스트리밍으로 반환하지만, 이벤트 루프가 다른 작업(예: 클라이언트로 데이터 전송)을 수행할 시간을 확보하지 못함
                # asyncio.sleep은 다음 작업을 실행할 기회를 주기 때문에 데이터가 클라이언트로 즉시 전송될 수 있음

            print("==============================")
            print(f"질문:  {query}")
            print("==============================")
            print(f"답변:  {full_response}")

            yield 'data: \u200c\n\n'

            # 6. 챗히스토리 저장
            chat_history.add_message(BaseMessage(type="human", role="user", content=query))
            chat_history.add_message(BaseMessage(type="ai", role="assistant", content=full_response))

            # return response_text

        except Exception as e:
            raise RuntimeError(f"Error processing chat request: {e}")
