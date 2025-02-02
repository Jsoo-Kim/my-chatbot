from pymongo import MongoClient, errors
from sapie.rag.chathistory.mongo.mongodb_client import db_client


class SimpleMongoDBChatHistory:
    def __init__(self, 
                 connection_string: str, 
                 session_id: str,
                 database_name: str, 
                 collection_name: str
                 ):
        try:
            self.session_id: session_id # type: ignore
            self.client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
            # 서버 연결 확인 (ping 테스트)
            self.client.admin.command('ping')
            print("MongoDB 연결 성공")

            self.db = self.client[database_name]
            self.collection = self.db[collection_name]

        except errors.ConnectionFailure as error:
            print(f"MongoDB 연결 실패: {error}")
            raise error  # 초기화 실패 시 예외 발생
        except Exception as error:
            print(f"MongoDB 초기화 중 알 수 없는 오류 발생: {error}")
            raise error

    def get_messages(self):
        """특정 세션의 메시지를 가져옵니다."""
        try:
            document = self.collection.find_one({"SessionId": self.session_id})
            if document:
                return document.get("History", [])
            return []
        except errors.OperationFailure as error:
            print(f"메시지 조회 실패: {error}")
            raise error  # 호출자가 처리하도록 예외 발생

    def add_message(self, message: dict):
        """새 메시지를 추가합니다."""
        try:
            result = self.collection.update_one(
                {"SessionId": self.session_id},
                {"$push": {"History": message}},
                upsert=True
            )
            if result.modified_count == 0 and result.upserted_id:
                print(f"새로운 세션 생성 및 메시지 추가 (SessionId: {self.session_id})")
        except errors.WriteError as error:
            print(f"메시지 추가 실패: {error}")
            raise error

    def clear_history(self):
        """특정 세션의 메시지를 삭제합니다."""
        try:
            result = self.collection.delete_one({"SessionId": self.session_id})
            if result.deleted_count == 0:
                print(f"삭제할 히스토리가 없습니다 (SessionId: {self.session_id})")
        except errors.WriteError as error:
            print(f"히스토리 삭제 실패: {error}")
            raise error

    @staticmethod
    def initialize_chat_history(session_id: str):
        """대화 히스토리를 초기화하고 기존 기록을 반환"""
        try:
            chat_history = SimpleMongoDBChatHistory(
                session_id=session_id,
                connection_string=db_client.connection_string,
                database_name="saltware",
                collection_name="chat_histories"
            )
            return chat_history, chat_history.get_messages(session_id)
        except Exception as error:
            print(f"대화 히스토리 초기화 중 오류 발생: {error}")
            raise error
