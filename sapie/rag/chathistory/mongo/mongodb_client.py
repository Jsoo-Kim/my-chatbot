import os
from dotenv import load_dotenv
from pymongo import MongoClient, errors

class MongoDBClient:
    def __init__(self):
        load_dotenv()
        server = os.getenv("MONGODB_SERVER")
        if not server:
            raise ValueError("환경 변수 MONGODB_SERVER가 설정되지 않았습니다.")
            
        print(f"Connection Mongo DB Server: {server}")
        # currentEnv = os.getenv('FAST_ENV')
        # if currentEnv == 'production':
        #     self.connectionString = "mongodb://dba:20240925@localhost:11084/"
        # else:
        #     self.connectionString = "mongodb://localhost:27017/"
        
        ##추후 설정 필요
        self.connection_string = f"mongodb://{server}/"
        self.client = None

        # MongoDB 클라이언트 초기화
        try:
            self.client = MongoClient(self.connection_string)
            # 서버 연결 확인 (ping 테스트)
            self.client.admin.command('ping')
            print("MongoDB 연결 성공")
        except errors.ConnectionFailure as error:
            print(f"MongoDB 연결 실패: {error}")
            raise error  # 초기화 실패 시 예외 발생

    
    def get_database(self, db_name: str):
        """특정 데이터베이스 반환"""
        if not self.client:
            raise errors.ConnectionFailure("MongoDB 클라이언트가 초기화되지 않았습니다.")
        try:
            return self.client[db_name]
        except errors.ConnectionFailure as error:
            print(f"MongoDB 데이터베이스 연결 실패: {error}")
            raise error  # 예외를 다시 발생시켜 호출자가 처리하도록 함
        

    def close(self):
        """MongoDB 클라이언트를 닫는 메서드"""
        if self.client:
            self.client.close()
            print("MongoDB 클라이언트 연결이 닫혔습니다.")
        else:
            print("MongoDB 클라이언트가 이미 닫혀 있습니다.")

# 인스턴스 생성        
db_client = MongoDBClient()

    