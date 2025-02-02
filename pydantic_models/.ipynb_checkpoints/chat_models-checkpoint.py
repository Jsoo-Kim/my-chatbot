from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatReqeust(BaseModel):
    model: str = "/home/jskim/data_js/test_241226/sapie/models/local_models/Qwen2.5-32B-Instruct-AWQ"  # 기본 모델 경로
    session_id: str = "test"
    messages: List[Message]
    streaming: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
