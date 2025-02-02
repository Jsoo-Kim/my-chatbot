from pydantic import BaseModel
from typing import List, Optional

class Message(BaseModel):
    role: str
    content: str

class ChatReqeust(BaseModel):
    session_id: str = "test"
    # messages: List[Message]
    question: str

class ChatResponse(BaseModel):
    response: str
