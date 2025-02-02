from pydantic import BaseModel
from typing import Optional

class MongoSaveRequest(BaseModel):
    collection: str
    document: dict

class MongoQueryRequest(BaseModel):
    collection: str
    query: Optional[dict] = {}
