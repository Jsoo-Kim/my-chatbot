from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from itertools import chain
from sapie.rag.chathistory.mongo.mongodb_client import db_client
from pydantic_models.chat_models import ChatReqeust, ChatResponse
from sapie.services.sapie_service import SapieService

router = APIRouter(
    prefix="/saltware",
    tags=["saltware"],
    responses={404: {"description": "Not found"}}
)

# headers = {
#     'Cache-Control': 'no-cache',
#     'Connection': 'keep-alive',
#     'Content-Type': 'text/event-stream'
# }

headers = {
    "Cache-Control": "no-cache",
    "Content-Type": "text/event-stream",
    "Connection": "keep-alive",
    "Transfer-Encoding": "chunked",
}

db = db_client.get_database('saltware')
historyCollection = db["chat_histories"]

# print(f"히스토리콜렉션: {historyCollection}")

sapieService = SapieService()


@router.post("/messages")
async def post_message(request: Request):
    data = await request.json()
    query = data.get('question')
    session_id = data.get('session_id')

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")
    
    print(f"Session ID: {session_id}, Query: {query}")

    # try:
    #     return await chatService.process_chat(session_id, query)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))
    try:
        return StreamingResponse(
            sapieService.process_chat(session_id=session_id, query=query),  # process_chat은 비동기 제너레이터가 됨
            media_type='text/event-stream',
            headers = headers
        )
    except Exception as e:
        print(f"Error while generating response: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    

@router.delete("/messages")
async def delete_message(request: Request):
    data = await request.json()
    session_id = data.get("session_id")

    try:
        chatHistory = historyCollection.find_one({"SessionId": session_id})
        historyCollection.delete_many({"SessionId":session_id})

        if chatHistory:
            message = "Session store and query list initialized"
        else:
            message = "Session not found"
            
        return JSONResponse(content={"message": message})
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error deleting session")


@router.post("/messageList")
async def post_message_list(request: Request):
    print("Request received at /messageList")
    data = await request.json()
    print(f"messageList data: {data}")
    session_id = data.get("session_id")

    try:
        chatHistories = historyCollection.find(
            {"SessionId": session_id},
            {"History": 1, "_id": 0}  # History 필드만 가져오고 _id 필드는 제외
        )
        print(f"챗히스토리 객체: {chatHistories}")
        messageList=[]
        all_histories = chain.from_iterable(chatHistory.get('History', []) for chatHistory in chatHistories)
        
        for history in all_histories:
            speaker = history['type']
            content = history['data']['content']
            # 디버깅 로그
            if not isinstance(content, str):
                print(f"🚨 비정상 데이터 감지! content: {content} (type: {type(content)})")
            if isinstance(content, list):
                content = " ".join(map(str, content))
            url_list = history.get('data', {}).get('response_metadata', {}).get('url_list', []) #없으면 []
            message = {
                "speaker": speaker,
                "content": content,
                "url_list": url_list
            }
            messageList.append(message)

        return JSONResponse(content={"messageList": messageList})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error fetching message list")
        



# @app.post("/mongo/save")
# async def save_to_mongo(request: MongoSaveRequest):
#     """
#     Save a document to a MongoDB collection.
#     """
#     try:
#         result = mongo_db[request.collection].insert_one(request.document)
#         return {"message": "Document saved", "id": str(result.inserted_id)}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error saving to MongoDB: {str(e)}")


# @app.post("/mongo/query")
# async def query_mongo(request: MongoQueryRequest):
#     """
#     Query documents from a MongoDB collection.
#     """
#     try:
#         results = list(mongo_db[request.collection].find(request.query))
#         for doc in results:
#             doc["_id"] = str(doc["_id"])  # Convert ObjectId to string for JSON serialization
#         return {"data": results}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error querying MongoDB: {str(e)}")



