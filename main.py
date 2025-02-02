from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sapie.api.sapie import router as sapie_router

app = FastAPI()

# CORS 구성
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def home():
    # logger.info("Accessed home endpoint.")
    return {"message": "Welcome to the API"}

app.include_router(sapie_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5406, log_level="debug")