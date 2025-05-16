from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# Load the fake news detection model
MODEL_NAME = "jy46604790/Fake-News-Bert-Detect"
classifier = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, framework="pt")


# MongoDB setup
MONGO_DETAILS = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.newsDB
collection = database.predictions

# FastAPI setup
app = FastAPI()

# Request schema
class NewsInput(BaseModel):
    text: str

# POST /predict endpoint
@app.post("/predict")
async def predict_news(news: NewsInput):
    result = classifier(news.text)[0]
    label = result['label']
    score = float(result['score'])

    # Save prediction to MongoDB
    await collection.insert_one({
        "text": news.text,
        "label": label,
        "score": score,
        "timestamp": datetime.utcnow()
    })

    return {"label": label, "confidence": round(score, 4)}

# GET /logs endpoint
@app.get("/logs")
async def get_logs():
    cursor = collection.find({}, {"_id": 0})
    logs = []
    async for doc in cursor:
        logs.append(doc)
    return {"logs": logs}
