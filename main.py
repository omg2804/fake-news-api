from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from pymongo import MongoClient

# ✅ Public, lightweight zero-shot model
model_name = "valhalla/distilbart-mnli-12-1"

classifier = pipeline("zero-shot-classification", model=model_name, framework="pt")

# MongoDB setup
client = MongoClient("mongodb+srv://gaikwadom992:xqSKA1ztPUdljf8R@cluster0.nzncblj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.newsDB
collection = db.predictions

app = FastAPI()  # enables default docs at /docs



class NewsInput(BaseModel):
    text: str

@app.post("/predict")
def predict(news: NewsInput):
    try:
        candidate_labels = ["fake news", "real news"]
        result = classifier(news.text, candidate_labels)

        scores = result['scores']
        labels = result['labels']

        top_label = labels[0]
        top_score = round(scores[0] * 100, 2)
        numeric_label = 0 if top_label == "fake news" else 1

        collection.insert_one({
            "text": news.text,
            "raw_label": top_label,
            "numeric_label": numeric_label,
            "confidence": top_score
        })

        return {
            "label": numeric_label,
            "raw_label": top_label,
            "confidence": f"{top_score} %"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
