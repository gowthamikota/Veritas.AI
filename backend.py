from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from typing import List

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure logging
logging.basicConfig(filename="app.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Load models and vectorizer
models = {
    "logistic": joblib.load("C:\\Users\\GOWTHAMI\\Desktop\\AI TEXT\\text_detection_model.pkl"),  # Optional, if using a Transformer model
}
vectorizer = joblib.load("vectorizer.pkl")

# Input schema
class TextInput(BaseModel):
    text: str
    model: str = "logistic"  # Default model

# Home Page
@app.get("/", response_class=HTMLResponse)
def read_root():
    try:
        with open("static/index.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except Exception as e:
        logging.error(f"Error loading home page: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Detector Page
@app.get("/detector", response_class=HTMLResponse)
def read_detector():
    try:
        with open("static/detector.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read(), status_code=200)
    except Exception as e:
        logging.error(f"Error loading detector page: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Predict API
@app.post("/predict")
def predict_text(data: TextInput):
    try:
        if data.model not in models:
            raise HTTPException(status_code=400, detail="Model not found")
        
        model = models[data.model]
        text_vectorized = vectorizer.transform([data.text])
        prediction = model.predict(text_vectorized)[0]

        confidence = "Unknown"
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vectorized)[0]
            confidence = max(probabilities) * 100

        result = "AI-generated" if prediction == 1 else "Human-written"
        
        # Log prediction
        logging.info(f"Prediction made | Model: {data.model} | Result: {result} | Confidence: {confidence}%")

        return {
            "prediction": result,
            "confidence": round(confidence, 2) if confidence != "Unknown" else "Unknown",
            "model": data.model
        }
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run Server: uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
