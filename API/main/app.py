import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import logging
import cv2
import pickle
import uvicorn

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# Middleware to log requests
@app.middleware("http")
async def log_requests(request, call_next):
    logging.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logging.info(f"Response: {response.status_code}")
    return response

# Load the encoding and model using relative paths
base_path = os.path.dirname(os.path.dirname(__file__))  # Adjust base path if necessary
model_path = os.path.join(base_path, "Model", "VGG19_model.keras")


model = load_model(model_path)

# Home endpoint
@app.get("/")
async def home():
    return {
        "Project": "Gastritis Prediction",
        "Date": "Oct2024",
        "version": "v0.01"
    }

# Function to prepare the image for prediction
def image_preparation(image):
    img_array = np.fromstring(image.file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# Prediction endpoint
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Prepare the image
        img = image_preparation(image)
        # Predict
        prediction = model.predict(img)
        result = np.argmax(prediction)

        id2label = { 0:'Mild', 1:'Moderate', 2:'Severe'}
        return JSONResponse({"Prediction":id2label[result]})
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
