import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import os
import gdown

app = FastAPI(
    title="Lung Disease Classification API",
    description="EfficientNetB0-based model to classify lung X-ray images into: Lung Opacity, Normal, or Viral Pneumonia",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class labels — must match training order
CLASS_LABELS = ["Lung_Opacity", "Normal", "Viral Pneumonia"]

IMG_SIZE = (224, 224)

GDRIVE_FILE_ID = "1eB0uQYEOH1tUXZDy_LL7OHqD9gCCODJl"
MODEL_PATH = "/tmp/pneumonia_classifier.keras"

model = None

@app.on_event("startup")
async def load_model():
    global model

    # Download from Google Drive if not already cached
    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Download complete.")
    else:
        print("✅ Model already cached, skipping download.")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image to match training pipeline:
    - Resize to 224x224
    - Convert to RGB
    - NO /255 normalization (scalar function was used in training)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(image)  # shape: (224, 224, 3)
    img_array = tf.expand_dims(img_array, axis=0)                  # shape: (1, 224, 224, 3)
    return img_array


@app.get("/")
async def root():
    return {
        "message": "Lung Disease Classification API is running",
        "classes": CLASS_LABELS,
        "model": "EfficientNetB0"
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()

    try:
        img_array = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

    # Run inference
    predictions = model.predict(img_array)  # shape: (1, 3)
    confidence_scores = predictions[0].tolist()

    predicted_index = int(np.argmax(confidence_scores))
    predicted_class = CLASS_LABELS[predicted_index]
    confidence = float(confidence_scores[predicted_index])

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence * 100, 2),
        "all_scores": {
            CLASS_LABELS[i]: round(float(confidence_scores[i]) * 100, 2)
            for i in range(len(CLASS_LABELS))
        }
    }
