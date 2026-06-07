import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image, UnidentifiedImageError
import io
import os
import gdown

CLASS_LABELS = ["Lung_Opacity", "Normal", "Viral Pneumonia"]
IMG_SIZE = (224, 224)

GDRIVE_FILE_ID = "1eB0uQYEOH1tUXZDy_LL7OHqD9gCCODJl"
MODEL_PATH = "/tmp/pneumonia_classifier.keras"

MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024  # 10 MB

model = None


def is_valid_keras_file(path: str) -> bool:
    """Check if file is a real Keras/ZIP archive, not an HTML error page."""
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        # .keras files are ZIP archives — always start with PK\x03\x04
        return header == b"PK\x03\x04"
    except Exception:
        return False


def download_model():
    """Download model from Google Drive with full corruption protection."""
    tmp_path = MODEL_PATH + ".downloading"

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print("⬇️ Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    output = gdown.download(url, tmp_path, quiet=False, fuzzy=True)

    if output is None or not os.path.exists(tmp_path):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(
            "❌ Download failed. Check file permissions (must be 'Anyone with the link')."
        )

    if not is_valid_keras_file(tmp_path):
        os.remove(tmp_path)
        raise RuntimeError(
            "❌ Downloaded file is not a valid Keras model. "
            "Google Drive may have returned an error page. "
            "Check the file ID and sharing permissions."
        )

    # Atomic rename: MODEL_PATH only exists if download was fully successful
    os.rename(tmp_path, MODEL_PATH)
    print("✅ Download complete.")


def load_model_safe() -> tf.keras.Model:
    """
    Load model from disk.
    - Downloads first if not cached.
    - If loading fails (corrupt cache), deletes and re-downloads once, then retries.
    """
    if not os.path.exists(MODEL_PATH):
        download_model()
    else:
        print("✅ Model file found in cache.")

    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"⚠️ Cached model failed to load ({e}). Re-downloading...")
        try:
            os.remove(MODEL_PATH)
            download_model()
            return tf.keras.models.load_model(MODEL_PATH, compile=False)
        except Exception as retry_e:
            raise RuntimeError(
                f"❌ Failed to load model even after re-download: {retry_e}"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model_safe()
    print("✅ Model loaded successfully.")
    yield


app = FastAPI(
    title="Lung Disease Classification API",
    description="EfficientNetB0-based model to classify lung X-ray images into: Lung Opacity, Normal, or Viral Pneumonia",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image to match training pipeline:
    - Resize to 224x224 using LANCZOS (best quality for downscaling)
    - Convert to RGB
    - NO /255 normalization (scalar function was used in training)
    """
    if len(image_bytes) == 0:
        raise ValueError("Received an empty file. Please upload a valid image.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError("File is not a recognizable image format.")
    except Exception as e:
        raise ValueError(f"Could not open image: {str(e)}")

    image = image.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.array(image, dtype=np.float32)  # shape: (224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
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
    # Enforce max upload size before reading full content into memory
    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Maximum allowed size is {MAX_IMAGE_SIZE_MB} MB."
        )

    try:
        img_array = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Try again in a moment.")

    try:
        predictions = model.predict(img_array, verbose=0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

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
