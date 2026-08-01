from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.predict import load_model_and_vectorizer, predict_message
from api.schemas import PredictionRequest, PredictionResponse
from src.logger import get_logger

logger = get_logger(__name__)

# A dictionary to hold our loaded model + vectorizer in memory,
# shared across all incoming requests once the server starts.
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server STARTS UP (before any requests arrive),
    and once when the server SHUTS DOWN.
    We use this to load our model + vectorizer a single time,
    instead of reloading them on every single request (which would be slow).
    """
    logger.info("Starting up API - loading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer()
    ml_models["model"] = model
    ml_models["vectorizer"] = vectorizer
    logger.info("Model and vectorizer loaded. API is ready.")

    yield  # server runs and handles requests here

    logger.info("Shutting down API...")
    ml_models.clear()


app = FastAPI(
    title="Spam Email Classifier API",
    description="An API for classifying SMS/email messages as spam or ham.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def read_root():
    """
    A simple health-check endpoint - lets us confirm the API is running.
    """
    return {"status": "ok", "message": "Spam Classifier API is running."}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Accepts a message and returns a spam/ham prediction with confidence score.
    """
    try:
        result = predict_message(
            request.text,
            ml_models["model"],
            ml_models["vectorizer"],
        )
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.")