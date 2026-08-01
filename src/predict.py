import joblib
from src.config import BEST_MODEL_FILE, VECTORIZER_FILE
from src.preprocessing import preprocess_text
from src.logger import get_logger

logger = get_logger(__name__)


def load_model_and_vectorizer():
    """
    Loads the saved model and TF-IDF vectorizer from disk using Joblib.
    This is fast (no training involved) — just reading the saved files back into memory.
    """
    model = joblib.load(BEST_MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    logger.info("Loaded model and vectorizer successfully.")
    return model, vectorizer


def predict_message(text: str, model, vectorizer) -> dict:
    """
    Takes a raw message, runs it through the same preprocessing + vectorization
    pipeline used during training, and returns the prediction with a confidence score.
    """
    cleaned = preprocess_text(text)
    features = vectorizer.transform([cleaned])  # note: wrapped in a list — explained below

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    label = "spam" if prediction == 1 else "ham"
    confidence = probabilities[prediction]

    result = {
        "original_text": text,
        "cleaned_text": cleaned,
        "prediction": label,
        "confidence": round(float(confidence), 4),
    }

    logger.info(f"Prediction: {label} (confidence: {result['confidence']}) for text: '{text[:50]}...'")

    return result


if __name__ == "__main__":
    model, vectorizer = load_model_and_vectorizer()

    test_messages = [
        "You've won a free iPhone! Click here to claim your prize now!!!",
        "Hey, are we still meeting for lunch tomorrow?",
    ]

    for msg in test_messages:
        result = predict_message(msg, model, vectorizer)
        print(f"\nMessage   : {result['original_text']}")
        print(f"Prediction: {result['prediction']} (confidence: {result['confidence']})")