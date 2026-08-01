from pathlib import Path

# BASE_DIR points to the project root folder (spam-classifier/)
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data paths ---
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_FILE = DATA_RAW_DIR / "spam.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "processed_data.csv"

# --- Model paths ---
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
VECTORIZER_FILE = MODELS_DIR / "vectorizer.pkl"

# --- Logging ---
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "app.log"

# --- ML settings ---
RANDOM_SEED = 42
TEST_SIZE = 0.2