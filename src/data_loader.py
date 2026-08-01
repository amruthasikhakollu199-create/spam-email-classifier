import pandas as pd
from src.config import RAW_DATA_FILE
from src.logger import get_logger

logger = get_logger(__name__)


def load_raw_data() -> pd.DataFrame:
    """
    Loads the raw SMS Spam Collection dataset from data/raw/spam.csv,
    keeps only the relevant columns, and renames them for clarity.
    Returns a DataFrame with two columns: 'label' and 'text'.
    """
    try:
        df = pd.read_csv(RAW_DATA_FILE, encoding="latin-1")
    except FileNotFoundError:
        logger.error(f"Dataset not found at: {RAW_DATA_FILE}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading dataset: {e}")
        raise

    # The Kaggle CSV has extra junk columns (Unnamed: 2, 3, 4) — we ignore them
    df = df[["v1", "v2"]]
    df.columns = ["label", "text"]

    # Convert text labels into numbers: ham -> 0, spam -> 1
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    logger.info(f"Loaded dataset successfully. Shape: {df.shape}")
    logger.info(f"Class distribution:\n{df['label'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    data = load_raw_data()
    print(data.head())