import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_FILE, RANDOM_SEED, TEST_SIZE
from src.logger import get_logger

logger = get_logger(__name__)


def load_processed_data() -> pd.DataFrame:
    """
    Loads the cleaned dataset we saved in Stage 4.
    """
    df = pd.read_csv(PROCESSED_DATA_FILE)
    df["cleaned_text"] = df["cleaned_text"].fillna("")  # safety: avoid errors on empty rows
    logger.info(f"Loaded processed data. Shape: {df.shape}")
    return df


def create_tfidf_features(texts: pd.Series):
    """
    Converts a column of cleaned text into TF-IDF numeric features.
    Returns the transformed feature matrix AND the fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(max_features=3000)
    features = vectorizer.fit_transform(texts)

    logger.info(f"TF-IDF matrix shape: {features.shape}")
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    return features, vectorizer


def split_data(features, labels):
    """
    Splits features and labels into training and testing sets.
    80% train, 20% test (as defined in config.py).
    stratify=labels ensures both sets keep the same spam/ham ratio.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    logger.info(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_processed_data()
    features, vectorizer = create_tfidf_features(df["cleaned_text"])
    X_train, X_test, y_train, y_test = split_data(features, df["label"])

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)