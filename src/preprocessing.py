import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.config import PROCESSED_DATA_FILE, DATA_PROCESSED_DIR
from src.data_loader import load_raw_data
from src.logger import get_logger

logger = get_logger(__name__)

# Load these once at the top, so we don't reload/recreate them on every function call
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def to_lowercase(text: str) -> str:
    """
    Converts all characters in the given text to lowercase.
    Example: "WINNER!!" -> "winner!!"
    """
    return text.lower()


def remove_punctuation_and_numbers(text: str) -> str:
    """
    Removes all punctuation marks, special symbols, and digits,
    keeping only letters and spaces.
    Example: "winner!! ... £900 prize!" -> "winner  prize"
    """
    return re.sub(r"[^a-z\s]", "", text)


def tokenize(text: str) -> list:
    """
    Splits a cleaned sentence into a list of individual words (tokens).
    Example: "winner you have a prize" -> ["winner", "you", "have", "a", "prize"]
    """
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """
    Removes common filler words (like 'the', 'is', 'a') from a list of tokens,
    keeping only the words that carry meaningful information.
    Example: ["winner", "you", "have", "a", "prize"] -> ["winner", "prize"]
    """
    return [word for word in tokens if word not in STOPWORDS]


def stem_words(tokens: list) -> list:
    """
    Reduces each word to its root/base form by stripping common suffixes.
    Example: ["selected", "winning", "prizes"] -> ["select", "win", "prize"]
    """
    return [STEMMER.stem(word) for word in tokens]


def preprocess_text(text: str) -> str:
    """
    Runs the full cleaning pipeline on a single piece of text:
    lowercase -> remove punctuation/numbers -> tokenize ->
    remove stopwords -> stem -> join back into a string.

    Example:
        Input:  "WINNER!! You have been selected to receive a £900 prize reward!"
        Output: "winner select receiv prize reward"
    """
    text = to_lowercase(text)
    text = remove_punctuation_and_numbers(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_words(tokens)
    return " ".join(tokens)


def preprocess_dataset() -> pd.DataFrame:
    """
    Loads the raw dataset, applies preprocess_text() to every message,
    saves the cleaned result to data/processed/processed_data.csv,
    and returns the cleaned DataFrame.
    """
    df = load_raw_data()

    logger.info("Starting preprocessing on full dataset...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)
    logger.info("Finished preprocessing all messages.")

    # Make sure the processed folder exists before saving into it
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    logger.info(f"Saved processed dataset to: {PROCESSED_DATA_FILE}")

    return df


if __name__ == "__main__":
    df = preprocess_dataset()
    print(df[["text", "cleaned_text"]].head())