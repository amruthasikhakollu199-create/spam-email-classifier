from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

from src.feature_engineering import load_processed_data, create_tfidf_features, split_data
from src.evaluate import evaluate_all_models, select_best_model
from src.config import RANDOM_SEED, BEST_MODEL_FILE, VECTORIZER_FILE, MODELS_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def get_models() -> dict:
    """
    Returns a dictionary of model name -> model object.
    All 4 models are initialized here with sensible starting settings.
    """
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_SEED),
        "SVM": SVC(probability=True, random_state=RANDOM_SEED),
    }
    return models


def train_models(X_train, y_train) -> dict:
    """
    Trains every model in get_models() on the training data.
    Returns a dictionary of model name -> trained model object.
    """
    models = get_models()
    trained_models = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"Finished training {name}.")

    return trained_models


def save_best_model(best_model_result: dict, vectorizer):
    """
    Saves the winning model and the fitted TF-IDF vectorizer to disk
    using Joblib, so they can be reloaded instantly later without retraining.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model_result["model_object"], BEST_MODEL_FILE)
    logger.info(f"Saved best model ({best_model_result['model_name']}) to: {BEST_MODEL_FILE}")

    joblib.dump(vectorizer, VECTORIZER_FILE)
    logger.info(f"Saved TF-IDF vectorizer to: {VECTORIZER_FILE}")


def run_full_pipeline():
    """
    Runs the entire training pipeline end-to-end:
    load data -> create features -> split -> train all models ->
    evaluate all models -> select best -> save best model + vectorizer.
    """
    df = load_processed_data()
    features, vectorizer = create_tfidf_features(df["cleaned_text"])
    X_train, X_test, y_train, y_test = split_data(features, df["label"])

    trained_models = train_models(X_train, y_train)
    results = evaluate_all_models(trained_models, X_test, y_test)
    best_model_result = select_best_model(results)

    save_best_model(best_model_result, vectorizer)

    return best_model_result


if __name__ == "__main__":
    best = run_full_pipeline()
    print(f"\n✅ Pipeline complete. Best model: {best['model_name']} (F1-score: {best['f1_score']:.4f})")