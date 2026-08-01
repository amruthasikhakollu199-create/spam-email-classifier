from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from src.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluates a single trained model on the test set.
    Returns a dictionary of metric name -> value.
    """
    predictions = model.predict(X_test)

    results = {
        "model_name": model_name,
        "model_object": model,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
    }

    logger.info(
        f"{model_name} | Accuracy: {results['accuracy']:.4f} | "
        f"Precision: {results['precision']:.4f} | "
        f"Recall: {results['recall']:.4f} | "
        f"F1: {results['f1_score']:.4f}"
    )

    return results


def evaluate_all_models(trained_models: dict, X_test, y_test) -> list:
    """
    Evaluates every model in trained_models.
    Returns a list of result dictionaries, one per model.
    """
    all_results = []
    for name, model in trained_models.items():
        result = evaluate_model(model, X_test, y_test, name)
        all_results.append(result)
    return all_results


def select_best_model(results: list) -> dict:
    """
    Picks the best model based on the highest F1-score.
    F1-score is used (rather than accuracy) because our dataset
    is imbalanced (far more ham than spam) - F1 fairly balances
    precision and recall instead of being skewed by the majority class.
    """
    best = max(results, key=lambda r: r["f1_score"])
    logger.info(f"Best model selected: {best['model_name']} (F1: {best['f1_score']:.4f})")
    return best


if __name__ == "__main__":
    from src.feature_engineering import load_processed_data, create_tfidf_features, split_data
    from src.train import train_models

    df = load_processed_data()
    features, vectorizer = create_tfidf_features(df["cleaned_text"])
    X_train, X_test, y_train, y_test = split_data(features, df["label"])

    trained_models = train_models(X_train, y_train)
    results = evaluate_all_models(trained_models, X_test, y_test)

    print("\n--- Evaluation Results ---")
    for r in results:
        print(f"\n{r['model_name']}")
        print(f"  Accuracy : {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall   : {r['recall']:.4f}")
        print(f"  F1-score : {r['f1_score']:.4f}")
        print(f"  Confusion Matrix:\n{r['confusion_matrix']}")

    best_model_result = select_best_model(results)
    print(f"\n🏆 Best Model: {best_model_result['model_name']} (F1-score: {best_model_result['f1_score']:.4f})")