# 📧 AI-Powered Spam Email Classifier

A production-style machine learning system that classifies SMS/email messages as **spam** or **ham (legitimate)**, complete with a REST API, an interactive web UI, and Docker support.

Built as an end-to-end ML engineering project — from raw text data to a deployed, containerized application.

---

## 🎯 What This Project Does

- Cleans and preprocesses raw SMS text using NLP techniques (tokenization, stopword removal, stemming)
- Converts text into numerical features using **TF-IDF**
- Trains and compares **4 machine learning models**: Logistic Regression, Naive Bayes, Random Forest, and SVM
- Automatically selects the best-performing model based on **F1-score**
- Serves predictions through a **FastAPI** backend
- Provides a clean, interactive **Streamlit** web interface
- Fully containerized with **Docker** for one-command deployment

---

## 📊 Model Performance

Trained and evaluated on the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (5,572 messages).

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Naive Bayes | 97.22% | 99.17% | 79.87% | 88.48% |
| Logistic Regression | 96.50% | 98.25% | 75.17% | 85.17% |
| Random Forest | 97.76% | 99.21% | 83.89% | 90.91% |
| **SVM (selected)** | **98.03%** | **100.00%** | **85.23%** | **92.03%** |

**SVM was automatically selected** as the best model based on F1-score — the fairest metric given the dataset's class imbalance (4,825 ham vs. 747 spam messages). Notably, SVM achieved **zero false positives** on the test set — it never misclassified a legitimate message as spam.

---

## 🏗️ Architecture

Raw Text → Preprocessing (NLTK) → TF-IDF Vectorization → SVM Model → Prediction
↓
FastAPI Backend (/predict)
↓
Streamlit Frontend (UI)

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **NLP:** NLTK (tokenization, stopword removal, stemming)
- **Machine Learning:** Scikit-learn (TF-IDF, Logistic Regression, Naive Bayes, Random Forest, SVM)
- **Model Persistence:** Joblib
- **Backend API:** FastAPI, Pydantic, Uvicorn
- **Frontend:** Streamlit
- **Testing:** Pytest
- **Deployment:** Docker, Docker Compose

---

## 📁 Project Structure

spam-classifier/
├── api/ # FastAPI backend
│ ├── main.py
│ └── schemas.py
├── app/ # Streamlit frontend
│ ├── streamlit_app.py
│ └── Dockerfile
├── src/ # Core ML pipeline
│ ├── config.py
│ ├── data_loader.py
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── train.py
│ ├── evaluate.py
│ ├── predict.py
│ └── logger.py
├── tests/ # Unit tests
├── data/ # Raw and processed datasets
├── models/ # Saved model + vectorizer (.pkl)
├── Dockerfile # FastAPI container
├── docker-compose.yml # Orchestrates both containers
└── requirements.txt

---

## 🚀 Running the Project

### Option 1: Docker (recommended)

```bash
docker compose up --build
```

- API available at: `http://localhost:8000/docs`
- Web app available at: `http://localhost:8501`

### Option 2: Run locally

```bash
pip install -r requirements.txt

# Train the model (only needed once)
python -m src.train

# Start the API (in one terminal)
uvicorn api.main:app --reload

# Start the frontend (in a second terminal)
streamlit run app/streamlit_app.py
```

### Running tests

```bash
python -m pytest tests\
```

---

## 🧠 Key Design Decisions

- **F1-score over accuracy** for model selection — the dataset is imbalanced (87% ham), so accuracy alone would be misleading.
- **TF-IDF over simple word counts** — captures word importance relative to the full dataset, not just raw frequency.
- **Separate model training from serving** — the model is trained once and saved with Joblib; the API loads it once at startup rather than retraining per request.
- **Modular architecture** — each pipeline stage (loading, preprocessing, features, training, evaluation) lives in its own file with a single responsibility.

---

## ⚠️ Known Limitations

The model was trained on a dataset of primarily UK/US-style SMS spam from the early 2000s. Testing shows it performs excellently on messages containing classic spam vocabulary ("free", "prize", "claim", "urgent"), but can be less confident on short messages or messages using different regional phrasing that weren't well-represented in training data. This is a common limitation of models trained on a single, fixed dataset — highlighting the importance of diverse, representative training data in real-world deployment.

---

## 📈 Possible Future Improvements

- Expand training data with more diverse, recent spam examples
- Add lemmatization as an alternative to stemming
- Experiment with word embeddings (Word2Vec, BERT) instead of TF-IDF
- Add model monitoring/retraining pipeline for production use

---

## 👤 Author

Built as a hands-on learning project covering the full ML engineering lifecycle: data preprocessing, model training/evaluation, API development, frontend development, and containerized deployment."# spam-email-classifier" 
