import os
import joblib
import re
import numpy as np
import pandas as pd

from tempfile import mkdtemp
from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score
from config import MODEL_CACHE, MODEL_DIR, INSIGNIFICANT_WORDS
from app_logging import writelog

os.makedirs(MODEL_DIR, exist_ok=True)

cachedir = mkdtemp()
memory = Memory(location=cachedir, verbose=0)

x1n = 'ВидОперации'
x2n = 'НазначениеПлатежа'
yn = 'СтатьяДвиженияДенежныхСредств'

def clean_text(text):
    text = text.lower()
    for word in INSIGNIFICANT_WORDS:
        text = re.sub(r'\b' + word + r'\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d', '0', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def train_model(data, model_id):
    data = data[[x1n, x2n, yn]].dropna()
    data[x2n] = data[x2n].apply(clean_text)
    
    x = data[[x1n, x2n]]
    y = data[yn]
    column_transformer = ColumnTransformer([
        ('x1_tfidf', TfidfVectorizer(), x1n),
        ('x2_tfidf', TfidfVectorizer(), x2n)
    ])
   
    pipeline_rf = Pipeline([
        ('transformer', column_transformer),
        ('clf', RandomForestClassifier(
            min_samples_leaf=2, max_features="log2", class_weight="balanced_subsample", random_state=42))
    ], memory=memory)

    pipeline = pipeline_rf

    pipeline.fit(x, y)

    # Remove the old model from cache if it exists
    if model_id in MODEL_CACHE:
        del MODEL_CACHE[model_id]
        writelog(f"Model {model_id} removed from cache.")

    model_path = os.path.join(MODEL_DIR, f'model_{model_id}.joblib')
    joblib.dump(pipeline, model_path)
    writelog(f"Model saved to {model_path}")

def test_model(data, model_id):
    data = data[[x1n, x2n, yn]].dropna()
    data[x2n] = data[x2n].apply(clean_text)
    x = data[[x1n, x2n]]
    y = data[yn]
    model = load_model(model_id)
    predictions = model.predict(x)
    acc = accuracy_score(y, predictions)
    macro_f1 = f1_score(y, predictions, average='macro')
    weighted_f1 = f1_score(y, predictions, average='weighted')
    report = classification_report(y, predictions, zero_division=0)
    result = (
        f"Accuracy:           {acc:.5f}\n"
        f"Macro F1-score:     {macro_f1:.5f}\n"
        f"Weighted F1-score:  {weighted_f1:.5f}\n\n"
        f"Classification report:\n{report}\n"
    )
    return result

def predict_model(data, model_id):
    data[x2n] = data[x2n].apply(clean_text)
    model = load_model(model_id)
    prediction = model.predict(data)
    confidence = model.predict_proba(data).max(axis=1)
    predata = {
        "prediction": prediction[0],
        "confidence": confidence[0]
    }
    return predata

def load_model(model_id):
    if model_id in MODEL_CACHE:
        return MODEL_CACHE[model_id]
    else:
        model_path = os.path.join(MODEL_DIR, f'model_{model_id}.joblib')
        model = joblib.load(model_path)
        MODEL_CACHE[model_id] = model
        writelog(f"Model {model_id} loaded from file.")
        return model
