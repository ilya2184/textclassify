import os
import joblib
import re

from tempfile import mkdtemp
from joblib import Memory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
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
    text= re.sub(r'\d', '0', text)
    return text

def train_model(data, model_id):
    data = data[[x1n, x2n, yn]].dropna()
    
    data[x2n] = data[x2n].apply(clean_text)

    X = data[[x1n, x2n]]
    y = data[yn]
    column_transformer = ColumnTransformer([
        ('x1_tfidf', TfidfVectorizer(), x1n),
        ('x2_tfidf', TfidfVectorizer(), x2n)
    ])
    pipeline = Pipeline([
        ('transformer', column_transformer),
        ('clf', LogisticRegression(random_state=42))
    ], memory=memory)
    
    pipeline.fit(X, y)

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

    X = data[[x1n, x2n]]
    y = data[yn]
    model = load_model(model_id)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy

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
