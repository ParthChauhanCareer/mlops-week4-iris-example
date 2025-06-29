import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    df = pd.read_csv("iris.csv")
    X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = df['species']

    model = joblib.load("model.joblib")
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    assert acc > 0.8, f"Model accuracy too low: {acc}"
