import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("iris.csv")
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(X_train, y_train)

joblib.dump(model, "model.joblib")

accuracy = accuracy_score(y_test, model.predict(X_test))
with open("metrics.csv", "w") as f:
    f.write(f"accuracy,{accuracy:.3f}")