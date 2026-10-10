from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

data_path=Path("data/all_tickets_processed_improved_v3.csv")
df=pd.read_csv(data_path)


X = df["Document"]
y = df["Topic_group"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,    
    stratify=y,
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

def create_pipline(classifier):
    return Pipeline([
        ("tfidf",
         TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,

         ),
         ),
         ("classifier",classifier),
    ])

models={
    "Logistic Regression": create_pipline(
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,

        )
    ),
    "Linear_SVM": create_pipline(
        LinearSVC(
            class_weight="balanced",
            random_state=42,
        )
    ),
}

results = {}
trained_models = {}

for name,model in models.items():
    print("-" * 30)
    print(f"Training {name}")
    print("-" * 30)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics={
        "Accuracy":accuracy_score(y_test,y_pred),
        "Precision":precision_score(
            y_test,y_pred,
            average="macro",
        ),
        "Recall":recall_score(
            y_test,y_pred,
            average="macro",
        ),
        "F1-Score":f1_score(
            y_test,y_pred,
            average="macro",
                    
        ),
    }
    results[name]=metrics
    trained_models[name]=model

    for metric,score in metrics.items():
        print(f"{metric}: {score:.4f}")


results_df = pd.DataFrame(results).T
results_df.index.name = "Model"

print("\nMODEL COMPARISON")
print(results_df.round(4).to_string())

best_model_name = results_df["F1-Score"].idxmax()
best_model = trained_models[best_model_name]

print(f"\nBest model: {best_model_name}")
print(
    "Best macro F1-score:",
    round(results_df.loc[best_model_name, "F1-Score"], 4),
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    best_model,
    MODEL_DIR / "ticket_classifier.joblib",
)
print("\nSaved successfully:")
print(MODEL_DIR / "ticket_classifier.joblib")