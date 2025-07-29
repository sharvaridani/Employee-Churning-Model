import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)

# 1. Load Dataset
df = pd.read_csv("data/churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce').fillna(0)
df.drop(["customerID"], axis=1, inplace=True)

# 2. Encode Categorical Variables
label_encoders = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 3. Define Features and Target
features = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges',
    'TotalCharges', 'Contract', 'PaymentMethod', 'InternetService'
]
X = df[features]
y = df["Churn"]

# 4. Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Train Models
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

trained_models = {}
predictions = {}
probs = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    predictions[name] = model.predict(X_test)
    probs[name] = model.predict_proba(X_test)[:, 1]

# 7. Evaluation & Visualization
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Stayed", "Churned"],
                yticklabels=["Stayed", "Churned"])
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/plots/confusion_{title.lower().replace(' ', '_')}.png")
    plt.close()

def plot_roc_curve(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {title}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"outputs/plots/roc_{title.lower().replace(' ', '_')}.png")
    plt.close()

for name in models:
    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, predictions[name]))
    print(classification_report(y_test, predictions[name]))
    plot_confusion_matrix(y_test, predictions[name], name)
    plot_roc_curve(y_test, probs[name], name)

# 8. Compare Models (Bar Plot)
def compare_models(metrics_dict):
    metric_names = ["Precision", "Recall", "F1-Score", "Accuracy"]
    x = np.arange(len(metric_names))
    width = 0.25

    plt.figure(figsize=(10, 6))
    for i, (name, scores) in enumerate(metrics_dict.items()):
        plt.bar(x + i * width, scores, width, label=name)

    plt.xticks(x + width, metric_names)
    plt.ylabel("Score")
    plt.ylim(0, 1.1)
    plt.title("Model Comparison")
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig("outputs/plots/model_comparison.png")
    plt.close()

def extract_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    return [
        report["weighted avg"]["precision"],
        report["weighted avg"]["recall"],
        report["weighted avg"]["f1-score"],
        accuracy_score(y_true, y_pred)
    ]

metrics = {
    name: extract_metrics(y_test, predictions[name])
    for name in models
}

compare_models(metrics)



📦 Supporting Files
    data/churn.csv – your dataset.
    outputs/plots/ – directory where confusion matrices, ROC curves, and comparison charts are saved.
