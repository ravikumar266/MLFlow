print("Script started")

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load data
wine = load_wine()
X = wine.data
y = wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Hyperparameters
max_depth = 10
n_estimators = 10


mlflow.set_experiment("mlopspart2")
# Start MLflow run
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    # Log to MLflow
    mlflow.log_metric('accuracy', score)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)
    
    cm= confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('predected')
    plt.title('confussion matrix')
    plt.savefig('confusion-matrix.png')
    mlflow.log_artifact('confusion-matrix.png') 
    # Output
    print(f"Accuracy: {score}")
