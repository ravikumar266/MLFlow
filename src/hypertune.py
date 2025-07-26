from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import mlflow.sklearn

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model & hyperparameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Set experiment (optional, but recommended)
mlflow.set_experiment("breast_cancer_tracker")

# Main run
with mlflow.start_run() as parent_run:

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Log each grid search result as nested run
    for i in range(len(grid_search.cv_results_['params'])):
        with mlflow.start_run(nested=True):
            params = grid_search.cv_results_['params'][i]
            score = grid_search.cv_results_['mean_test_score'][i]
            mlflow.log_params(params)
            mlflow.log_metric("mean_accuracy", score)

    # Best model details
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log best params & score
    for key, val in best_params.items():
        mlflow.log_param(key, val)
    mlflow.log_metric("best_accuracy", best_score)

    # Save & log train
