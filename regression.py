# regression.py

from utils import load_data, preprocess_data
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import os
import joblib  # for saving models


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def main():
    # Create directory to save models
    save_dir = "saved_models"
    os.makedirs(save_dir, exist_ok=True)

    # Load and preprocess data
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Model definitions with hyperparameter grids
    models = {
        "Ridge Regression": {
            "model": Ridge(),
            "params": {
                "alpha": [0.01, 0.1, 1.0, 10.0],
                "solver": ['auto', 'svd', 'cholesky'],
                "fit_intercept": [True, False]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(random_state=42),
            "params": {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        },
        "Random Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        }
    }

    # Run GridSearch for each model
    for name, mp in models.items():
        print(f"\n{name}")
        grid = GridSearchCV(mp["model"], mp["params"],
                            scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        mse, r2 = evaluate_model(best_model, X_test, y_test)

        print(f"Best Parameters: {grid.best_params_}")
        print(f"Test MSE = {mse:.4f}, R² = {r2:.4f}")

        # Save the best model
        model_filename = f"{save_dir}/{name.replace(' ', '_')}.joblib"
        joblib.dump(best_model, model_filename)
        print(f"Saved model to {model_filename}")


if __name__ == "__main__":
    main()
