import os
import joblib
import numpy as np
import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Import the FeatureEngineer from your feature engineering module
# from feature_engineering import FeatureEngineer

class AdvancedTrainer:
    """
    Handles advanced model training, hyperparameter tuning with Optuna,
    cross-validation, held-out evaluation, and pipeline serialization.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.test_metrics = None

    def _get_objective(self, X: pd.DataFrame, y: np.ndarray, preprocessor):
        """
        Creates an Optuna objective function for hyperparameter tuning.
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                # Parallelism lives at the cross_val_score level (one fold per
                # core); leaving XGBoost's own n_jobs at -1 here would have
                # each parallel fold also fan out across all cores, causing
                # CPU oversubscription and contention. Keep XGBoost
                # single-threaded per fold instead.
                'n_jobs': 1
            }

            # Combine feature engineering preprocessor and classifier into a single pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', XGBClassifier(**params))
            ])

            # Use Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

            # Score using ROC-AUC or Accuracy
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)

            return scores.mean()

        return objective

    def train(self, X: pd.DataFrame, y: np.ndarray, preprocessor, n_trials: int = 10, test_size: float = 0.2):
        """
        Splits off a held-out test set, runs Optuna hyperparameter
        optimization (via cross-validation on the training split only),
        fits the final pipeline on the training split, and evaluates it
        on the untouched test split.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        print("Starting hyperparameter optimization with Optuna...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self._get_objective(X_train, y_train, preprocessor), n_trials=n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        print(f"Best Hyperparameters found: {self.best_params}")
        print(f"Best Cross-Validation ROC-AUC: {study.best_value:.4f}")

        # Build final pipeline with best parameters
        self.best_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(**self.best_params, random_state=self.random_state, eval_metric='logloss', n_jobs=-1))
        ])

        print("Training final model on the training split...")
        self.best_model.fit(X_train, y_train)

        # Evaluate on the held-out test split for an unbiased performance estimate.
        # (study.best_value is a CV score on the training data, which was itself
        # the optimization target during tuning, so it tends to run optimistic.)
        y_pred = self.best_model.predict(X_test)
        y_proba = self.best_model.predict_proba(X_test)[:, 1]

        test_auc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred)
        self.test_metrics = {'roc_auc': test_auc, 'classification_report': report}

        print(f"\nHeld-out Test ROC-AUC: {test_auc:.4f}")
        print("Held-out Test Classification Report:")
        print(report)

        return self.best_model

    def save_artifact(self, filepath: str = "model_pipeline.pkl"):
        """
        Saves the trained end-to-end pipeline to disk.
        """
        if self.best_model is None:
            raise ValueError("No trained model found. Run 'train' first.")

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        joblib.dump(self.best_model, filepath)
        print(f"Model pipeline successfully saved to {filepath}")


# --- Example Execution ---
if __name__ == "__main__":
    from sklearn.datasets import make_classification

    # Generate synthetic binary classification dataset
    X_raw, y_raw = make_classification(
        n_samples=500, n_features=5, n_informative=3,
        n_classes=2, random_state=42
    )

    df = pd.DataFrame(X_raw, columns=['age', 'salary', 'score_a', 'score_b', 'tenure'])
    # Add a categorical column to test encoding (seeded for reproducibility)
    rng = np.random.default_rng(42)
    df['department'] = rng.choice(['IT', 'HR', 'Finance'], size=len(df))

    # Mock preprocessor (In production, import your custom FeatureEngineer class)
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    numeric_cols = ['age', 'salary', 'score_a', 'score_b', 'tenure']
    categorical_cols = ['department']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Initialize and execute training pipeline
    trainer = AdvancedTrainer(random_state=42)
    trained_pipeline = trainer.train(df, y_raw, preprocessor, n_trials=5)

    # Save the pipeline artifact
    trainer.save_artifact("artifacts/model_pipeline.pkl")
