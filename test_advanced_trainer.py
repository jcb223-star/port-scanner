import optuna
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer

from advanced_trainer import AdvancedTrainer

# Keep Optuna's per-trial logging out of test output.
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURE_COLS = ["f1", "f2", "f3", "f4"]


@pytest.fixture
def small_dataset():
    # Small, fast synthetic dataset — just enough for a 3-fold CV split
    # during hyperparameter search plus a held-out test split.
    X_raw, y_raw = make_classification(
        n_samples=60, n_features=4, n_informative=3, n_redundant=0,
        n_classes=2, random_state=42,
    )
    df = pd.DataFrame(X_raw, columns=FEATURE_COLS)
    return df, y_raw


@pytest.fixture
def preprocessor():
    return ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), FEATURE_COLS),
    ])


class TestAdvancedTrainer:
    def test_train_returns_fitted_pipeline_and_metrics(self, small_dataset, preprocessor):
        X, y = small_dataset
        trainer = AdvancedTrainer(random_state=42)

        model = trainer.train(X, y, preprocessor, n_trials=2, test_size=0.25)

        assert model is trainer.best_model
        assert "n_estimators" in trainer.best_params
        assert 0.0 <= trainer.test_metrics["roc_auc"] <= 1.0
        assert "precision" in trainer.test_metrics["classification_report"]

        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_save_artifact_without_training_raises(self):
        trainer = AdvancedTrainer()
        with pytest.raises(ValueError):
            trainer.save_artifact("unused.pkl")

    def test_save_artifact_writes_file(self, small_dataset, preprocessor, tmp_path):
        X, y = small_dataset
        trainer = AdvancedTrainer(random_state=42)
        trainer.train(X, y, preprocessor, n_trials=2, test_size=0.25)

        out_path = tmp_path / "artifacts" / "model_pipeline.pkl"
        trainer.save_artifact(str(out_path))

        assert out_path.exists()
