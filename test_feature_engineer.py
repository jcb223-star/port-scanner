import numpy as np
import pandas as pd
import pytest

from feature_engineer import FeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age": [25, 32, np.nan, 47],
        "salary": [50000, 60000, 80000, np.nan],
        "department": ["HR", "IT", "Finance", np.nan],
    })


class TestFeatureEngineer:
    def test_fit_transform_shape(self, sample_df):
        fe = FeatureEngineer()
        transformed = fe.fit_transform(sample_df)
        assert transformed.shape[0] == len(sample_df)
        assert transformed.shape[1] == len(fe.get_feature_names_out())

    def test_numeric_missing_values_imputed(self, sample_df):
        fe = FeatureEngineer()
        transformed = fe.fit_transform(sample_df)
        assert not np.isnan(transformed).any()

    def test_categorical_columns_onehot_encoded(self, sample_df):
        fe = FeatureEngineer()
        fe.fit(sample_df)
        names = fe.get_feature_names_out()
        assert any("department" in name for name in names)

    def test_column_type_detection(self, sample_df):
        fe = FeatureEngineer()
        fe.fit(sample_df)
        assert fe.numeric_cols == ["age", "salary"]
        assert fe.categorical_cols == ["department"]

    def test_transform_before_fit_raises(self, sample_df):
        fe = FeatureEngineer()
        with pytest.raises(ValueError):
            fe.transform(sample_df)

    def test_get_feature_names_out_before_fit_raises(self):
        fe = FeatureEngineer()
        with pytest.raises(ValueError):
            fe.get_feature_names_out()

    def test_transform_new_data_reuses_fitted_params(self, sample_df):
        fe = FeatureEngineer()
        fe.fit(sample_df)
        new_data = pd.DataFrame({
            "age": [30],
            "salary": [55000],
            "department": ["IT"],
        })
        transformed = fe.transform(new_data)
        assert transformed.shape[0] == 1
        assert transformed.shape[1] == len(fe.get_feature_names_out())

    def test_unseen_category_does_not_raise(self, sample_df):
        fe = FeatureEngineer()
        fe.fit(sample_df)
        new_data = pd.DataFrame({
            "age": [30],
            "salary": [55000],
            "department": ["Legal"],  # not present during fit
        })
        # handle_unknown='ignore' on the OneHotEncoder means this should not raise
        transformed = fe.transform(new_data)
        assert transformed.shape[1] == len(fe.get_feature_names_out())
