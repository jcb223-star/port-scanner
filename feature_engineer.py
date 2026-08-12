import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Feature Engineering class compatible with Scikit-Learn pipelines.
    Handles missing values, encoding, scaling, and custom feature creation.
    """

    def __init__(self):
        self.pipeline = None
        self.numeric_cols = None
        self.categorical_cols = None

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 1: Domain-specific feature creation or extraction.
        Modify this method based on your dataset's specific requirements.
        """
        df = df.copy()

        # Example 1: Combining features (e.g., total income from components)
        # if 'income_primary' in df.columns and 'income_secondary' in df.columns:
        #     df['total_income'] = df['income_primary'] + df['income_secondary']

        # Example 2: Extracting date features
        # if 'date' in df.columns:
        #     df['date'] = pd.to_datetime(df['date'])
        #     df['year'] = df['date'].dt.year
        #     df['month'] = df['date'].dt.month
        #     df.drop(columns=['date'], inplace=True)

        return df

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """
        Learn parameters (means, categories, etc.) from the training data.
        """
        X_transformed = self._create_features(X)

        # Automatically identify column types
        self.numeric_cols = X_transformed.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = X_transformed.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Define preprocessing for numeric data (Impute missing + Scale)
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Define preprocessing for categorical data (Impute missing + One-Hot Encode)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine preprocessing steps using ColumnTransformer.
        # numeric_cols + categorical_cols already partition every dtype that
        # select_dtypes can hand back, so anything left over is unexpected —
        # drop (rather than silently passthrough) so a new dtype fails loudly.
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='drop'
        )

        # Fit the underlying pipeline
        self.pipeline.fit(X_transformed)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply the fitted transformations to new data.
        """
        if self.pipeline is None:
            raise ValueError("The FeatureEngineer instance has not been fitted yet. Call 'fit' first.")

        X_transformed = self._create_features(X)
        return self.pipeline.transform(X_transformed)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """
        Names of the output columns/features, e.g. for wrapping the
        transformed array back into a DataFrame.
        """
        if self.pipeline is None:
            raise ValueError("The FeatureEngineer instance has not been fitted yet. Call 'fit' first.")

        return self.pipeline.get_feature_names_out(input_features)


# --- Example Usage ---
if __name__ == "__main__":
    # Dummy dataset for testing
    data = pd.DataFrame({
        'age': [25, 32, np.nan, 47, 51],
        'salary': [50000, 60000, 80000, np.nan, 120000],
        'department': ['HR', 'IT', 'Finance', 'IT', np.nan],
        'purchased': ['No', 'Yes', 'No', 'Yes', 'Yes']
    })

    X = data.drop(columns=['purchased'])
    y = data['purchased']

    # Initialize and apply feature engineer
    fe = FeatureEngineer()
    X_processed = fe.fit_transform(X)

    print("Processed Feature Shape:", X_processed.shape)
    print("Processed Array:\n", X_processed)
    print("Feature Names:\n", fe.get_feature_names_out())
