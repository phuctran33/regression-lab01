from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ("species", "island", "sex")


def make_pipeline(features):
    numeric_features = [feature for feature in features if feature not in CATEGORICAL]
    categorical_features = [feature for feature in features if feature in CATEGORICAL]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("regression", LinearRegression()),
    ])
