from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "body_mass_g"
FEATURES = ["flipper_length_mm", "bill_length_mm", "bill_depth_mm", "species", "sex"]
CATEGORICAL = ("species", "island", "sex")


def load_penguins():
    return pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "penguins.csv")


def split_penguins(penguins):
    labeled = penguins.dropna(subset=[TARGET])
    return train_test_split(labeled, test_size=0.20, random_state=42)


def make_pipeline(estimator=None, *, features=None):
    features = FEATURES if features is None else features
    numeric_features = [feature for feature in features if feature not in CATEGORICAL]
    categorical_features = [feature for feature in features if feature in CATEGORICAL]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("regression", LinearRegression() if estimator is None else clone(estimator)),
    ])


def regression_metrics(y_true, y_pred):
    return {
        "MAE (g)": mean_absolute_error(y_true, y_pred),
        "RMSE (g)": root_mean_squared_error(y_true, y_pred),
        "R²": r2_score(y_true, y_pred),
    }


def run_model(name, model, *, is_example):
    train, test = split_penguins(load_penguins())
    model.fit(train[FEATURES], train[TARGET])
    predicted = model.predict(test[FEATURES])

    estimator_name = type(model.named_steps["regression"]).__name__
    print(f"{name} | {estimator_name}")
    if is_example:
        print("Đang chạy cấu hình mẫu. Điền ESTIMATOR khi đã chọn thuật toán.")
    for metric, value in regression_metrics(test[TARGET], predicted).items():
        print(f"{metric}  {value:.3f}")
