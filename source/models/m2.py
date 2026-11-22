from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer

from ..modeling import make_pipeline, run_model

NAME = "M2"

# Estimator hồi quy cuối cùng
ESTIMATOR = LinearRegression()

# Cấu hình Spline Regression ban đầu
SPLINE_TRANSFORMER = SplineTransformer(
    degree=3,
    n_knots=3,
    knots="uniform",
    include_bias=False,
    extrapolation="constant",
)


def build_model():
    model = make_pipeline(ESTIMATOR)

    # Thay StandardScaler trong nhánh numeric bằng SplineTransformer
    model.set_params(
        preprocess__numeric__scaler=clone(SPLINE_TRANSFORMER)
    )

    return model


if __name__ == "__main__":
    run_model(NAME, build_model(), is_example=ESTIMATOR is None)
