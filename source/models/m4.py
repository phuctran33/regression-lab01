from sklearn.linear_model import HuberRegressor

from ..modeling import make_pipeline, run_model

NAME = "M4"
ESTIMATOR = HuberRegressor(epsilon=1.2, alpha=0.001, max_iter=1000, tol=1e-6)
GRID = {
    "regression__epsilon": (1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 2.5, 3.0),
    "regression__alpha": (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
}


def build_model():
    return make_pipeline(ESTIMATOR)


if __name__ == "__main__":
    run_model(NAME, build_model(), is_example=ESTIMATOR is None)
