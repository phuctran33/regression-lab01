from sklearn.linear_model import HuberRegressor

from ..modeling import make_pipeline, run_model

NAME = "M4"
ESTIMATOR = HuberRegressor(epsilon=1.2, alpha=0.001, max_iter=1000, tol=1e-6)


def build_model():
    return make_pipeline(ESTIMATOR)


if __name__ == "__main__":
    run_model(NAME, build_model(), is_example=ESTIMATOR is None)
