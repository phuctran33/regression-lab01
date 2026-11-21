from sklearn.linear_model import Ridge

from ..modeling import make_pipeline, run_model

NAME = "M1"
ESTIMATOR = Ridge(alpha=1.0)


def build_model():
    return make_pipeline(ESTIMATOR)


if __name__ == "__main__":
    run_model(NAME, build_model(), is_example=ESTIMATOR is None)
