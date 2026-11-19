from ..modeling import make_pipeline, run_model

NAME = "M2"
ESTIMATOR = None  # TODO: Gán một estimator hồi quy tuyến tính do bạn chọn.


def build_model():
    return make_pipeline(ESTIMATOR)


if __name__ == "__main__":
    run_model(NAME, build_model(), is_example=ESTIMATOR is None)
