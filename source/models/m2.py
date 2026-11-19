from ..modeling import make_pipeline

NAME = "M2 · Số đo cơ thể"
FEATURES = ["flipper_length_mm", "bill_length_mm", "bill_depth_mm"]


def build_model():
    return make_pipeline(FEATURES)
