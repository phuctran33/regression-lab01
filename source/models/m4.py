from ..modeling import make_pipeline

NAME = "M4 · Số đo + loài + giới tính"
FEATURES = ["flipper_length_mm", "bill_length_mm", "bill_depth_mm", "species", "sex"]


def build_model():
    return make_pipeline(FEATURES)
