from ..modeling import make_pipeline

NAME = "M3 · Số đo + loài"
FEATURES = ["flipper_length_mm", "bill_length_mm", "bill_depth_mm", "species"]


def build_model():
    return make_pipeline(FEATURES)
