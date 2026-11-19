from ..modeling import make_pipeline

NAME = "M1 · Chiều dài cánh bơi"
FEATURES = ["flipper_length_mm"]


def build_model():
    return make_pipeline(FEATURES)
