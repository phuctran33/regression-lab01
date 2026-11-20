from sklearn.linear_model import LinearRegression

from ..modeling import FEATURES, TARGET, load_penguins, make_pipeline, regression_metrics, split_penguins

NAME = "OLS"


def build_model():
    return make_pipeline(LinearRegression())


def main():
    train, test = split_penguins(load_penguins())
    model = build_model().fit(train[FEATURES], train[TARGET])
    predictions = model.predict(test[FEATURES])

    regression = model.named_steps["regression"]
    preprocessor = model.named_steps["preprocess"]
    print(f"{NAME} | {type(regression).__name__}")
    for metric, value in regression_metrics(test[TARGET], predictions).items():
        print(f"{metric}  {value:.3f}")
    print(f"Hệ số chặn (g)  {regression.intercept_:.3f}")
    for feature, coefficient in zip(preprocessor.get_feature_names_out(), regression.coef_):
        print(f"{feature}  {coefficient:+.3f}")


if __name__ == "__main__":
    main()
