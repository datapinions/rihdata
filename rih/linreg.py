import logging
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import yaml

from rih.loggingargparser import LoggingArgumentParser
from rih.util import xyw, read_data, VARIABLE_MEDIAN_INCOME

logger = logging.getLogger(__name__)


def linreg_score(gdf_cbsa_bg: pd.DataFrame, year: int, group_lh_together: bool) -> Dict:
    X, w, y = xyw(gdf_cbsa_bg, year, group_lh_together)

    # X = X.copy()
    # X[VARIABLE_MEDIAN_INCOME] = X[VARIABLE_MEDIAN_INCOME] / MAX_INCOME
    # y = y / MAX_PRICE

    y_log = np.log(y)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_log,
        y_test_log,
        w_train,
        w_test,
    ) = train_test_split(X, y, y_log, w, train_size=0.8, random_state=1771)

    regressor = LinearRegression()
    model = regressor.fit(X_train, y_train, w_train)
    score = model.score(X_test, y_test, w_test)

    regressor_log = LinearRegression()
    model_log = regressor_log.fit(X_train, y_train_log, w_train)
    score_log = model_log.score(X_test, y_test_log, w_test)

    X_train_1 = X_train[[VARIABLE_MEDIAN_INCOME]]
    X_test_1 = X_test[[VARIABLE_MEDIAN_INCOME]]

    regressor_1 = LinearRegression()
    model_1 = regressor_1.fit(X_train_1, y_train, w_train)
    score_1 = model_1.score(X_test_1, y_test, w_test)

    regressor_1_log = LinearRegression()
    model_log_1 = regressor_1_log.fit(X_train_1, y_train_log, w_train)
    score_log_1 = model_log_1.score(X_test_1, y_test_log, w_test)

    mape_score = 1 - mean_absolute_percentage_error(
        y_true=y_test, y_pred=model.predict(X_test), sample_weight=w_test
    )

    mape_score_1 = 1 - mean_absolute_percentage_error(
        y_true=y_test, y_pred=model_1.predict(X_test_1), sample_weight=w_test
    )

    mape_score_log = 1 - mean_absolute_percentage_error(
        y_true=y_test, y_pred=np.exp(model_log.predict(X_test)), sample_weight=w_test
    )

    mape_score_log_1 = 1 - mean_absolute_percentage_error(
        y_true=y_test,
        y_pred=np.exp(model_log_1.predict(X_test_1)),
        sample_weight=w_test,
    )

    mean_y = np.average(y_train, weights=w_train)
    mape_score_mean = 1 - mean_absolute_percentage_error(
        y_true=y_test,
        y_pred=[mean_y] * len(y_test),
        sample_weight=w_test,
    )

    return {
        "mean": {
            "mape_score": float(mape_score_mean),
        },
        "full": {
            "coefficients": {
                col: coefficient
                for col, coefficient in zip(X_train.columns, model.coef_.tolist())
            },
            "intercept": float(model.intercept_),
            "r2_score": float(score),
            "mape_score": float(mape_score),
        },
        "one": {
            "coefficients": {
                col: coefficient
                for col, coefficient in zip(X_train_1.columns, model_1.coef_.tolist())
            },
            "intercept": float(model_1.intercept_),
            "r2_score": float(score_1),
            "mape_score": float(mape_score_1),
        },
        "full_log": {
            "coefficients": {
                col: coefficient
                for col, coefficient in zip(X_train.columns, model_log.coef_.tolist())
            },
            "intercept": float(model_log.intercept_),
            "r2_score": float(score_log),
            "mape_score": float(mape_score_log),
        },
        "one_log": {
            "coefficients": {
                col: coefficient
                for col, coefficient in zip(
                    X_train_1.columns, model_log_1.coef_.tolist()
                )
            },
            "intercept": float(model_log_1.intercept_),
            "r2_score": float(score_log_1),
            "mape_score": float(mape_score_log_1),
        },
    }


def main():
    parser = LoggingArgumentParser(logger)

    parser.add_argument(
        "-v", "--vintage", required=True, type=int, help="Year to get data."
    )
    parser.add_argument("--group-hispanic-latino", action="store_true")
    parser.add_argument("-o", "--output-file", help="Output file for parameters.")
    parser.add_argument("input_file", help="Input file, as created by datagen.py")

    args = parser.parse_args()

    logger.info(f"Input file: {args.input_file}")

    gdf_cbsa_bg = read_data(args.input_file, drop_outliers=True)

    result = linreg_score(gdf_cbsa_bg, args.vintage, args.group_hispanic_latino)

    output_file = args.output_file

    with open(output_file, "w") as f:
        yaml.dump(result, f, sort_keys=True)


if __name__ == "__main__":
    main()
