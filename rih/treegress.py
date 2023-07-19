import logging

import censusdis.data as ced
import numpy as np
import pandas as pd
import sklearn.metrics
import xgboost
import yaml
from bayes_opt import BayesianOptimization
from censusdis.datasets import ACS5
from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV, KFold

import rih.util as util
from rih.loggingargparser import LoggingArgumentParser
from rih.util import xyw, read_data

logger = logging.getLogger(__name__)


def validate_gdf_cbsa_bg(gdf_cbsa_bg: pd.DataFrame, year: int):
    # There should be only one CBSA at this point.
    assert (
        len(
            gdf_cbsa_bg[
                "METROPOLITAN_STATISTICAL_AREA_MICROPOLITAN_STATISTICAL_AREA"
            ].unique()
        )
        == 1
    )

    leaf_cols = ced.variables.group_leaves(ACS5, year, util.GROUP_RACE_ETHNICITY)
    leaf_frac_cols = [f"frac_{col}" for col in leaf_cols]

    all_frac_cols = [col for col in gdf_cbsa_bg.columns if col.startswith("frac_")]

    # The fractions for leaves should be between zerp and one.
    assert (gdf_cbsa_bg[leaf_frac_cols] >= 0.0).all().all()
    assert (gdf_cbsa_bg[leaf_frac_cols] <= 1.0).all().all()

    assert (gdf_cbsa_bg[all_frac_cols] >= 0.0).all().all()
    assert (gdf_cbsa_bg[all_frac_cols] <= 1.0).all().all()

    # The fractions across the leaves should add up to 1.0
    sum_frac = gdf_cbsa_bg[leaf_frac_cols].sum(axis="columns")

    # Quick way to assert almost equal outside a unit test environment.
    assert (sum_frac >= 0.9999999).all()
    assert (sum_frac <= 1.0000001).all()

    logger.info("All validations passed.")


def xgb_r2_objective(X, y, w, random_state):
    def objective(n_estimators, max_depth, **kwargs):
        scores = []

        n_splits = 5
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for train_index, test_index in folds.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            w_train, w_test = w.iloc[train_index], w.iloc[test_index]

            # Truncate to ints as suggested in
            # https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
            xgb = xgboost.XGBRegressor(
                # eval_metric="rmsle",
                eval_metric="mape",
                n_estimators=int(np.round(n_estimators)),
                max_depth=int(np.round(max_depth)),
                **kwargs,
            )

            xgb = xgb.fit(X=X_train, y=y_train, sample_weight=w_train)
            # score = xgb.score(X=X_test, y=y_test, sample_weight=w_test)
            y_hat = xgb.predict(X_test)
            score = 1 - sklearn.metrics.mean_absolute_percentage_error(
                y_true=y_test, y_pred=y_hat, sample_weight=w_test
            )
            scores.append(score)

        logger.info(f"Individual scores: {scores}.")

        return np.mean(scores)

    return objective


def hp_optimize(X, y, w):
    pbounds = {
        "n_estimators": (10, 50),
        "max_depth": (2, 10),
        # 'subsample': (0.1, 0.9),
        "learning_rate": (0.02, 0.2),
        # 'alpha': (0, 1),
    }

    optimizer = BayesianOptimization(
        f=xgb_r2_objective(X, y, w, 324644339),
        pbounds=pbounds,
        verbose=1,
        random_state=17 * 93,
        allow_duplicate_points=True,
    )

    optimizer.maximize(init_points=5, n_iter=25)

    result = optimizer.max

    result["params"]["max_depth"] = int(np.round(result["params"]["max_depth"]))
    result["params"]["n_estimators"] = int(np.round(result["params"]["n_estimators"]))
    result["params"]["learning_rate"] = float(result["params"]["learning_rate"])

    result["target"] = float(result["target"])

    return result


def optimize(gdf_cbsa_bg: pd.DataFrame, year: int, group_lh_together: bool):
    X, w, y = xyw(gdf_cbsa_bg, year, group_lh_together)

    result = hp_optimize(X, y, w)

    return result


def optimize2(gdf_cbsa_bg: pd.DataFrame, year: int, group_lh_together: bool):
    X, w, y = xyw(gdf_cbsa_bg, year, group_lh_together)

    reg_xgb = xgboost.XGBRegressor(eval_metric="mape")

    # reg_rf = xgboost.XGBRFRegressor(eval_metric="mape")

    param_dist = {
        "n_estimators": stats.randint(10, 100),
        "learning_rate": stats.uniform(0.01, 0.07),
        # 'subsample': stats.uniform(0.3, 0.7),
        "max_depth": stats.randint(2, 6),
        # 'colsample_bytree': stats.uniform(0.5, 0.45),
        "min_child_weight": stats.randint(1, 4),
    }

    def score_neg_weighted_mean_absolute_percentage_error(estimator, X_val, y_val):
        # Get the weights that correspond to the fold
        # we are trying to validate.
        w_val = w.loc[X_val.index.values]
        y_val_hat = estimator.predict(X_val)
        score = mean_absolute_percentage_error(y_val, y_val_hat, sample_weight=w_val)

        return -score

    reg = RandomizedSearchCV(
        reg_xgb,
        param_distributions=param_dist,
        n_iter=100,
        error_score=0,
        scoring=score_neg_weighted_mean_absolute_percentage_error,  # 'neg_mean_absolute_percentage_error',
        verbose=0,
        n_jobs=-1,
        random_state=17,
    )

    reg.fit(X, y)

    result = {
        "params": reg.best_params_,
        "target": 1 + float(reg.best_score_),
    }

    result["params"]["learning_rate"] = float(result["params"]["learning_rate"])

    return result


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

    validate_gdf_cbsa_bg(gdf_cbsa_bg, args.vintage)

    # estimators, results, score = optimize(gdf_cbsa_bg, args.vintage)

    # for estimator in estimators:
    #     logger.info(f"Params: {estimator.max_depth} {estimator.n_estimators}")

    result = optimize2(gdf_cbsa_bg, args.vintage, args.group_hispanic_latino)

    output_file = args.output_file

    with open(output_file, "w") as f:
        yaml.dump(result, f, sort_keys=True)


if __name__ == "__main__":
    main()
