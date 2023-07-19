"""Utilities to support notebooks in this project."""
import os.path

import numpy as np
import pandas as pd
import geopandas as gpd
from censusdis import data as ced
from censusdis.datasets import ACS5
from sklearn.model_selection import train_test_split
import xgboost
from bayes_opt import BayesianOptimization
from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

# The top 50 most populated CBSA in 2020

CBSA_NYC = "35620"  # New York-Newark-Jersey City, NY-NJ-PA Metro Area
CBSA_LAX = "31080"  # Los Angeles-Long Beach-Anaheim, CA Metro Area
CBSA_ORD = "16980"  # Chicago-Naperville-Elgin, IL-IN-WI Metro Area
CBSA_DFW = "19100"  # Dallas-Fort Worth-Arlington, TX Metro Area
CBSA_IAH = "26420"  # Houston-The Woodlands-Sugar Land, TX Metro Area

CBSA_DCA = "47900"  # Washington-Arlington-Alexandria, DC-VA-MD-WV Metro Area
CBSA_MIA = "33100"  # Miami-Fort Lauderdale-Pompano Beach, FL Metro Area
CBSA_PHL = "37980"  # Philadelphia-Camden-Wilmington, PA-NJ-DE-MD Metro Area
CBSA_ATL = "12060"  # Atlanta-Sandy Springs-Alpharetta, GA Metro Area
CBSA_PHX = "38060"  # Phoenix-Mesa-Chandler, AZ Metro Area

CBSA_BOS = "14460"  # Boston-Cambridge-Newton, MA-NH Metro Area
CBSA_SFO = "41860"  # San Francisco-Oakland-Berkeley, CA Metro Area
CBSA_ONT = "40140"  # Riverside-San Bernardino-Ontario, CA Metro Area
CBSA_DTW = "19820"  # Detroit-Warren-Dearborn, MI Metro Area
CBSA_SEA = "42660"  # Seattle-Tacoma-Bellevue, WA Metro Area

CBSA_MSP = "33460"  # Minneapolis-St. Paul-Bloomington, MN-WI Metro Area
CBSA_SAN = "41740"  # San Diego-Chula Vista-Carlsbad, CA Metro Area
CBSA_TPA = "45300"  # Tampa-St. Petersburg-Clearwater, FL Metro Area
CBSA_DEN = "19740"  # Denver-Aurora-Lakewood, CO Metro Area
CBSA_STL = "41180"  # St. Louis, MO-IL Metro Area

CBSA_BWI = "12580"  # Baltimore-Columbia-Towson, MD Metro Area
CBSA_CLT = "16740"  # Charlotte-Concord-Gastonia, NC-SC Metro Area
CBSA_MCO = "36740"  # Orlando-Kissimmee-Sanford, FL Metro Area
CBSA_SAT = "41700"  # San Antonio-New Braunfels, TX Metro Area
CBSA_PDX = "38900"  # Portland-Vancouver-Hillsboro, OR-WA Metro Area

CBSA_SMF = "40900"  # Sacramento-Roseville-Folsom, CA Metro Area
CBSA_PIT = "38300"  # Pittsburgh, PA Metro Area
CBSA_LAS = "29820"  # Las Vegas-Henderson-Paradise, NV Metro Area
CBSA_CVG = "17140"  # Cincinnati, OH-KY-IN Metro Area
CBSA_AUS = "12420"  # Austin-Round Rock-Georgetown, TX Metro Area

CBSA_MCI = "28140"  # Kansas City, MO-KS Metro Area
CBSA_CMH = "18140"  # Columbus, OH Metro Area
CBSA_SJU = "41980"  # San Juan-Bayamón-Caguas, PR Metro Area
CBSA_CLE = "17460"  # Cleveland-Elyria, OH Metro Area
CBSA_IND = "26900"  # Indianapolis-Carmel-Anderson, IN Metro Area

CBSA_SJC = "41940"  # San Jose-Sunnyvale-Santa Clara, CA Metro Area
CBSA_BNA = "34980"  # Nashville-Davidson--Murfreesboro--Franklin, TN Metro Area
CBSA_ORF = "47260"  # Virginia Beach-Norfolk-Newport News, VA-NC Metro Area
CBSA_PVD = "39300"  # Providence-Warwick, RI-MA Metro Area
CBSA_MKE = "33340"  # Milwaukee-Waukesha, WI Metro Area

CBSA_JAX = "27260"  # Jacksonville, FL Metro Area
CBSA_OKC = "36420"  # Oklahoma City, OK Metro Area
CBSA_RDU = "39580"  # Raleigh-Cary, NC Metro Area
CBSA_MEM = "32820"  # Memphis, TN-MS-AR Metro Area
CBSA_RIC = "40060"  # Richmond, VA Metro Area

CBSA_MSY = "35380"  # New Orleans-Metairie, LA Metro Area
CBSA_SDF = "31140"  # Louisville/Jefferson County, KY-IN Metro Area
CBSA_SLC = "41620"  # Salt Lake City, UT Metro Area
CBSA_BDL = "25540"  # Hartford-East Hartford-Middletown, CT Metro Area
CBSA_BUF = "15380"  # Buffalo-Cheektowaga, NY Metro Area

CBSA_TOP_50_ACS5_2020 = [
    CBSA_NYC, CBSA_LAX, CBSA_ORD, CBSA_DFW, CBSA_IAH,
    CBSA_DCA, CBSA_MIA, CBSA_PHL, CBSA_ATL, CBSA_PHX,
    CBSA_BOS, CBSA_SFO, CBSA_ONT, CBSA_DTW, CBSA_SEA,
    CBSA_MSP, CBSA_SAN, CBSA_TPA, CBSA_DEN, CBSA_STL,
    CBSA_BWI, CBSA_CLT, CBSA_MCO, CBSA_SAT, CBSA_PDX,
    CBSA_SMF, CBSA_PIT, CBSA_LAS, CBSA_CVG, CBSA_AUS,
    CBSA_MCI, CBSA_CMH, CBSA_SJU, CBSA_CLE, CBSA_IND,
    CBSA_SJC, CBSA_BNA, CBSA_ORF, CBSA_PVD, CBSA_MKE,
    CBSA_JAX, CBSA_OKC, CBSA_RDU, CBSA_MEM, CBSA_RIC,
    CBSA_MSY, CBSA_SDF, CBSA_SLC, CBSA_BDL, CBSA_BUF,
]

# This is what the column for CBSA is returned as.
COLUMN_CBSA = 'METROPOLITAN_STATISTICAL_AREA_MICROPOLITAN_STATISTICAL_AREA'


# Black and white alone populations.
# See https://api.census.gov/data/2021/acs/acs5/groups/B02001.html
VARIABLE_BLACK_ALONE = "B02001_003E"
VARIABLE_WHITE_ALONE = "B02001_002E"

FRAC_BLACK_ALONE = f"frac_{VARIABLE_BLACK_ALONE}"
FRAC_WHITE_ALONE = f"frac_{VARIABLE_WHITE_ALONE}"

# A group of variables counting people by race and ethnicity.
# See https://api.census.gov/data/2019/acs/acs5/groups/B03002.html
GROUP_RACE_ETHNICITY = "B03002"

VARIABLE_TOTAL_POP = f"{GROUP_RACE_ETHNICITY}_001E"

VARIABLE_NH_TOTAL = f"{GROUP_RACE_ETHNICITY}_002E"
VARIABLE_H_TOTAL = f"{GROUP_RACE_ETHNICITY}_012E"

VARIABLE_NH_BLACK = f"{GROUP_RACE_ETHNICITY}_004E"
VARIABLE_NH_WHITE = f"{GROUP_RACE_ETHNICITY}_003E"
VARIABLE_NH_ASIAN = f"{GROUP_RACE_ETHNICITY}_006E"

VARIABLE_H_WHITE = f"{GROUP_RACE_ETHNICITY}_013E"
VARIABLE_H_BLACK = f"{GROUP_RACE_ETHNICITY}_014E"
VARIABLE_H_OTHER = f"{GROUP_RACE_ETHNICITY}_018E"

FRAC_NH_BLACK = f"frac_{VARIABLE_NH_BLACK}"
FRAC_NH_WHITE = f"frac_{VARIABLE_NH_WHITE}"
FRAC_NH_ASIAN = f"frac_{VARIABLE_NH_ASIAN}"

FRAC_H_WHITE = f"frac_{VARIABLE_H_WHITE}"
FRAC_H_OTHER = f"frac_{VARIABLE_H_OTHER}"

FRAC_VARIABLES = {
    "Non-Hispanic White": FRAC_NH_WHITE,
    "Non-Hispanic Black": FRAC_NH_BLACK,
    "Non-Hispanic Asian": FRAC_NH_ASIAN,
    "Hispanic White": FRAC_H_WHITE,
    "Hispanic White": FRAC_H_OTHER,
}

# Median houshold income in last 12 months.
# See https://api.census.gov/data/2021/acs/acs5/groups/B19013.html
GROUP_MEDIAN_INCOME = "B19013"
VARIABLE_MEDIAN_INCOME = f"{GROUP_MEDIAN_INCOME}_001E"

# Median home value.
# See https://api.census.gov/data/2021/acs/acs5/groups/B25077.html
VARIABLE_MEDIAN_VALUE = "B25077_001E"

# Total owner-occupied households.
# See https://api.census.gov/data/2020/acs/acs5/groups/B25003.html
VARIABLE_TOTAL_OWNER_OCCUPIED = "B25003_002E"


_BASE_VAR_NAMES = {
    VARIABLE_NH_WHITE: "Non-Hispanic White",
    VARIABLE_NH_BLACK: "Non-Hispanic Black",
    VARIABLE_NH_ASIAN: "Non-Hispanic Asian",

    VARIABLE_H_WHITE: "Hispanic or Latino White",
    VARIABLE_H_BLACK: "Hispanic or Latino Black",
    VARIABLE_H_OTHER: "Hispanic or Latino Some Other Race",
}

VAR_NAMES = dict(
    {f"frac_{var}": f"Fraction of population that is {name}" for var, name in _BASE_VAR_NAMES.items()},
    **_BASE_VAR_NAMES
)

VAR_NAMES = dict(
    {f"imp_frac_{var}": f"Impact of {name} on Median Home Value" for var, name in _BASE_VAR_NAMES.items()},
    **VAR_NAMES
)

VAR_NAMES = dict(
    {f"rel_imp_frac_{var}": f"Relative Impact of {name} on Median Home Value" for var, name in _BASE_VAR_NAMES.items()},
    **VAR_NAMES
)

VAR_NAMES = dict(
    {
        VARIABLE_MEDIAN_INCOME: "Median Household Income",
        VARIABLE_MEDIAN_VALUE: "Medain Home Value",
    },
    **VAR_NAMES
)


def add_fractional_population(leaves_race_ethnicity, df: pd.DataFrame) -> pd.DataFrame:
    for leaf in leaves_race_ethnicity + [
        VARIABLE_BLACK_ALONE, VARIABLE_WHITE_ALONE, VARIABLE_NH_BLACK, VARIABLE_NH_WHITE
    ]:
        df[f'frac_{leaf}'] = df[leaf] / df[VARIABLE_TOTAL_POP]

    return df


def xgb_cv_objective(X, y, w):
    dmatrix = xgboost.DMatrix(X, y, weight=w)


def xgb_r2_objective(X_train, X_test, y_train, y_test, w_train, w_test):
    def objective(n_estimators, max_depth):
        # Truncate to ints as suggested in
        # https://github.com/fmfn/BayesianOptimization/blob/master/examples/advanced-tour.ipynb
        xgb = xgboost.XGBRegressor(
            n_estimators=int(np.round(n_estimators)),
            max_depth=int(np.round(max_depth)),
        )
        xgb = xgb.fit(X=X_train, y=y_train, sample_weight=w_train)
        score = xgb.score(X=X_test, y=y_test, sample_weight=w_test)
        return score

    return objective


def split_xyw(df, demographic_fraction_vars):
    X = df[[VARIABLE_MEDIAN_INCOME] + demographic_fraction_vars]
    y = df[VARIABLE_MEDIAN_VALUE]
    w = df[VARIABLE_TOTAL_OWNER_OCCUPIED]

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, w, test_size=0.2, random_state=17
    )

    return X_train, X_test, y_train, y_test, w_train, w_test


def hp_optimize(df, demographic_fraction_vars):
    X_train, X_test, y_train, y_test, w_train, w_test = split_xyw(df, demographic_fraction_vars)

    pbounds = {
        'n_estimators': (10, 50),
        'max_depth': (2, 10),
    }

    optimizer = BayesianOptimization(
        f=xgb_r2_objective(X_train, X_test, y_train, y_test, w_train, w_test),
        pbounds=pbounds,
        verbose=1,
        random_state=17*93,
        allow_duplicate_points=True,
    )

    optimizer.maximize(init_points=5, n_iter=50)

    result = optimizer.max

    result['params']['max_depth'] = int(np.round(result['params']['max_depth']))
    result['params']['n_estimators'] = int(np.round(result['params']['n_estimators']))

    return result


def force_plots(cbsa_short_name, shap_values, X_test, output_dir: str):

    dollar_formatter = FuncFormatter(
        lambda d, pos: f'\${d:,.0f}' if d >= 0 else f'(\${-d:,.0f})'
    )

    plot_variables = [
        VARIABLE_NH_WHITE, VARIABLE_NH_BLACK, VARIABLE_NH_ASIAN,
        VARIABLE_H_WHITE,
        VARIABLE_H_BLACK,
        VARIABLE_H_OTHER,
    ]

    fig, axes = plt.subplots(
        nrows=len(plot_variables), ncols=1, figsize=(12, 4 * len(plot_variables))
    )

    fig.tight_layout(pad=5.0)

    max_force = min_force = 0

    # max_force = np.quantile(shap_values[:,1:], 0.997)
    # min_force = np.quantile(shap_values[:,1:], 0.01)

    # print(max_force)

    for force_variable, ax in zip(plot_variables, axes):
        force_frac = f"frac_{force_variable}"

        df_force = pd.DataFrame(
            shap_values,
            columns=X_test.columns,
            index=X_test.index,
        )[[force_frac]]

        df_force = df_force.rename({force_frac: f"SHAP_{force_variable}"}, axis='columns')

        df_force[force_frac] = X_test[force_frac]

        df_force = df_force.sort_values(force_frac)

        df_force['smooth'] = gaussian_filter1d(df_force[f"SHAP_{force_variable}"], 5)

        max_force = max([max_force, df_force['smooth'].max()])
        min_force = min([min_force, df_force['smooth'].min()])

        ax = df_force.plot.scatter(
            force_frac, f"SHAP_{force_variable}",
            color='darkgrey',
            s=10,
            ax=ax
        )

        ax = df_force.plot(
            force_frac, 'smooth', color='C1', legend=False,
            linewidth=3,
            ax=ax
        )

        ax.set_xticks(np.arange(0.0, 1.01, 0.1))

        ax.yaxis.set_major_formatter(dollar_formatter)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

        ax.set_title(
            f'{cbsa_short_name.title()} Area CBSA - Impact of {VAR_NAMES[force_frac].title()} on Median Home Value')
        ax.set_xlabel(VAR_NAMES[force_frac])
        ax.set_ylabel("Impact")

        ax.axhline(0, color='black', zorder=1)
        ax.grid()

    max_force = (max_force // 10_000) * 10_000 + 20_000
    min_force = (min_force // 10_000) * 10_000 - 20_000

    for ax in axes:
        ax.set_ylim(min_force, max_force)

    fig.savefig(os.path.join(output_dir, f"{cbsa_short_name.replace(' ', '-')}_tear_sheet.png"))

    plt.close(fig)


def xyw(gdf_cbsa_bg, year, group_lh_together: bool):
    # There should be only one CBSA at this point.
    assert len(gdf_cbsa_bg['METROPOLITAN_STATISTICAL_AREA_MICROPOLITAN_STATISTICAL_AREA'].unique()) == 1

    leaves_race_ethnicity = ced.variables.group_leaves(ACS5, year, GROUP_RACE_ETHNICITY)

    if group_lh_together:
        demographic_fraction_vars = [
            f'frac_{leaf}'
            for leaf in leaves_race_ethnicity
            if leaf <= VARIABLE_H_TOTAL
        ] + [f'frac_{VARIABLE_H_TOTAL}']
    else:
        demographic_fraction_vars = [f'frac_{leaf}' for leaf in leaves_race_ethnicity]

    X = gdf_cbsa_bg[[VARIABLE_MEDIAN_INCOME] + demographic_fraction_vars]
    y = gdf_cbsa_bg[VARIABLE_MEDIAN_VALUE]
    w = gdf_cbsa_bg[VARIABLE_TOTAL_OWNER_OCCUPIED]

    return X, w, y


MAX_INCOME = 250_001
MAX_PRICE = 2_000_001


def read_data(filename: str, drop_outliers: bool = True, **kwargs) -> gpd.GeoDataFrame:
    """Read a data file and optionally drop outliers."""
    gdf = gpd.read_file(filename, **kwargs)

    if drop_outliers:
        gdf = gpd.GeoDataFrame(
            gdf[
                (gdf[VARIABLE_MEDIAN_VALUE] < MAX_PRICE) &
                (gdf[VARIABLE_MEDIAN_INCOME] < MAX_INCOME)
            ]
        )

    return gdf
