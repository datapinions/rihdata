"""Utilities to support notebooks in this project."""

import pandas as pd
import geopandas as gpd
from censusdis import data as ced
from censusdis.datasets import ACS5

# This is what the column for CBSA is returned as.
COLUMN_CBSA = "METROPOLITAN_STATISTICAL_AREA_MICROPOLITAN_STATISTICAL_AREA"

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
    "Hispanic Other": FRAC_H_OTHER,
}

# Median household income in last 12 months.
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
    {
        f"frac_{var}": f"Fraction of population that is {name}"
        for var, name in _BASE_VAR_NAMES.items()
    },
    **_BASE_VAR_NAMES,
)

VAR_NAMES = dict(
    {
        f"imp_frac_{var}": f"Impact of {name} on Median Home Value"
        for var, name in _BASE_VAR_NAMES.items()
    },
    **VAR_NAMES,
)

VAR_NAMES = dict(
    {
        f"rel_imp_frac_{var}": f"Relative Impact of {name} on Median Home Value"
        for var, name in _BASE_VAR_NAMES.items()
    },
    **VAR_NAMES,
)

VAR_NAMES = dict(
    {
        VARIABLE_MEDIAN_INCOME: "Median Household Income",
        VARIABLE_MEDIAN_VALUE: "Medain Home Value",
    },
    **VAR_NAMES,
)


def add_fractional_population(leaves_race_ethnicity, df: pd.DataFrame) -> pd.DataFrame:
    for leaf in leaves_race_ethnicity + [
        VARIABLE_BLACK_ALONE,
        VARIABLE_WHITE_ALONE,
        VARIABLE_NH_BLACK,
        VARIABLE_NH_WHITE,
    ]:
        df[f"frac_{leaf}"] = df[leaf] / df[VARIABLE_TOTAL_POP]

    return df


def xyw(gdf_cbsa_bg, year, group_lh_together: bool):
    # There should be only one CBSA at this point.
    assert (
        len(
            gdf_cbsa_bg[
                "METROPOLITAN_STATISTICAL_AREA_MICROPOLITAN_STATISTICAL_AREA"
            ].unique()
        )
        == 1
    )

    leaves_race_ethnicity = ced.variables.group_leaves(ACS5, year, GROUP_RACE_ETHNICITY)

    if group_lh_together:
        demographic_fraction_vars = [
            f"frac_{leaf}" for leaf in leaves_race_ethnicity if leaf <= VARIABLE_H_TOTAL
        ] + [f"frac_{VARIABLE_H_TOTAL}"]
    else:
        demographic_fraction_vars = [f"frac_{leaf}" for leaf in leaves_race_ethnicity]

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
                (gdf[VARIABLE_MEDIAN_VALUE] < MAX_PRICE)
                & (gdf[VARIABLE_MEDIAN_INCOME] < MAX_INCOME)
            ]
        )

    return gdf
