import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from rih.loggingargparser import LoggingArgumentParser
from rih.util import (
    VARIABLE_MEDIAN_INCOME,
    VARIABLE_MEDIAN_VALUE,
    read_data,
    MAX_INCOME,
    MAX_PRICE,
)

logger = logging.getLogger(__name__)


def main():
    parser = LoggingArgumentParser(logger)

    parser.add_argument(
        "-v", "--vintage", required=True, type=int, help="Year to get data."
    )
    parser.add_argument("-o", "--output-file", help="Output file for parameters.")
    parser.add_argument("input_file", help="Input file, as created by datagen.py")

    args = parser.parse_args()

    logger.info(f"Input file: {args.input_file}")

    input_file = args.input_file
    output_file = args.output_file

    gdf_cbsa_bg = read_data(input_file, drop_outliers=False)

    df_data = gdf_cbsa_bg[[VARIABLE_MEDIAN_INCOME, VARIABLE_MEDIAN_VALUE]][
        (gdf_cbsa_bg[VARIABLE_MEDIAN_INCOME] < MAX_INCOME)
        & (gdf_cbsa_bg[VARIABLE_MEDIAN_VALUE] < MAX_PRICE)
    ]
    df_outliers = gdf_cbsa_bg[[VARIABLE_MEDIAN_INCOME, VARIABLE_MEDIAN_VALUE]][
        (gdf_cbsa_bg[VARIABLE_MEDIAN_INCOME] >= MAX_INCOME)
        | (gdf_cbsa_bg[VARIABLE_MEDIAN_VALUE] >= MAX_PRICE)
    ]

    ax = df_data.plot.scatter(
        VARIABLE_MEDIAN_INCOME,
        VARIABLE_MEDIAN_VALUE,
        label=f"Data Points (n = {len(df_data.index):,d})",
        legend=True,
        figsize=(12, 8),
        s=2,
    )

    ax = df_outliers.plot.scatter(
        VARIABLE_MEDIAN_INCOME,
        VARIABLE_MEDIAN_VALUE,
        color="red",
        label=f"Outliers (n = {len(df_outliers.index):,d})",
        legend=True,
        ax=ax,
        s=1,
    )

    ax.grid()
    ax.set_xlim(-0.05 * MAX_INCOME, 1.05 * MAX_INCOME)
    ax.set_ylim(-0.05 * MAX_PRICE, 1.05 * MAX_PRICE)

    location = Path(output_file).parent.parent.stem

    ax.set_title(f"Median Home Value vs. Median Household Income\n{location}")

    ax.set_xlabel("Median Household Income")
    ax.set_ylabel("Median Home Value")

    for handle in ax.legend().legend_handles:
        handle._sizes = [25]

    dollar_formatter = FuncFormatter(
        lambda d, pos: f"\\${d:,.0f}" if d >= 0 else f"(\\${-d:,.0f})"
    )
    ax.xaxis.set_major_formatter(dollar_formatter)
    ax.yaxis.set_major_formatter(dollar_formatter)

    logger.info(f"Output file: {output_file}")

    plt.savefig(output_file)


if __name__ == "__main__":
    main()
