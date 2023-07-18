
import logging
from pathlib import Path

import censusdis.data as ced
import matplotlib.pyplot as plt
from censusdis.datasets import ACS5
from matplotlib.ticker import FuncFormatter, PercentFormatter

import rih.util as util
from rih.loggingargparser import LoggingArgumentParser
from rih.util import read_data

logger = logging.getLogger(__name__)


def main():

    parser = LoggingArgumentParser(logger)

    parser.add_argument('-v', '--vintage', required=True, type=int, help="Year to get data.")
    parser.add_argument('--group-hispanic-latino', action='store_true')

    parser.add_argument("-o", "--output-dir", required=True, help="Output directory for plots.")

    parser.add_argument("input_file", help="Input file, as created by datagen.py")

    args = parser.parse_args()

    logging.info(f"{args.input_file} -> {args.output_dir}")

    gdf_cbsa_bg = read_data(args.input_file, drop_outliers=True)

    year = args.vintage
    all_variables = ced.variables.all_variables(ACS5, year, util.GROUP_RACE_ETHNICITY)

    dollar_formatter = FuncFormatter(
        lambda d, pos: f'\\${d:,.0f}' if d >= 0 else f'(\\${-d:,.0f})'
    )

    X, _, _ = util.xyw(gdf_cbsa_bg, year, group_lh_together=args.group_hispanic_latino)

    for feature in X.columns:
        if not feature.startswith('frac_'):
            continue

        variable = feature[5:]  # Remove leading "frac_"

        label = all_variables[all_variables['VARIABLE'] == variable]['LABEL'].iloc[0]

        label = label.replace("Estimate!!Total:!!", "")
        label = label.replace(":!!", "; ")

        ax = gdf_cbsa_bg.plot.scatter(
            feature,
            util.VARIABLE_MEDIAN_VALUE,
            figsize=(12, 8),
            s=2,
        )

        ax.grid()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05 * util.MAX_PRICE, 1.05 * util.MAX_PRICE)

        ax.yaxis.set_major_formatter(dollar_formatter)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

        name = Path(args.output_dir).parent.name.replace('_', ' ')
        ax.set_title(f'Median Home Value vs.\nPercentage {label}\nin {name}')
        ax.set_xlabel(label)
        ax.set_ylabel("Median Home Value")

        filename = label.replace(" ", "-").replace(";", '')

        plt.savefig(Path(args.output_dir) / f"{filename}.png")


if __name__ == "__main__":
    main()