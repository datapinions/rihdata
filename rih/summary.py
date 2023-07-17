import logging

import geopandas as gpd
import pandas as pd

import rih.util as util
from rih.loggingargparser import LoggingArgumentParser

logger = logging.getLogger(__name__)


def main():

    parser = LoggingArgumentParser(logger)

    parser.add_argument("-o", "--output-file", required=True, help="Output file for results.")
    parser.add_argument("input_file", nargs="+", help="Input data file, as created by datagen.py")

    args = parser.parse_args()

    logger.info(f"Reading from {args.input_file}")

    gdf = pd.concat(gpd.read_file(file) for file in args.input_file)

    stats = gdf[
        [
            util.VARIABLE_TOTAL_POP,
            util.VARIABLE_MEDIAN_INCOME,
            util.VARIABLE_TOTAL_OWNER_OCCUPIED,
            util.VARIABLE_MEDIAN_VALUE
        ]
    ].describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )

    df_stats = pd.DataFrame(stats).reset_index(names="stat")

    logger.info(f"Writing to {args.output_file}")

    df_stats.to_csv(args.output_file)


if __name__ == "__main__":
    main()