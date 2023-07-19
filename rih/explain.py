"""Generate a model and explain the predictions."""

import logging
from argparse import ArgumentParser
import geopandas as gpd

logger = logging.getLogger(__name__)


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--log",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
        default="WARNING",
    )
    parser.add_argument("input", type=str, help="Input file.")

    args = parser.parse_args()

    level = getattr(logging, args.log)

    logging.basicConfig(level=level)
    logger.setLevel(level)

    input_file = args.input

    logger.info(f"Reading from input file {input_file}")
    gdf_data = gpd.GeoDataFrame.from_file(input_file)

    logger.info(f"Input shape: {gdf_data.shape}")


if __name__ == "__main__":
    main()
