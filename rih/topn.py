
import logging
from argparse import ArgumentParser
import geopandas as gpd
import censusdis.data as ced

from censusdis.datasets import ACS5

import rih.util as util


logger = logging.getLogger(__name__)


def top_n_cbsas(n: int, vintage: int) -> gpd.GeoDataFrame:
    """The n largest CBSAs by total population."""
    df_all_cbsa = ced.download(
        ACS5,
        vintage,
        ["NAME", util.VARIABLE_TOTAL_POP],
        metropolitan_statistical_area_micropolitan_statistical_area='*',
    )

    df_top_n = df_all_cbsa[
        [util.COLUMN_CBSA, "NAME", util.VARIABLE_TOTAL_POP]
    ].nlargest(n, util.VARIABLE_TOTAL_POP)

    return df_top_n


def main():
    parser = ArgumentParser()

    parser.add_argument(
        '--log',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help="Logging level.",
        default='WARNING'
    )
    parser.add_argument('-n', required=True, type=int, help="Number of top CBSAs.")
    parser.add_argument('-v', '--vintage', type=int, help="Year to get data.")
    parser.add_argument('-s', '--suffix', type=str, help='Data file suffix.', default='.geojson')
    args = parser.parse_args()

    level = getattr(logging, args.log)

    logging.basicConfig(level=level)
    logger.setLevel(level)

    df_top_n = top_n_cbsas(args.n, args.vintage)

    for row in df_top_n[
        ['METROPOLITAN_STATISTICAL_AREA_MICROPOLITAN_STATISTICAL_AREA', 'NAME']
    ].itertuples(index=False):
        print(f"{row[1].replace(' ', '_').replace('/', '_')}/{row[0]}{args.suffix}")


if __name__ == "__main__":
    main()