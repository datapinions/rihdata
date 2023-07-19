"""Utilities to download on-spine data clipped to off-spine geometries like CBSAs."""

from typing import Generator, Iterable, List
import logging
from dataclasses import dataclass, field
from functools import cached_property
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import geopandas as gpd
import censusdis.data as ced
import censusdis.maps as cem
from censusdis.datasets import ACS5
from censusdis.states import NAMES_FROM_IDS
from censusdis.values import ALL_SPECIAL_VALUES

import rih.util as util


logger = logging.getLogger(__name__)


@dataclass
class Downloader:
    dataset: str
    vintage: ced.VintageType

    top_n_cbsas: List[str]

    variable_total_pop: str = util.VARIABLE_TOTAL_POP

    download_variables: List[str] = field(
        default_factory=lambda: [
            "NAME",
            util.VARIABLE_MEDIAN_VALUE,
            util.VARIABLE_MEDIAN_INCOME,
            util.VARIABLE_TOTAL_POP,
            util.VARIABLE_TOTAL_OWNER_OCCUPIED,
            util.VARIABLE_WHITE_ALONE,
            util.VARIABLE_BLACK_ALONE,
            util.VARIABLE_NH_TOTAL,
            util.VARIABLE_H_TOTAL,
        ]
    )

    @cached_property
    def all_states(self) -> gpd.GeoDataFrame:
        """The names and geometries of all states."""
        return ced.download(
            self.dataset,
            self.vintage,
            ["NAME"],
            state="*",
            with_geometry=True,
        )

    @cached_property
    def gdf_top_n_cbsas(self) -> gpd.GeoDataFrame:
        gdf = ced.download(
            dataset=self.dataset,
            vintage=self.vintage,
            download_variables=["NAME"],
            metropolitan_statistical_area_micropolitan_statistical_area=self.top_n_cbsas,
            with_geometry=True,
        )
        return gdf

    @cached_property
    def states_covered_by_cbsas(self) -> List[str]:
        """Generate a list of the states that intersect one or more of the top n CBSAs"""
        gdf_state = self.all_states
        gdf_top_cbsas = self.gdf_top_n_cbsas

        gdf_state_cbsa = gdf_top_cbsas.sjoin(gdf_state, lsuffix="STATE", rsuffix="CBSA")

        states_covered = list(gdf_state_cbsa["STATE"].unique())

        return states_covered

    @cached_property
    def bg_data_for_all_covered_states(self) -> gpd.GeoDataFrame:
        """Download date for all block groups in all states that intersect one of top n CBSAs"""
        states_covered = self.states_covered_by_cbsas

        def _info_generator(states: Iterable[str]) -> Generator[str, None, None]:
            for state in states:
                logger.info(f"Downloading data for {state} ({NAMES_FROM_IDS[state]})")
                yield state

        gdf_bg_data = gpd.GeoDataFrame(
            pd.concat(
                ced.download(
                    self.dataset,
                    self.vintage,
                    download_variables=self.download_variables,
                    leaves_of_group=util.GROUP_RACE_ETHNICITY,
                    with_geometry=True,
                    set_to_nan=ALL_SPECIAL_VALUES,
                    state=state,
                    block_group="*",
                ).dropna()
                for state in _info_generator(states_covered)
            )
        )

        return gdf_bg_data

    @cached_property
    def bg_data(self) -> gpd.GeoDataFrame:
        """Download data for all block groups contained in the top n CBSAs."""
        gdf_top_cbsas = self.gdf_top_n_cbsas
        gdf_bg_data = self.bg_data_for_all_covered_states

        logger.info("Joining CBSAs and block group data.")

        gdf_bg_by_cbsa = (
            cem.sjoin_mostly_contains(
                gdf_large_geos=gdf_top_cbsas,
                gdf_small_geos=gdf_bg_data,
                large_suffix="CBSA",
                small_suffix="BG",
            )
            .drop("index_CBSA", axis="columns")
            .reset_index(drop=True)
        )

        return gdf_bg_by_cbsa


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--log",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
        default="WARNING",
    )
    parser.add_argument(
        "-c", "--cbsa", nargs="+", type=str, help="The CBSAs to get data for."
    )
    parser.add_argument(
        "-v", "--vintage", required=True, type=int, help="Year to get data."
    )
    parser.add_argument("-o", "--output", required=True, type=str, help="Output file.")

    args = parser.parse_args()

    level = getattr(logging, args.log)

    logging.basicConfig(level=level)
    logger.setLevel(level)

    output_dir = Path(args.output)

    downloader = Downloader(
        dataset=ACS5,
        vintage=args.vintage,
        top_n_cbsas=[arg.split("/")[1].split(".")[0] for arg in args.cbsa],
    )

    gdf_data = downloader.bg_data

    # Add in fractional values.
    for leaf in ced.variables.group_leaves(
        dataset=ACS5, year=args.vintage, name=util.GROUP_RACE_ETHNICITY
    ) + [
        util.VARIABLE_TOTAL_POP,
        util.VARIABLE_BLACK_ALONE,
        util.VARIABLE_WHITE_ALONE,
        util.VARIABLE_NH_TOTAL,
        util.VARIABLE_NH_BLACK,
        util.VARIABLE_NH_WHITE,
        util.VARIABLE_H_TOTAL,
        util.VARIABLE_H_BLACK,
        util.VARIABLE_H_WHITE,
    ]:
        gdf_data[f"frac_{leaf}"] = gdf_data[leaf] / gdf_data[util.VARIABLE_TOTAL_POP]

    for file_path in args.cbsa:
        output = output_dir / file_path
        output.parent.mkdir(exist_ok=True)
        logger.info(f"Writing to output file {output}")
        gdf_data[
            gdf_data["METROPOLITAN_STATISTICAL_AREA_MICROPOLITAN_STATISTICAL_AREA"]
            == output.stem
        ].to_file(output)
        logger.info("Writing complete.")


if __name__ == "__main__":
    main()
