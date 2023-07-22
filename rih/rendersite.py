import logging

import geopandas as gpd
import pandas as pd
import jinja2
from pathlib import Path

import rih.util as util
from rih.loggingargparser import LoggingArgumentParser

logger = logging.getLogger(__name__)


def main():
    parser = LoggingArgumentParser(logger)

    parser.add_argument(
        "-v", "--vintage", required=True, type=int, help="Year to get data."
    )
    parser.add_argument(
        "-o", "--output-file", required=True, help="Output file for results."
    )
    parser.add_argument(
        "-t", "--top-n-file", required=True, help="Top n list file we should render into template file."
    )
    parser.add_argument(
        "template_file",  help="Template file."
    )

    args = parser.parse_args()

    logger.info(f"Reading template from {args.template_file}")
    logger.info(f"Reading top n data from {args.top_n_file}")

    with open(args.top_n_file, 'r') as f:
        top_n_list = f.readlines()

    top_n_dict = {
        cbsa.split('/')[0].replace('_', ' '): f"./images/impact_charts/{cbsa.replace('.geojson', '/')}"
        for cbsa in sorted(top_n_list)
    }

    template_args = dict(
        vintage=args.vintage,
        top_n=top_n_dict,
    )

    searchpath = Path(__file__).absolute().parent.parent

    template_loader = jinja2.FileSystemLoader(searchpath)
    template_env = jinja2.Environment(loader=template_loader, autoescape=True)
    template = template_env.get_template(args.template_file)

    rendered_text = template.render(template_args)

    with open(args.output_file, 'w') as f:
        f.write(rendered_text)


if __name__ == "__main__":
    main()
