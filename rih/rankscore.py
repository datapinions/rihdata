import logging
from pathlib import Path

import pandas as pd
import yaml

from rih.loggingargparser import LoggingArgumentParser

logger = logging.getLogger(__name__)


def main():

    parser = LoggingArgumentParser(logger)

    parser.add_argument("-o", "--output-file", required=True, help="Output file for results.")
    parser.add_argument("input_file", nargs="+", help="Input param yaml files, as created by treegress.py")

    args = parser.parse_args()

    rows = []

    for file_name in args.input_file:
        file_path = Path(file_name)
        cbsa = file_path.stem.split('.')[0]
        name = file_path.parent.name

        with open(file_path) as f:
            contents = yaml.full_load(f)
            mape = contents['target']
            row = dict(
                CBSA=cbsa,
                NAME=name,
                XGB_MAPE=mape,
            )

            for k, v in contents['params'].items():
                row[k.upper()] = v

        linreg_file_name = file_name.replace(".params.", ".linreg.")
        linreg_file_path = Path(linreg_file_name)
        if linreg_file_path.exists():
            with open(linreg_file_path) as f:
                contents = yaml.full_load(f)
                row['LINREG_FULL_MAPE'] = contents['full']['mape_score']
                row['LINREG_1_MAPE'] = contents['one']['mape_score']
                row['LINREG_FULL_LOG_MAPE'] = contents['full_log']['mape_score']
                row['LINREG_1_LOG_MAPE'] = contents['one_log']['mape_score']
                row['MEAN_PRED_MAPE'] = contents['mean']['mape_score']
                row['DELTA_MAPE'] = row['XGB_MAPE'] - row['LINREG_FULL_MAPE']
                row['DELTA_LOG_MAPE'] = row['XGB_MAPE'] - row['LINREG_FULL_LOG_MAPE']
        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(by='XGB_MAPE', ascending=False)

    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()