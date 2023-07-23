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

    parser.add_argument(
        "-v", "--vintage", required=True, type=int, help="Year to get data."
    )
    parser.add_argument("--group-hispanic-latino", action="store_true")

    parser.add_argument(
        "-o", "--output-dir", required=True, help="Output directory for plots."
    )
    parser.add_argument("-F", "--output-file-name", help="Output file name override.")

    parser.add_argument("--feature", help="Generate plot only for this feature.")

    # To add highlighting or emphasis (further highlighting of a subset).
    parser.add_argument("--highlight-feature-above", type=float)
    parser.add_argument("--highlight-feature-below", type=float)
    parser.add_argument("--highlight-value-above", type=int)
    parser.add_argument("--highlight-value-below", type=int)

    parser.add_argument("--emphasize-feature-above", type=float)
    parser.add_argument("--emphasize-feature-below", type=float)
    parser.add_argument("--emphasize-value-above", type=int)
    parser.add_argument("--emphasize-value-below", type=int)

    parser.add_argument("input_file", help="Input file, as created by datagen.py")

    args = parser.parse_args()

    logging.info(f"{args.input_file} -> {args.output_dir}")

    highlight_feature_above = args.highlight_feature_above
    highlight_feature_below = args.highlight_feature_below
    highlight_value_above = args.highlight_value_above
    highlight_value_below = args.highlight_value_below

    emphasize_feature_above = args.emphasize_feature_above
    emphasize_feature_below = args.emphasize_feature_below
    emphasize_value_above = args.emphasize_value_above
    emphasize_value_below = args.emphasize_value_below

    do_highlight_feature = (
        highlight_feature_above is not None or highlight_feature_below is not None
    )
    do_highlight_value = (
        highlight_value_above is not None or highlight_value_below is not None
    )
    do_highlight = do_highlight_feature or do_highlight_value

    do_emphasize_feature = (
        emphasize_feature_above is not None or emphasize_feature_below is not None
    )
    do_emphasize_value = (
        emphasize_value_above is not None or emphasize_value_below is not None
    )
    do_emphasize = do_emphasize_feature or do_emphasize_value

    gdf_cbsa_bg = read_data(args.input_file, drop_outliers=True)

    year = args.vintage
    all_variables = ced.variables.all_variables(ACS5, year, util.GROUP_RACE_ETHNICITY)

    dollar_formatter = FuncFormatter(
        lambda d, pos: f"\\${d:,.0f}" if d >= 0 else f"(\\${-d:,.0f})"
    )

    X, _, _ = util.xyw(gdf_cbsa_bg, year, group_lh_together=args.group_hispanic_latino)

    if args.feature is not None:
        features = [args.feature]
    else:
        features = X.columns

    for feature in features:
        if not feature.startswith("frac_"):
            logger.info(f"Skipping feature '{feature}'")
            continue

        variable = feature[5:]  # Remove leading "frac_"

        label = all_variables[all_variables["VARIABLE"] == variable]["LABEL"].iloc[0]

        label = label.replace("Estimate!!Total:!!", "")
        label = label.replace(":!!", "; ")
        label = label.replace(":", "")

        if do_emphasize or do_highlight:
            color = "lightgray"
        else:
            color = "C0"

        ax = gdf_cbsa_bg.plot.scatter(
            feature,
            util.VARIABLE_MEDIAN_VALUE,
            figsize=(12, 8),
            s=2,
            color=color,
        )

        if do_highlight:
            ax, gdf_cbsa_bg = filter_and_plot(
                ax,
                gdf_cbsa_bg,
                feature,
                do_highlight_feature,
                do_highlight_value,
                highlight_feature_above,
                highlight_feature_below,
                highlight_value_above,
                highlight_value_below,
                label,
                "Median Home Value",
                "orange",
                2,
            )

        if do_emphasize:
            ax, gdf_cbsa_bg = filter_and_plot(
                ax,
                gdf_cbsa_bg,
                feature,
                do_emphasize_feature,
                do_emphasize_value,
                emphasize_feature_above,
                emphasize_feature_below,
                emphasize_value_above,
                emphasize_value_below,
                label,
                "Median Home Value",
                "darkgreen",
                8,
            )

        ax.grid()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05 * util.MAX_PRICE, 1.05 * util.MAX_PRICE)

        ax.yaxis.set_major_formatter(dollar_formatter)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

        name = Path(args.output_dir).parent.name.replace("_", " ")
        ax.set_title(f"Median Home Value vs.\nPercentage {label}\nin {name}")
        ax.set_xlabel(label)
        ax.set_ylabel("Median Home Value")

        if do_highlight or do_emphasize:
            for handle in ax.legend().legend_handles:
                handle._sizes = [25]

        if args.output_file_name is not None:
            file_path = Path(args.output_dir) / args.output_file_name
        else:
            filename = label.replace(" ", "-").replace(";", "")
            file_path = Path(args.output_dir) / f"{filename}.png"

        plt.savefig(file_path)


def filter_and_plot(
    ax,
    gdf_cbsa_bg,
    feature,
    do_feature,
    do_value,
    feature_above,
    feature_below,
    value_above,
    value_below,
    feature_label,
    value_label,
    color,
    size,
):
    if feature_above is not None:
        gdf_cbsa_bg = gdf_cbsa_bg[gdf_cbsa_bg[feature] >= feature_above]
    if feature_below is not None:
        gdf_cbsa_bg = gdf_cbsa_bg[gdf_cbsa_bg[feature] < feature_below]
    if value_above is not None:
        gdf_cbsa_bg = gdf_cbsa_bg[
            gdf_cbsa_bg[util.VARIABLE_MEDIAN_VALUE] >= value_above
        ]
    if value_below is not None:
        gdf_cbsa_bg = gdf_cbsa_bg[gdf_cbsa_bg[util.VARIABLE_MEDIAN_VALUE] < value_below]

    def _range_label(low, high, var, as_dollar=False):
        if low is not None:
            if as_dollar:
                range_label = f"${low:,.0f} ≤ {var}"
            else:
                range_label = f"{low * 100:.0f}% ≤ {var}"
        else:
            range_label = var
        if high is not None:
            if as_dollar:
                range_label = f"${range_label} < {high:,.0f}"
            else:
                range_label = f"{range_label} < {high * 100:.0f}%"
        return range_label

    if do_feature:
        label_feature = _range_label(feature_above, feature_below, feature_label)
    else:
        label_feature = None
    if do_value:
        label_value = _range_label(
            value_above, value_below, value_label, as_dollar=True
        )
        if label_feature is not None:
            label_value = f"{label_feature} and {label_value}"
    else:
        label_value = None
    plot_label = "; ".join(
        sublabel for sublabel in [label_feature, label_value] if sublabel is not None
    )

    plot_label = f"{plot_label} (n = {len(gdf_cbsa_bg.index):,.0f})"

    ax = gdf_cbsa_bg.plot.scatter(
        feature,
        util.VARIABLE_MEDIAN_VALUE,
        figsize=(12, 8),
        s=size,
        label=plot_label,
        color=color,
        ax=ax,
    )
    return ax, gdf_cbsa_bg


if __name__ == "__main__":
    main()
