from argparse import ArgumentParser, Namespace
import logging
from typing import Optional, Sequence


class LoggingArgumentParser(ArgumentParser):
    """An arg parser that accepts --log and sets up logging accordingly."""

    def __init__(self, logger: logging.Logger, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._logger = logger

        self.add_argument(
            "--log",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level.",
            default="WARNING",
        )

    def parse_args(self, args: Optional[Sequence[str]] = None) -> Namespace:
        args = super().parse_args(args)

        level = getattr(logging, args.log)

        logging.basicConfig(level=level)
        self._logger.setLevel(level)

        return args
