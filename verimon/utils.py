import datetime
import logging
import sys
from logging import Logger

logger: Logger = logging.getLogger(__name__)
logger.propagate = False


def get_pos(labels: list[str]):
    pos = [int(l[5:-1]) for l in labels if len(l) > 5 and l.startswith("[pos")]
    if pos:
        return pos[0]
    else:
        return None


class TimeFilter(logging.Filter):

    def filter(self, record):
        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        delta = datetime.datetime.fromtimestamp(
            record.relativeCreated / 1000.0
        ) - datetime.datetime.fromtimestamp(last / 1000.0)

        record.relative = "{0:.2f}".format(
            delta.seconds + delta.microseconds / 1000000.0
        )

        self.last = record.relativeCreated
        return True


def filter_maker(level):
    def filter(record):
        return record.levelno < level

    return filter


def setup_logging(level=logging.DEBUG):
    global logger
    logger.setLevel(level)

    formatter_info = logging.Formatter(
        "\033[1;34m%(levelname)s:%(asctime)s - (%(relative)ss) - %(filename)s - %(message)s \033[0m"
    )
    formatter_debug = logging.Formatter(
        "\033[1;37m%(levelname)s:%(asctime)s - (%(relative)ss) - %(filename)s - %(message)s \033[0m"
    )

    time_filter = TimeFilter()

    s_info = logging.StreamHandler(sys.stdout)
    s_info.setLevel(logging.INFO)
    s_info.addFilter(time_filter)
    s_info.setFormatter(formatter_info)

    s_debug = logging.StreamHandler(sys.stdout)
    s_debug.setLevel(logging.DEBUG)
    s_debug.addFilter(time_filter)
    s_debug.addFilter(filter_maker(logging.INFO))
    s_debug.setFormatter(formatter_debug)

    logger.handlers.clear()
    logger.addHandler(s_info)
    logger.addHandler(s_debug)
