import contextlib
import datetime
import logging
import sys
from typing import IO

logger = logging.getLogger(__name__)
logger.propagate = False


class TimeFilter(logging.Filter):

    def filter(self, record):
        if record.levelno == logging.DEBUG + 1:
            record.relative = ""
            return True

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


class MultiLineFormatter(logging.Formatter):
    """Multi-line formatter."""

    def get_header_length(self, record):
        """Get the header length of a given record."""
        rec = logging.LogRecord(
            name=record.name,
            level=record.levelno,
            pathname=record.pathname,
            lineno=record.lineno,
            msg="",
            args=(),
            exc_info=None,
        )
        rec.relative = record.relative
        return len(super().format(rec))

    def format(self, record):
        """Format a record with added indentation."""
        indent = " " * self.get_header_length(record)
        head, *trailing = super().format(record).splitlines(True)
        return head + "".join(indent + line for line in trailing)


def filter_maker(level):
    def filter(record):
        return record.levelno < level

    return filter


def clear_logging():
    logger.handlers.clear()


def setup_logging(level=logging.DEBUG, path=None, output_to_stdout=True):
    global logger

    logger.setLevel(level)
    logger.handlers.clear()
    logging.addLevelName(logging.DEBUG + 1, "PRINT")

    if path:
        formatter_file = MultiLineFormatter(
            "%(levelname)s:%(asctime)s - (%(relative)ss) - %(filename)s:%(lineno)d - %(message)s"
        )

        file_time_filter = TimeFilter()
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter_file)
        file_handler.addFilter(file_time_filter)
        logger.addHandler(file_handler)

    if output_to_stdout:
        time_filter = TimeFilter()

        formatter_warn = MultiLineFormatter(
            "\033[1;33m%(levelname)s:%(processName)s:%(asctime)s - (%(relative)ss) - %(filename)s:%(lineno)d - %(message)s \033[0m"
        )
        formatter_info = MultiLineFormatter(
            "\033[1;34m%(levelname)s:%(processName)s:%(asctime)s - (%(relative)ss) - %(filename)s:%(lineno)d - %(message)s \033[0m"
        )
        formatter_debug = MultiLineFormatter(
            "\033[37m%(levelname)s:%(processName)s:%(asctime)s - (%(relative)ss) - %(filename)s:%(lineno)d - %(message)s \033[0m"
        )

        s_warn = logging.StreamHandler(sys.__stdout__)
        s_warn.setLevel(logging.WARN)
        s_warn.addFilter(time_filter)
        s_warn.setFormatter(formatter_warn)

        s_info = logging.StreamHandler(sys.__stdout__)
        s_info.setLevel(logging.INFO)
        s_info.addFilter(time_filter)
        s_info.addFilter(filter_maker(logging.WARN))
        s_info.setFormatter(formatter_info)

        s_debug = logging.StreamHandler(sys.__stdout__)
        s_debug.setLevel(logging.DEBUG)
        s_debug.addFilter(time_filter)
        s_debug.addFilter(filter_maker(logging.INFO))
        s_debug.setFormatter(formatter_debug)

        logger.addHandler(s_warn)
        logger.addHandler(s_info)
        logger.addHandler(s_debug)


def handle_exception(exc_type, exc_value, exc_traceback):
    if not issubclass(exc_type, KeyboardInterrupt):
        logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.__excepthook__(exc_type, exc_value, exc_traceback)


class OutputLogger(IO[str]):
    def __init__(self, level=logging.DEBUG + 1):
        self.logger = logger
        self.level = level
        self._redirector = contextlib.redirect_stdout(self)
        self._redirector_err = contextlib.redirect_stderr(self)

    def write(self, msg: str):
        if msg and not msg.isspace():
            self.logger.log(self.level, msg.strip())

    def flush(self):
        pass

    def __enter__(self):
        self._redirector.__enter__()
        self._redirector_err.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._redirector.__exit__(exc_type, exc_value, traceback)
        self._redirector_err.__exit__(exc_type, exc_value, traceback)
