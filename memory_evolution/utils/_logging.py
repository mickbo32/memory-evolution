import logging
import os
import sys
import time
from typing import Optional

import pandas as pd


# logging settings:
def set_main_logger(
                logger_name: Optional[str] = None,
                logging_dir: str = 'logs',
                file_handler_now: Optional[int] = logging.DEBUG,
                file_handler_all: Optional[int] = logging.DEBUG,
                stderr_handler: Optional[int] = logging.WARNING,
                stdout_handler: Optional[int] = logging.INFO,
        ) -> tuple[str, str]:
    """Set root logger: this function should be run as first line of the main file.
    It returns the ``logging_dir`` and UTCNOW string (the one used as tag in the filename of
    ``file_handler_now`` file).

    If ``logger_name`` is not provided or ``None``, it uses the root logger by default.
    """
    # get utcnow string:
    utcnow = pd.Timestamp.utcnow().strftime('%Y-%m-%d_%H%M%S.%f%z')

    logFormatter = logging.Formatter(
        "%(asctime)s [%(processName)-12s %(process)-7d] [%(threadName)-12s %(thread)-7d] "
        "[%(levelname)-5s] %(module)-15s:  %(message)s")

    os.makedirs(logging_dir, exist_ok=True)

    if logger_name is None:
        rootLogger = logging.getLogger()
    else:
        rootLogger = logging.getLogger(logger_name)

    if file_handler_now is not None:
        fileHandler = logging.FileHandler(os.path.join("logs", f"log_{utcnow}.log"), mode='w')  # default mode='a'
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(file_handler_now)
        rootLogger.addHandler(fileHandler)

    if file_handler_all is not None:
        fileHandler = logging.FileHandler(os.path.join("logs", "log_all.log"), mode='a')  # default mode='a'
        fileHandler.setFormatter(logFormatter)
        fileHandler.setLevel(file_handler_all)
        rootLogger.addHandler(fileHandler)

    if stderr_handler is not None:
        consoleHandler = logging.StreamHandler(sys.stderr)  # sys.stderr default
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(stderr_handler)
        rootLogger.addHandler(consoleHandler)

    if stdout_handler is not None:
        consoleStdoutHandler = logging.StreamHandler(sys.stdout)
        consoleStdoutHandler.setFormatter(logging.Formatter("%(message)s"))  # default, it logs just the message
        consoleStdoutHandler.setLevel(stdout_handler)
        rootLogger.addHandler(consoleStdoutHandler)

    rootLogger.setLevel(logging.NOTSET)  # logging.WARNING default for root, logging.NOTSET default for others.

    return logging_dir, utcnow

