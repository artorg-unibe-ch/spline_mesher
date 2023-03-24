import logging
import coloredlogs

LEVEL = logging.INFO
coloredlogs.install(level=LEVEL)
LOGGING_NAME = "SIMONE"


def logging_setup():
    logger = logging.getLogger(LOGGING_NAME)
    logger.setLevel(LEVEL)
    console_handler = logging.StreamHandler()
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    return logger
