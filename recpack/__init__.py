import logging
import sys

logger = logging.getLogger("recpack")
logger.setLevel(logging.INFO)


if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(module)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
