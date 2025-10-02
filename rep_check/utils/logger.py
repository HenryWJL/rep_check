import logging
from pathlib import Path
from typing import Union


def create_logger(logging_dir: Union[str, Path, None] = None) -> logging.Logger:
    if logging_dir is not None:
        if isinstance(logging_dir, str):
            logging_dir = Path(logging_dir).expanduser().absolute()
        logging_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(str(logging_dir.joinpath("train.log")))
            ]
        )
    logger = logging.getLogger(__name__)
    return logger