import configparser
from pathlib import Path


def load_config(file_path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(Path(file_path))
    return config
