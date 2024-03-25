import os
from pathlib import Path

from utils import create_directory_if_missing

SPAN = 0.4
LOWER_QUANTILE = 0.05
HIGHER_QUANTILE = 0.95

DATA_PATH = Path("./data")
create_directory_if_missing(DATA_PATH)

PLOTS_PATH = Path("./plots")
create_directory_if_missing(PLOTS_PATH)