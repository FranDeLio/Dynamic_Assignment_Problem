import os
from pathlib import Path

DATA_PATH = Path("./data")
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

PLOTS_PATH = Path("./plots")
if not os.path.exists(PLOTS_PATH):
    os.makedirs(PLOTS_PATH)