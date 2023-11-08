"""
File: __init__.py
Author: Chuncheng Zhang
Date: 2023-11-07
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-11-07 ------------------------
# Requirements and constants

from rich import print, inspect

from loguru import logger as LOGGER
from tqdm.auto import tqdm

from pathlib import Path
from datetime import datetime

# %% ---- 2023-11-07 ------------------------
# Function and class


class ProjectPath(dict):
    root = Path(__file__).parent.parent
    log = root.joinpath('log')
    data = root.joinpath('data')

    def __init__(self):
        super(dict, self).__init__()
        self.log.mkdir(exist_ok=True)
        assert self.data.is_dir(), f'Data directory not found: {self.data}'


PPATH = ProjectPath()

LOGGER.add(PPATH.log.joinpath(
    f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"))


# %% ---- 2023-11-07 ------------------------
# Play ground


# %% ---- 2023-11-07 ------------------------
# Pending


# %% ---- 2023-11-07 ------------------------
# Pending
