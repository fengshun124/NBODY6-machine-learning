from pathlib import Path
from typing import Union

from module.nbody6.base.base import NBody6OutputFile


class NBody6OUT9(NBody6OutputFile):
    def __init__(self, filepath: Union[str, Path]) -> None:
        super().__init__(
            filepath,
            {
                "header_prefix": "#",
                "header_schema": {
                    "time": (1, float),
                    "npairs": (2, int),
                },
                "row_schema": {
                    "ecc": (3, float),
                    "semi": (4, float),
                    "p": (5, float),
                    "mass1": (6, float),
                    "mass2": (7, float),
                    "name1": (8, int),
                    "name2": (9, int),
                    "kstar1": (10, int),
                    "kstar2": (11, int),
                    "cmName": (13, int),
                },
            },
        )
