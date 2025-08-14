from pathlib import Path
from typing import Union

from module.nbody6.file.base import NBody6FileParserBase


class NBody6OUT9(NBody6FileParserBase):
    def __init__(self, filepath: Union[str, Path]) -> None:
        super().__init__(
            filepath,
            {
                "header_prefix": "#",
                "header_length": 3,
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
