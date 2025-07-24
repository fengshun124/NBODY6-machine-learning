from pathlib import Path
from typing import Union

from module.nbody6.file.base import NBody6OutputFile


class NBody6Fort83(NBody6OutputFile):
    def __init__(self, filepath: Union[str, Path]) -> None:
        super().__init__(
            filepath,
            {
                "header_prefix": "## BEGIN",
                "footer_prefix": "## END",
                "header_schema": {"time": (1, float)},
                "row_schema": {
                    "name": (0, int),
                    "x": (2, float),
                    "y": (3, float),
                    "z": (4, float),
                    "mass": (5, float),
                    "zlum": (6, float),
                    "rad": (7, float),
                    "tempe": (8, float),
                },
            },
        )
