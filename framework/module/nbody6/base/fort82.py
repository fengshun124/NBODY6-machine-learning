from pathlib import Path
from typing import Union

from module.nbody6.base.base import NBody6OutputFile


class NBody6Fort82(NBody6OutputFile):
    def __init__(self, filepath: Union[str, Path]) -> None:
        super().__init__(
            filepath,
             {
            "header_prefix": "## BEGIN",
            "footer_prefix": "## END",
            "header_schema": {"time": (1, float)},
            "row_schema": {
                "name1": (0, int),
                "name2": (1, int),
                "x": (5, float),
                "y": (6, float),
                "z": (7, float),
                "mass1": (11, float),
                "mass2": (12, float),
                "zlum1": (13, float),
                "zlum2": (14, float),
                "rad1": (15, float),
                "rad2": (16, float),
                "tempe1": (17, float),
                "tempe2": (18, float),
            },
        },
        )
