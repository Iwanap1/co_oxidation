from typing import Literal, List

Metal = Literal[
    'Al', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Sr', 'Y', 'Zr', 'Nb', 'Ru', 'Ag', 'Sn', 'La', 'Ce',
    'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Ho', 'Yb', 'Lu',
    'Hf', 'Bi'
]

METALS = list(Metal.__args__)


class DopantFeaturiser:
    def __init__(self, metals: List[Metal]=METALS) -> None:
        self.metals = metals
        