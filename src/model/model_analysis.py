from typing import List
from .model import LightOffModel
from torch.utils.data import TensorDataset


class ModelAnalyser:
    def __init__(self):
        pass

    def analyse_model(self, model: LightOffModel, datasets: List[TensorDataset]):
        return