from typing import List
import torch.nn as nn
import torch


class HybridisedWHSV(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: List[int]=[16, 16], output_dim=1, activation="SiLU") -> None:
        super().__init__()

        act = getattr(nn, activation)()

        layers = []
        prev = input_dim
        for h in hidden_dim:
            layers += [nn.Linear(prev, h), act]
            prev = h
        layers += [nn.Linear(prev, output_dim)]

        self.net = nn.Sequential(*layers)


    def forward(self, feature_tensor: torch.Tensor, whsv_tensor: torch.Tensor) -> torch.Tensor:
        return NotImplemented
