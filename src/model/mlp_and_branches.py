from typing import Dict, Optional

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, cfg: Dict):
        super().__init__()

        layers = []
        prev = input_dim

        activation_name = cfg.get("activation")
        activation_cls = getattr(nn, activation_name) if activation_name else None

        dropout = cfg.get("dropout", 0.0)

        for h in cfg.get("hidden_dim", []):
            layers.append(nn.Linear(prev, h))

            if activation_cls is not None:
                layers.append(activation_cls())

            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev = h

        layers.append(nn.Linear(prev, cfg["output_dim"]))

        self.net = nn.Sequential(*layers)
        self.output_dim = cfg["output_dim"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Branch(nn.Module):
    """
    Generic auxiliary branch.

    Example branches:
        h2_tpr: features -> z_tpr -> temp
        osc:    features -> z_osc -> OSC target
        o2_tpd: features -> z_tpd -> TPD target(s)
        xps:    features -> z_xps -> XPS target(s)

    The latent z is used in the conversion net.
    The head predicts the branch-specific target.
    """

    def __init__(
        self,
        name: str,
        input_dim: int,
        target_dim: int,
        cfg: Dict,
        direct_input_dim: int = 0,
    ):
        super().__init__()

        self.name = name
        self.cfg = cfg
        self.direct_input_dim = direct_input_dim

        self.encoder = MLP(input_dim=input_dim, cfg=cfg)

        self.head = nn.Linear(
            self.encoder.output_dim + direct_input_dim,
            target_dim,
        )

    @property
    def latent_dim(self) -> int:
        return self.encoder.output_dim

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.encoder(features)

    def predict(
        self,
        features: torch.Tensor,
        direct_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z = self.encode(features)

        if self.direct_input_dim > 0:
            if direct_inputs is None:
                raise ValueError(
                    f"{self.name} branch requires direct_inputs "
                    f"with dim={self.direct_input_dim}."
                )
            z = torch.cat([z, direct_inputs], dim=-1)

        return self.head(z)
    

