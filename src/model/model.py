from typing import List, Dict, Optional
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset
import torch.nn.functional as F


class LightOffModel(nn.Module):
    def __init__(
        self,
        input_dims: Dict[str, int],
        model_config: Dict,
    ) -> None:
        super().__init__()

        self.model_config = model_config
        self.hybridise_whsv = model_config.get("hybridise_whsv", False)
        self.hybridise_pressures = model_config.get("hybridise_pressures", False)

        if self.hybridise_pressures and not self.hybridise_whsv:
            raise ValueError("Pressure hybridisation requires WHSV hybridisation.")

        self.osc_net = None
        self.tpr_net = None

        conv_cfg = model_config["conversion_net"]
        osc_cfg = model_config.get("osc_net")
        tpr_cfg = model_config.get("tpr_net")
        self.condition_tpr_with_ramp_rate = tpr_cfg is not None and tpr_cfg.get("condition_tpr_with_ramp_rate", False)
        self.include_conversion_features = conv_cfg.get("include_material_features", True)

        conv_input_dim = 0

        if self.include_conversion_features:
            conv_input_dim += input_dims["conversion"]

        if osc_cfg is not None:
            self.osc_net = self._make_mlp(input_dims["osc"], osc_cfg)
            self.osc_head = nn.Linear(osc_cfg["output_dim"], 1)
            conv_input_dim += osc_cfg["output_dim"]

        if tpr_cfg is not None:
            self.tpr_net = self._make_mlp(input_dims["tpr"], tpr_cfg)

            tpr_head_input_dim = tpr_cfg["output_dim"]
            if self.condition_tpr_with_ramp_rate:
                tpr_head_input_dim += 1

            self.tpr_head = nn.Linear(tpr_head_input_dim, 1)
            conv_input_dim += tpr_cfg["output_dim"]

        self.conversion_net = self._make_mlp(conv_input_dim, conv_cfg)

    def forward(
        self,
        conversion_features: Optional[torch.Tensor] = None,
        osc_features: Optional[torch.Tensor] = None,
        tpr_features: Optional[torch.Tensor] = None,
        whsv: Optional[torch.Tensor] = None,
        p_co: Optional[torch.Tensor] = None,
        p_o2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        z = self._nn_section(conversion_features, osc_features, tpr_features)
        # Black-box only
        if not self.hybridise_pressures and not self.hybridise_whsv:
            return torch.sigmoid(z)
        
        # Hybridised WHSV branch
        if self.hybridise_whsv and not self.hybridise_pressures:
            k_app = F.softplus(z)
            tau = 1.0 / (whsv + 1e-8)
            x = 1.0 - torch.exp(-k_app * tau) # Simple first order PFR model
            return x
        
        # Hybridised WHSV + pressure branch
        if self.hybridise_whsv and self.hybridise_pressures:
            if whsv is None or p_co is None or p_o2 is None:
                raise ValueError(
                    "whsv, p_co, and p_o2 are required when hybridise_pressures=True."
                )

            k_app = F.softplus(z)
            tau = 1.0 / (whsv + 1e-8)

            xmax = torch.clamp(
                2.0 * p_o2 / (p_co + 1e-8),
                min=0.0,
                max=1.0,
            )

            return xmax * (1.0 - torch.exp(-k_app * tau))

        raise RuntimeError("Unhandled model hybridisation configuration.")

    def _nn_section(self, conversion_features, osc_features, tpr_features):
        parts = []
        if self.include_conversion_features:
            if conversion_features is None:
                raise ValueError("conversion_features required.")
            parts.append(conversion_features)

        if self.osc_net is not None:
            if osc_features is None:
                raise ValueError("osc_features required when osc_net is enabled.")
            parts.append(self.encode_osc(osc_features))

        if self.tpr_net is not None:
            if tpr_features is None:
                raise ValueError("tpr_features required when tpr_net is enabled.")
            parts.append(self.encode_tpr(tpr_features))

        if not parts:
            raise ValueError("No inputs to conversion_net.")

        x = torch.cat(parts, dim=-1)
        return self.conversion_net(x)

    def _make_mlp(self, input_dim: int, net_config: Dict) -> nn.Sequential:
        layers = []
        prev = input_dim

        activation_name = net_config.get("activation")
        activation_cls = getattr(nn, activation_name) if activation_name else None

        for h in net_config.get("hidden_dim", []):
            layers.append(nn.Linear(prev, h))
            if activation_cls is not None:
                layers.append(activation_cls())
            prev = h

        layers.append(nn.Linear(prev, net_config["output_dim"]))
        return nn.Sequential(*layers)
    

    def predict_tpr(
        self,
        tpr_features: torch.Tensor,
        ramp_rate: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        z_tpr = self.encode_tpr(tpr_features)

        if self.condition_tpr_with_ramp_rate:
            if ramp_rate is None:
                raise ValueError("ramp_rate required when hybridise_ramp_rate=True.")
            z_tpr = torch.cat([z_tpr, ramp_rate], dim=-1)

        return self.tpr_head(z_tpr)
    
    def predict_osc(self, osc_features: torch.Tensor) -> torch.Tensor:
        z_osc = self.encode_osc(osc_features)
        return self.osc_head(z_osc)
    
    def encode_tpr(self, tpr_features: torch.Tensor) -> torch.Tensor:
        if self.tpr_net is None:
            raise ValueError("tpr_net is not enabled.")
        return self.tpr_net(tpr_features)

    def encode_osc(self, osc_features: torch.Tensor) -> torch.Tensor:
        if self.osc_net is None:
            raise ValueError("osc_net is not enabled.")
        return self.osc_net(osc_features)