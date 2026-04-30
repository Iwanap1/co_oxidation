from typing import List, Dict, Optional
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset


class GeneralLightOffMLP(nn.Module):
    def __init__(self, material_input_dim: int, model_config: Dict) -> None:
        super().__init__()
        osc_cfg = model_config.get("osc_net")
        tpr_cfg = model_config.get("tpr_net")
        self.contrastive_learning = tpr_cfg or osc_cfg
        self.hybridise_whsv = model_config.get("hybridise_whsv", False)
        self.hybridise_pressure = model_config.get("hybridise_pressure", False)

        if self.hybridise_pressure and not self.hybridise_whsv:
            raise NotImplementedError("Cannot hybridise Pressure and not WHSV, must be both, neither or only WHSV")

        conv_net_input_dim = 0 if model_config["conversion_net"].get("include_material_features", False) else material_input_dim
        if osc_cfg is not None:
            self.osc_net = self._make_mlp(material_input_dim, osc_cfg)
            conv_net_input_dim += 1
        
        if tpr_cfg is not None:
            self.tpr_net = self._make_mlp(material_input_dim, tpr_cfg)
            conv_net_input_dim += 1

        self.conversion_net = self._make_mlp(conv_net_input_dim, model_config["conversion_net"])


    def forward(
        self,
        conversion_feature_tensor: torch.Tensor,
        osc_feature_tensor: Optional[torch.Tensor]=None,
        tpr_feature_tensor: Optional[torch.Tensor]=None,
        whsv_tensor: Optional[torch.Tensor]=None,
        p_co_tensor: Optional[torch.Tensor]=None,
        p_o2_tensor: Optional[torch.Tensor]=None
    ):
        
        self._check_input_validity_for_architecture(osc_feature_tensor, tpr_feature_tensor, whsv_tensor, p_co_tensor, p_o2_tensor)
        return NotImplemented


    def _make_mlp(self, input_dim: int, net_config: Dict) -> nn.Sequential:
        if net_config["activation"] is not None:
            act = getattr(nn, net_config["activation"])()

        layers = []
        prev = input_dim
        for h in net_config["hidden_dim"]:
            layers += [nn.Linear(prev, h), act]
            prev = h
        layers += [nn.Linear(prev, net_config["output_dim"])]

        return nn.Sequential(*layers)
    
    
    def _check_input_validity_for_architecture(
        self,
        osc_feature_tensor: Optional[torch.Tensor]=None,
        tpr_feature_tensor: Optional[torch.Tensor]=None,
        whsv_tensor: Optional[torch.Tensor]=None,
        p_co_tensor: Optional[torch.Tensor]=None,
        p_o2_tensor: Optional[torch.Tensor]=None
    ):

        if self.osc_net is not None:
            if osc_feature_tensor is None:
                raise ValueError("osc_features must be provided when osc_net is enabled")

        if self.tpr_net is not None:
            if tpr_feature_tensor is None:
                raise ValueError("tpr_features must be provided when tpr_net is enabled")

        if self.hybridise_whsv:
            if whsv_tensor is None:
                raise ValueError("whsv must be provided when hybridise_whsv=True")

        if self.hybridise_pressure:
            if p_co_tensor is None or p_o2_tensor is None:
                raise ValueError("p_co and p_o2 must be provided when hybridise_pressures=True")