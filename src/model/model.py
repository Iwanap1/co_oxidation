from typing import Dict, Optional

import torch
import torch.nn as nn
from .mlp_and_branches import MLP, Branch

class LightOffModel(nn.Module):
    """
    Generic multi-branch light-off model.

    Main conversion input:
        conversion_features
        reaction_inputs
        encoded auxiliary branch latents

    Auxiliary branches:
        configured dynamically from model_config.
    """

    BRANCH_SPECS = {
        "osc": {
            "config_key": "osc_net",
            "feature_tensor": "osc_features",
            "direct_tensor": "osc_direct_inputs",
            "input_dim_key": "osc",
            "target_dim_key": "osc_target",
            "direct_dim_key": "osc_direct_inputs",
        },
        "tpr": {
            "config_key": "tpr_net",
            "feature_tensor": "tpr_features",
            "direct_tensor": "tpr_direct_inputs",
            "input_dim_key": "tpr",
            "target_dim_key": "tpr_target",
            "direct_dim_key": "tpr_direct_inputs",
        },
        "tpd": {
            "config_key": "tpd_net",
            "feature_tensor": "tpd_features",
            "direct_tensor": "tpd_direct_inputs",
            "input_dim_key": "tpd",
            "target_dim_key": "tpd_target",
            "direct_dim_key": "tpd_direct_inputs",
        },
        "xps": {
            "config_key": "xps_net",
            "feature_tensor": "xps_features",
            "direct_tensor": "xps_direct_inputs",
            "input_dim_key": "xps",
            "target_dim_key": "xps_target",
            "direct_dim_key": "xps_direct_inputs",
        }
    }

    def __init__(
        self,
        input_dims: Dict[str, int],
        model_config: Dict,
    ):
        super().__init__()

        self.input_dims = input_dims
        self.model_config = model_config

        self.hybridise_whsv = model_config.get("hybridise_whsv", False)
        self.hybridise_pressures = model_config.get("hybridise_pressures", False)

        if self.hybridise_pressures and not self.hybridise_whsv:
            raise ValueError("Pressure hybridisation requires WHSV hybridisation.")

        conv_cfg = model_config["conversion_net"]

        self.include_conversion_features = conv_cfg.get(
            "include_material_features",
            True,
        )
        self.input_reaction_cols = conv_cfg.get("input_reaction_cols", [])

        conv_input_dim = 0

        if self.input_reaction_cols:
            conv_input_dim += input_dims.get("reaction_inputs", 0)

        if self.include_conversion_features:
            conv_input_dim += input_dims["conversion"]

        self.branches = nn.ModuleDict()

        for branch_name, spec in self.BRANCH_SPECS.items():
            branch_cfg = model_config.get(spec["config_key"])

            if branch_cfg is None:
                continue

            branch = Branch(
                name=branch_name,
                input_dim=input_dims[spec["input_dim_key"]],
                target_dim=input_dims[spec["target_dim_key"]],
                direct_input_dim=input_dims.get(spec["direct_dim_key"], 0),
                cfg=branch_cfg,
            )

            self.branches[branch_name] = branch
            conv_input_dim += branch.latent_dim

        conv_cfg = dict(conv_cfg)
        self.conversion_net = MLP(
            input_dim=conv_input_dim,
            cfg=conv_cfg,
        )

    @property
    def active_branches(self):
        return list(self.branches.keys())

    def forward(
        self,
        conversion_features: Optional[torch.Tensor] = None,
        reaction_inputs: Optional[torch.Tensor] = None,
        whsv: Optional[torch.Tensor] = None,
        p_co: Optional[torch.Tensor] = None,
        p_o2: Optional[torch.Tensor] = None,
        **branch_feature_tensors,
    ) -> torch.Tensor:
        """
        branch_feature_tensors can contain:
            osc_features
            tpr_features
            tpd_features
            xps_features later
        """

        z = self._conversion_latent(
            conversion_features=conversion_features,
            reaction_inputs=reaction_inputs,
            branch_feature_tensors=branch_feature_tensors,
        )

        if not self.hybridise_pressures and not self.hybridise_whsv:
            return torch.sigmoid(z)

        if self.hybridise_whsv and not self.hybridise_pressures:
            if whsv is None:
                raise ValueError("whsv is required when hybridise_whsv=True.")

            k_app = torch.exp(z)
            tau = 1.0 / (whsv + 1e-8)
            return 1.0 - torch.exp(-k_app * tau)

        if self.hybridise_whsv and self.hybridise_pressures:
            if whsv is None or p_co is None or p_o2 is None:
                raise ValueError(
                    "whsv, p_co, and p_o2 are required when "
                    "hybridise_pressures=True."
                )

            k_app = torch.exp(z)
            tau = 1.0 / (whsv + 1e-8)

            xmax = torch.clamp(
                2.0 * p_o2 / (p_co + 1e-8),
                min=0.0,
                max=1.0,
            )

            return xmax * (1.0 - torch.exp(-k_app * tau))

        raise RuntimeError("Unhandled model hybridisation configuration.")

    def _conversion_latent(
        self,
        conversion_features: Optional[torch.Tensor],
        reaction_inputs: Optional[torch.Tensor],
        branch_feature_tensors: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        parts = []

        if self.input_reaction_cols:
            if reaction_inputs is None:
                raise ValueError(
                    "reaction_inputs required by conversion_net.input_reaction_cols."
                )
            parts.append(reaction_inputs)

        if self.include_conversion_features:
            if conversion_features is None:
                raise ValueError("conversion_features required.")
            parts.append(conversion_features)

        for branch_name, branch in self.branches.items():
            spec = self.BRANCH_SPECS[branch_name]
            tensor_name = spec["feature_tensor"]

            features = branch_feature_tensors.get(tensor_name)

            if features is None:
                raise ValueError(
                    f"{tensor_name} required because {branch_name} branch is enabled."
                )

            parts.append(branch.encode(features))

        if not parts:
            raise ValueError("No inputs to conversion_net.")

        x = torch.cat(parts, dim=-1)
        return self.conversion_net(x)

    def predict_branch(
        self,
        branch_name: str,
        features: torch.Tensor,
        direct_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if branch_name not in self.branches:
            raise ValueError(f"Branch '{branch_name}' is not enabled.")

        return self.branches[branch_name].predict(
            features=features,
            direct_inputs=direct_inputs,
        )