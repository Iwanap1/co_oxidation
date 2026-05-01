from typing import Dict, Optional
import torch
import torch.nn as nn


class CustomLoss:
    def __init__(self, loss_cfg: Dict):
        self.cfg = loss_cfg

        self.conversion_loss_fn = self._make_loss(self.cfg.get("conversion", {"name": "MSELoss"}))
        self.tpr_loss_fn = self._make_loss(self.cfg.get("tpr", {"name": "MSELoss"}))
        self.osc_loss_fn = self._make_loss(self.cfg.get("osc", {"name": "MSELoss"}))

        self.w_conversion = self.cfg.get("conversion", {}).get("weight", 1.0)
        self.w_tpr = self.cfg.get("tpr", {}).get("weight", 0.0)
        self.w_osc = self.cfg.get("osc", {}).get("weight", 0.0)

    def _make_loss(self, cfg: Dict):
        name = cfg.get("name", "MSELoss")

        if not hasattr(nn, name):
            raise ValueError(f"torch.nn has no loss function '{name}'.")

        kwargs = {k: v for k, v in cfg.items() if k not in ["name", "weight"]}
        return getattr(nn, name)(**kwargs)


    def __call__(
        self,
        predictions: Dict[str, torch.Tensor],
        data: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        device = next(iter(predictions.values())).device
        total = torch.tensor(0.0, device=device)

        if "conversion" in predictions:
            y = data["reactions"]["target"]
            loss = self.conversion_loss_fn(predictions["conversion"], y)
            losses["conversion"] = loss
            total = total + self.w_conversion * loss

        if "tpr" in predictions:
            y = data["h2_tpr"]["target"]
            loss = self.tpr_loss_fn(predictions["tpr"], y)
            losses["tpr"] = loss
            total = total + self.w_tpr * loss

        if "osc" in predictions:
            y = data["osc"]["target"]
            loss = self.osc_loss_fn(predictions["osc"], y)
            losses["osc"] = loss
            total = total + self.w_osc * loss

        losses["total"] = total
        return losses