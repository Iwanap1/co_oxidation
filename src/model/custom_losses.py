from typing import Dict, Optional
import torch
import torch.nn as nn


class CustomLoss:
    def __init__(self, loss_cfg: Dict):
        self.cfg = loss_cfg

        self.conversion_loss_fn = self._make_loss(self.cfg.get("conversion", {"name": "MSELoss"}))
        self.tpr_loss_fn = self._make_loss(self.cfg.get("tpr", {"name": "MSELoss"}))
        self.osc_loss_fn = self._make_loss(self.cfg.get("osc", {"name": "MSELoss"}))
        self.tpd_loss_fn = self._make_loss(self.cfg.get("tpd", {"name": "MSELoss"}))

        self.base_weights = {
            "conversion": self.cfg.get("conversion", {}).get("weight", 1.0),
            "tpr": self.cfg.get("tpr", {}).get("weight", 0.0),
            "osc": self.cfg.get("osc", {}).get("weight", 0.0),
            "tpd": self.cfg.get("tpd", {}).get("weight", 0.0),
        }

        self.decays = {
            "conversion": self.cfg.get("conversion", {}).get("decay", 1.0),
            "tpr": self.cfg.get("tpr", {}).get("decay", 1.0),
            "osc": self.cfg.get("osc", {}).get("decay", 1.0),
            "tpd": self.cfg.get("tpd", {}).get("decay", 1.0),
        }

        self.current_epoch = 0

    def _make_loss(self, cfg: Dict):
        name = cfg.get("name", "MSELoss")

        custom_losses = {
            "AutoMaskedMSELoss": AutoMaskedMSELoss,
            "AutoMaskMSELoss": AutoMaskedMSELoss,  # optional backwards-compatible alias
        }

        if name in custom_losses:
            kwargs = {k: v for k, v in cfg.items() if k not in ["name", "weight", "decay"]}
            return custom_losses[name](**kwargs)

        if not hasattr(nn, name):
            raise ValueError(f"torch.nn has no loss function '{name}'.")

        kwargs = {k: v for k, v in cfg.items() if k not in ["name", "weight", "decay"]}
        return getattr(nn, name)(**kwargs)
    
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def weight(self, task: str) -> float:
        return self.base_weights[task] * (self.decays[task] ** self.current_epoch)


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
            total = total + self.weight("conversion") * loss

        if "tpr" in predictions:
            y = data["h2_tpr"]["target"]
            loss = self.tpr_loss_fn(predictions["tpr"], y)
            losses["tpr"] = loss
            total = total + self.weight("tpr") * loss

        if "osc" in predictions:
            y = data["osc"]["target"]
            loss = self.osc_loss_fn(predictions["osc"], y)
            losses["osc"] = loss
            total = total + self.weight("osc") * loss

        if "tpd" in predictions:
            y = data["o2_tpd"]["target"]
            loss = self.tpd_loss_fn(predictions["tpd"], y)
            losses["tpd"] = loss
            total = total + self.weight("tpd") * loss

        losses["total"] = total
        return losses
    

class AutoMaskedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mask = ~torch.isnan(target)

        if mask.sum() == 0:
            return pred.sum() * 0.0

        return ((pred[mask] - target[mask]) ** 2).mean()
    
