from typing import Dict, Union, Any
from copy import deepcopy
from .model import LightOffModel
from pathlib import Path
import torch.optim as optimisers
import torch.optim.lr_scheduler as schedulers
from .custom_losses import CustomLosses


schedulers.ReduceLROnPlateau

class Trainer:
    def __init__(self, training_cfg: Dict[str, Dict]):
        self.cfg = deepcopy(training_cfg)
        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()

    def train(self, model: LightOffModel, outdir: Union[str, Path], datasets: Dict[str, Dict[str, Any]]):
        outdir = Path(outdir) if isinstance(outdir, str) else outdir


    def _make_optimizer(self):
        optimiser_string = self.cfg.get("optimiser", {}).pop("name", None)
        if optimiser_string is None:
            raise ValueError("Training config must have an 'optimiser' key as a Dict with a 'name' key corresponding to an optimiser in torch.optim and an 'lr' key for learning rate")
        if not hasattr(optimisers, self.cfg.get("optimiser")):
            raise ValueError(f"torch.optim does not have the optimiser {optimiser_string}")
        optimiser = getattr(optimisers, optimiser_string)
        try:
            return optimiser(**self.cfg.get("optimiser", {}))
        except Exception:
            raise ValueError("Could not create optimiser. Ensure all keys in the 'optimiser' Dict in the training config are valid arguments.")

    def _make_scheduler(self):
        scheduler_string = self.cfg.get("scheduler", {}).pop("name", None)
        if scheduler_string is None:
            raise ValueError("Training config must have an 'scheduler' key as a Dict with a 'name' key corresponding to an scheduler in torch.optim.lr_scheduler and that takes the optimiser as an argument")
        if not hasattr(schedulers, self.cfg.get("scheduler")):
            raise ValueError(f"torch.optim.lr_scheduler does not have the scheduler {scheduler_string}")
        scheduler = getattr(schedulers, scheduler_string)
        try:
            return scheduler(optimizer=self.optimizer, **self.cfg.get("scheduler", {}))
        except Exception:
            raise ValueError("Could not create scheduler. Ensure all keys in the 'scheduler' Dict in the training config are valid arguments.")