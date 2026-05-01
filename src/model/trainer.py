from typing import Dict, Union, Any
from copy import deepcopy
from .model import LightOffModel
from pathlib import Path
import torch.optim as optimisers
import torch.optim.lr_scheduler as schedulers
from torch.utils.data import DataLoader
from .custom_losses import CustomLoss
from itertools import cycle
import torch

class Trainer:
    def __init__(self, training_cfg: Dict[str, Dict]):
        self.cfg = deepcopy(training_cfg)
        self.train_critereon = CustomLoss(training_cfg["train_critereon"])
        if training_cfg.get("best_model_critereon"):
            self.eval_critereon = CustomLoss(training_cfg["eval_critereon"])
        else:
            self.eval_critereon = self.train_critereon

    def train(
        self,
        model: LightOffModel,
        outdir: Union[str, Path],
        datasets: Dict[str, Dict[str, Any]],
        device: Union[str, torch.device] = "cpu",
        print_status_epochs: int=10
    ):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        device = torch.device(device)
        model = model.to(device)

        optimiser = self._make_optimiser(model)
        scheduler = self._make_scheduler(optimiser)

        train_loaders = self._make_dataloaders(datasets, split="train")
        test_loaders = self._make_dataloaders(datasets, split="test", shuffle=False)

        self.history = []

        epochs = self.cfg.get("epochs", 100)

        for epoch in range(epochs):
            model.train()

            train_metrics = self._run_epoch(
                model=model,
                loaders=train_loaders,
                datasets=datasets,
                split="train",
                optimiser=optimiser,
                device=device,
                critereon=self.train_critereon,
            )

            model.eval()
            with torch.no_grad():
                test_metrics = self._run_epoch(
                    model=model,
                    loaders=test_loaders,
                    datasets=datasets,
                    split="test",
                    optimiser=None,
                    device=device,
                    critereon=self.eval_critereon
                )

            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
            self.history.append(row)

            if scheduler is not None:
                if isinstance(scheduler, schedulers.ReduceLROnPlateau):
                    scheduler.step(test_metrics["total"])
                else:
                    scheduler.step()

            if epoch % print_status_epochs == 0:
                print(
                    f"epoch {epoch:04d} "
                    f"train_total={train_metrics['total']:.6f} "
                    f"test_total={test_metrics['total']:.6f}"
                )

        torch.save(model.state_dict(), outdir / "model.pt")
        return 


    def _make_optimiser(self, model: LightOffModel) -> Any:
        opt_cfg = deepcopy(self.cfg.get("optimiser", {}))
        optimiser_name = opt_cfg.pop("name", None)
        group_cfg = opt_cfg.pop("parameter_groups", None)

        if optimiser_name is None:
            raise ValueError("optimiser config must contain a 'name' key.")

        if not hasattr(optimisers, optimiser_name):
            raise ValueError(f"torch.optim does not have optimiser '{optimiser_name}'.")

        optimiser_cls = getattr(optimisers, optimiser_name)

        if group_cfg is None:
            return optimiser_cls(model.parameters(), **opt_cfg)

        param_groups = []
        used_param_ids = set()

        self._add_group_for_optimiser("conversion", [model.conversion_net], group_cfg, used_param_ids, param_groups)
        self._add_group_for_optimiser("osc", [model.osc_net, getattr(model, "osc_head", None)], group_cfg, used_param_ids, param_groups)
        self._add_group_for_optimiser("tpr", [model.tpr_net, getattr(model, "tpr_head", None)], group_cfg, used_param_ids, param_groups)
        remaining_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in used_param_ids
        ]
        if remaining_params:
            param_groups.append({"params": remaining_params})
        return optimiser_cls(param_groups, **opt_cfg)

    def _make_scheduler(self, optimiser) -> Any:
        sched_cfg = deepcopy(self.cfg.get("scheduler", {}))
        scheduler_name = sched_cfg.pop("name", None)

        if scheduler_name is None:
            return None

        if not hasattr(schedulers, scheduler_name):
            raise ValueError(f"torch.optim.lr_scheduler does not have scheduler {scheduler_name}")

        scheduler_cls = getattr(schedulers, scheduler_name)
        return scheduler_cls(optimiser, **sched_cfg)
                

    def _run_epoch(
        self,
        model: LightOffModel,
        loaders: Dict[str, DataLoader],
        datasets,
        split,
        optimiser: Any,
        device,
        critereon: CustomLoss
    ):
        reaction_loader = loaders["reactions"]

        tpr_iter = cycle(loaders["h2_tpr"]) if "h2_tpr" in loaders else None
        osc_iter = cycle(loaders["osc"]) if "osc" in loaders else None

        totals = {}
        n_steps = 0

        for rxn_batch in reaction_loader:
            rxn = self._batch_to_named_dict(
                rxn_batch,
                datasets[split]["reactions"]["tensor_names"],
                device=device,
            )

            batch_data = {"reactions": rxn}

            predictions = {}

            pred_conversion = model(
                conversion_features=rxn.get("conversion_features"),
                osc_features=rxn.get("osc_features"),
                tpr_features=rxn.get("tpr_features"),
                whsv=rxn.get("whsv"),
                p_co=rxn.get("p_co"),
                p_o2=rxn.get("p_o2"),
            )
            predictions["conversion"] = pred_conversion

            if tpr_iter is not None:
                tpr_batch = next(tpr_iter)
                tpr = self._batch_to_named_dict(
                    tpr_batch,
                    datasets[split]["h2_tpr"]["tensor_names"],
                    device=device,
                )

                predictions["tpr"] = model.predict_tpr(
                    tpr_features=tpr["tpr_features"],
                    ramp_rate=tpr.get("ramp_rate"),
                )
                batch_data["h2_tpr"] = tpr

            if osc_iter is not None:
                osc_batch = next(osc_iter)
                osc = self._batch_to_named_dict(
                    osc_batch,
                    datasets[split]["osc"]["tensor_names"],
                    device=device,
                )

                predictions["osc"] = model.predict_osc(osc_features=osc["osc_features"])
                batch_data["osc"] = osc

            losses = critereon(predictions, batch_data)

            if optimiser is not None:
                optimiser.zero_grad()
                losses["total"].backward()
                optimiser.step()

            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach().cpu())

            n_steps += 1

        return {k: v / max(n_steps, 1) for k, v in totals.items()}

    def _make_dataloaders(self, datasets, split: str, shuffle: bool = None):
        loader_cfg = self.cfg.get("dataloader", {})
        batch_size = loader_cfg.get("batch_size", 32)

        if shuffle is None:
            shuffle = loader_cfg.get("shuffle", split == "train")

        drop_last = loader_cfg.get("drop_last", False)

        return {
            name: DataLoader(
                ds_info["dataset"],
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            for name, ds_info in datasets[split].items()
        }

    def _batch_to_named_dict(self, batch, tensor_names, device):
        return {
            name: tensor.to(device)
            for name, tensor in zip(tensor_names, batch)
        }

    def _add_group_for_optimiser(self, name, modules, group_cfg, used_param_ids, param_groups):
        cfg = group_cfg.get(name, {})
        params = []

        for module in modules:
            if module is None:
                continue
            for p in module.parameters():
                if p.requires_grad:
                    params.append(p)
                    used_param_ids.add(id(p))

        if params:
            group = {"params": params}
            group.update(cfg)
            param_groups.append(group)