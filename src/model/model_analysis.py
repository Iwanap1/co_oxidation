from typing import Dict, Union, Any
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

from .model import LightOffModel


class ModelAnalyser:
    def __init__(
        self,
        model: LightOffModel,
        datasets: Dict[str, Dict[str, Any]],
        device: Union[str, torch.device] = "cpu",
    ):
        self.model = model
        self.datasets = datasets
        self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.predictions = self._calculate_predictions()

    def _calculate_predictions(self) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
        predictions = {}

        with torch.no_grad():
            for split in ["train", "test"]:
                predictions[split] = {}

                for task, dataset_info in self.datasets.get(split, {}).items():
                    y_true, y_pred = self._predict_dataset(dataset_info, task)

                    predictions[split][task] = {
                        "y_true": y_true,
                        "y_pred": y_pred,
                    }

        return predictions

    def _predict_dataset(self, dataset_info: Dict[str, Any], task: str):
        dataset = dataset_info["dataset"]
        tensor_names = dataset_info["tensor_names"]

        data = {
            name: tensor.to(self.device)
            for name, tensor in zip(tensor_names, dataset.tensors)
        }

        if task == "reactions":
            pred = self.model(
                conversion_features=data.get("conversion_features"),
                reaction_inputs=data.get("reaction_inputs"),
                osc_features=data.get("osc_features"),
                tpr_features=data.get("tpr_features"),
                tpd_features=data.get("tpd_features"),
                whsv=data.get("whsv"),
                p_co=data.get("p_co"),
                p_o2=data.get("p_o2"),
            )
            true = data["target"]

        elif task == "h2_tpr":
            pred = self.model.predict_tpr(
                tpr_features=data["tpr_features"],
                ramp_rate=data.get("ramp_rate"),
            )
            true = data["target"]

        elif task == "osc":
            pred = self.model.predict_osc(
                osc_features=data["osc_features"],
                osc_direct_inputs=data.get("osc_direct_inputs"),
            )
            true = data["target"]
        
        elif task == "o2_tpd":
            pred = self.model.predict_tpd(
                tpd_features=data["tpd_features"],
                tpd_direct_inputs=data.get("tpd_direct_inputs"),
            )
            true = data["target"]

        else:
            raise ValueError(f"Unknown task: {task}")

        true = true.detach().cpu()
        pred = pred.detach().cpu()

        if true.ndim == 1 or true.shape[-1] == 1:
            true = true.flatten()
            pred = pred.flatten()

        return true, pred

    def conversion_metrics(self) -> Dict[str, Dict[str, float]]:
        results = self._metrics_for_task("reactions")
        self.stats = results
        return results

    def tpr_metrics(self) -> Dict[str, Dict[str, float]]:
        return self._metrics_for_task("h2_tpr")

    def osc_metrics(self) -> Dict[str, Dict[str, float]]:
        return self._metrics_for_task("osc")

    def _metrics_for_task(self, task: str) -> Dict[str, Dict[str, float]]:
        results = {}

        for split in ["train", "test"]:
            if task not in self.predictions.get(split, {}):
                continue

            y_true = self.predictions[split][task]["y_true"].numpy()
            y_pred = self.predictions[split][task]["y_pred"].numpy()

            # mask NaNs (important for O2-TPD)
            mask = ~np.isnan(y_true)

            if mask.sum() == 0:
                continue

            y_true_masked = y_true[mask]
            y_pred_masked = y_pred[mask]

            mse = mean_squared_error(y_true_masked, y_pred_masked)

            results[split] = {
                "r2": float(r2_score(y_true_masked, y_pred_masked)),
                "mse": float(mse),
                "mae": float(mean_absolute_error(y_true_masked, y_pred_masked)),
            }

        return results
    

    def tpd_metrics(self) -> Dict[str, Dict[str, float]]:
        return self._metrics_for_task("o2_tpd")

    def parity_plots(
        self,
        outdir: Union[str, Path],
        tasks=("reactions", "h2_tpr", "osc", "o2_tpd"),
        title_prefix: str = "",
    ):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            if not any(task in self.predictions.get(split, {}) for split in ["train", "test"]):
                continue

            fig, ax = plt.subplots(figsize=(6, 6))

            all_vals = []

            for split in ["train", "test"]:
                if task not in self.predictions.get(split, {}):
                    continue

                y_true = self.predictions[split][task]["y_true"]
                y_pred = self.predictions[split][task]["y_pred"]

                if y_true.ndim == 2:
                    for i in range(y_true.shape[1]):
                        mask = ~torch.isnan(y_true[:, i])

                        if mask.sum() == 0:
                            continue

                        yt = y_true[mask, i]
                        yp = y_pred[mask, i]

                        all_vals.append(yt)
                        all_vals.append(yp)

                        ax.scatter(
                            yt,
                            yp,
                            alpha=0.6,
                            label=f"{split} output {i}",
                        )
                else:
                    if task == "reactions" and hasattr(self, "stats"):
                        label = f"{split} (R²={self.stats[split]['r2']:.3f})"
                    else: 
                        label = split
                    mask = ~torch.isnan(y_true)

                    yt = y_true[mask]
                    yp = y_pred[mask]

                    all_vals.append(yt)
                    all_vals.append(yp)

                    ax.scatter(
                        yt,
                        yp,
                        alpha=0.6,
                        label=label,
                    )

            vals = torch.cat(all_vals)
            min_val = float(vals.min())
            max_val = float(vals.max())

            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
            ax.set_xlabel("True")
            ax.set_ylabel("Predicted")

            task_title = {
                "reactions": "Conversion",
                "h2_tpr": "H2-TPR",
                "osc": "OSC",
                "o2_tpd": "O2-TPD",
            }.get(task, task)

            ax.set_title(f"{title_prefix} {task_title} parity".strip())
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            fig.savefig(outdir / f"{task}_parity.png", dpi=300)
            plt.close(fig)


    def lightoff_curve_plots(
        self,
        outdir: Union[str, Path],
        n: int = 12,
        split: str = "test",
        seed: int = 42,
    ):
        """
        Plot predicted light-off curves alongside real conversion points
        for n random materials from the reaction dataset.
        """
        import numpy as np

        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        if "reactions" not in self.datasets.get(split, {}):
            raise KeyError(f"No reactions dataset found for split='{split}'.")

        rxn_info = self.datasets[split]["reactions"]

        if "metadata" not in rxn_info:
            raise KeyError(
                "Reaction dataset is missing metadata. "
                "Add metadata_cols to Data._make_named_tensor_dataset."
            )

        meta = rxn_info["metadata"].copy()

        material_col = "_id_material" if "_id_material" in meta.columns else "material_id"

        if material_col not in meta.columns:
            raise KeyError("Reaction metadata must contain '_id_material' or 'material_id'.")

        if "temperature" not in meta.columns:
            raise KeyError("Reaction metadata must contain 'temperature'.")

        y_true = self.predictions[split]["reactions"]["y_true"].numpy()
        y_pred = self.predictions[split]["reactions"]["y_pred"].numpy()

        meta["y_true"] = y_true
        meta["y_pred"] = y_pred

        materials = meta[material_col].dropna().unique()

        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            materials,
            size=min(n, len(materials)),
            replace=False,
        )

        for mat in chosen:
            df = meta.loc[meta[material_col] == mat].copy()
            df = df.sort_values("temperature")

            fig, ax = plt.subplots(figsize=(7, 5))

            ax.scatter(
                df["temperature"],
                df["y_true"],
                label="real data",
                alpha=0.8,
            )

            ax.plot(
                df["temperature"],
                df["y_pred"],
                label="prediction",
                linewidth=2,
            )

            ax.set_xlabel("Temperature")
            ax.set_ylabel("Conversion")
            ax.set_title(f"{split} light-off curve\n{mat}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            safe_mat = str(mat).replace("/", "_").replace(" ", "_")
            fig.savefig(outdir / f"{split}_lightoff_{safe_mat}.png", dpi=300)
            plt.close(fig)