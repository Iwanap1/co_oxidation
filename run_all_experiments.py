import json, os
from src.db import DB
from src.model import LightOffModel, Trainer, ModelAnalyser
from src.data import Preprocessor, Data
from src.visualisation.analyse_results import plot_remove_metal_test_mae_ptables
from pathlib import Path
import torch
import pandas as pd

EXPERIMENT_NAME = "2_dopant_featurisation_std_mlp_with_cs&lp"

DEFAULT_SPLIT_MODES = [
    ("Random_by_Material", 0.2), 
    ("Remove_Metal", "Fe"), 
    ("Above_WHSV_Threshold", 35000)
]

def main():
    experiment_dir = Path(f"experiments/{EXPERIMENT_NAME}")
    db = DB(os.getenv("MONGO"))
    pp = Preprocessor(database=db)
    data_cfgs = _load_json(f"configs/{EXPERIMENT_NAME}/data_configs.json")
    model_cfgs = _load_json(f"configs/{EXPERIMENT_NAME}/model_configs.json")
    train_cfgs = _load_json(f"configs/{EXPERIMENT_NAME}/training_configs.json")
    results = []
    try: 
        splits = _load_json(f"configs/{EXPERIMENT_NAME}/splits.json")
    except:
        print("Using default split modes because no splits.json found in config directory.")
        splits = DEFAULT_SPLIT_MODES

    for d_cfg in data_cfgs:
        data_name = d_cfg.get("name", "unnamed")
        data = Data(
            preprocessor=pp, 
            data_config=d_cfg, 
            data_config_name=data_name, 
            row_by_datapoint=True
        )

        for m_name, m_cfg in model_cfgs.items():
            for split_mode, split_value in splits:
                for i, train_cfg in enumerate(train_cfgs):
                    try:
                        tail = f"/train_config_{i}" if len(train_cfgs) > 1 else ""
                        parts = [m_name, data_name, f"{split_mode}_{split_value}"]
                        print("\n", "/".join(parts) + tail)

                        if tail:
                            parts.append(tail)
                        outdir = experiment_dir.joinpath(*parts)
                        try:
                            outdir.mkdir(parents=True, exist_ok=False)
                        except:
                            raise ValueError(f"Could not make directory {outdir}, ensure all config names are unique")
                        
                        data.set_split_and_scale(split_mode, split_value)
                        datasets = data.prepare_datasets(m_cfg)
                        save_dataset_debug_json(data, datasets, m_cfg, outdir)
                        data.save(outdir, save_scalers=True, save_preprocess_stats=True, save_scaled=False, save_unscaled=False, save_full=False)
                        m_cfg.update({"split_mode": split_mode, "split_value": split_value, "input_dims": data.input_dims})
                        config = {
                            "data_config": d_cfg,
                            "model_config": m_cfg,
                            "train_config": train_cfg,
                        }
                        with open(outdir / "config.json", "w") as f:
                            json.dump(config, f, indent=4)
                        model = LightOffModel(input_dims=data.input_dims, model_config=m_cfg)
                        result = {"data_config": data_name, "model_config": m_name, "split_mode": split_mode, "split_value": split_value, "train_config": train_cfg.get("name", f"train_config_{i}")}
                        trainer = Trainer(train_cfg)
                        best_model = trainer.train(model, outdir, datasets)
                        trainer.save_train_history(outdir, save_graph=True, save_csv=False)
                        analyser = ModelAnalyser(best_model, datasets)
                        metrics = analyser.conversion_metrics()

                        for split, vals in metrics.items():
                            for metric, value in vals.items():
                                result[f"{split}_{metric}"] = value

                        # Reaction dataset sizes
                        train_rxn = data.train_dataframes["reactions"]
                        test_rxn = data.test_dataframes["reactions"]

                        # Point counts
                        result["train_points"] = len(train_rxn)
                        result["test_points"] = len(test_rxn)

                        # Unique material counts
                        material_col = "_id_material"

                        result["train_materials"] = train_rxn[material_col].nunique()
                        result["test_materials"] = test_rxn[material_col].nunique()

                        analyser.parity_plots(outdir)
                        analyser.lightoff_curve_plots(outdir / "lightoff_curves", n=12, split="test")
                        results.append(result)
                    except Exception as e:
                        print(f"Error in experiment with model {m_name}, data config {data_name}, split mode {split_mode} and train config {train_cfg.get('name', f'train_config_{i}')}: {e}")
                        continue

    results_df = pd.DataFrame(results)
    results_df.to_csv(experiment_dir / "results_summary.csv", index=False)
    plot_remove_metal_test_mae_ptables(
        results_df=results_df,
        value_col="test_mae",
        group_cols=("data_config", "model_config", "train_config"),
        colorscale="Reds",
        save_dir=experiment_dir / "Figures/Dopant_Extrapolation_MAE_Ptables",
    )
    plot_remove_metal_test_mae_ptables(
        results_df=results_df,
        value_col="test_r2",
        group_cols=("data_config", "model_config", "train_config"),
        colorscale="Reds",
        save_dir=experiment_dir / "Figures/Dopant_Extrapolation_R2_Ptables",
    )


def _load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_dataset_debug_json(data, datasets, model_config, outdir):
    outdir = Path(outdir)

    debug = {
        "model_config_name": model_config.get("name", None),
        "data_config_name": data.config_name,
        "input_dims": getattr(data, "input_dims", None),
        "feature_cols": data.feature_cols,
        "target_cols": data.target_cols,
        "dataset_sizes": {},
        "tensor_names": {},
        "tensor_feature_names": {},
        "tensor_shapes": {},
    }

    for split, split_datasets in datasets.items():
        debug["dataset_sizes"][split] = {}
        debug["tensor_names"][split] = {}
        debug["tensor_feature_names"][split] = {}
        debug["tensor_shapes"][split] = {}

        for dataset_name, info in split_datasets.items():
            debug["dataset_sizes"][split][dataset_name] = info["n"]
            debug["tensor_names"][split][dataset_name] = info["tensor_names"]
            debug["tensor_feature_names"][split][dataset_name] = info["feature_names"]

            debug["tensor_shapes"][split][dataset_name] = {
                name: list(tensor.shape)
                for name, tensor in zip(
                    info["tensor_names"],
                    info["dataset"].tensors,
                )
            }

    with open(outdir / "dataset_debug.json", "w") as f:
        json.dump(debug, f, indent=4)

if __name__ == "__main__":
    main()