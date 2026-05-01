import json, os

from src.db import DB
from src.model import LightOffModel, Trainer, ModelAnalyser
from src.data import Preprocessor, Data

from pathlib import Path

EXPERIMENT_NAME = "test"
DATA_CONFIGS = "configs/data_configs.json"
MODEL_CONFIGS = "configs/model_configs.json"
TRAIN_CONFIGS = "configs/training_configs.json"
SPLIT_MODES = [
    ("Random_by_Material", 0.2), 
    ("Remove_Metal", "Fe"), 
    ("Above_WHSV_Threshold", 35000)
]


def main():
    experiment_dir = Path(f"experiments/{EXPERIMENT_NAME}")
    db = DB(os.getenv("MONGO"))
    pp = Preprocessor(database=db)
    data_cfgs = _load_json(DATA_CONFIGS)
    model_cfgs = _load_json(MODEL_CONFIGS)
    train_cfgs = _load_json(TRAIN_CONFIGS)

    for d_cfg in data_cfgs:
        data_name = d_cfg.pop("name", "unnamed")
        data = Data(preprocessor=pp, data_config=d_cfg, data_config_name=data_name)
        for m_name, m_cfg in model_cfgs.items():
            for split_mode, split_value in SPLIT_MODES:
                for i, train_cfg in enumerate(train_cfgs):
                    tail = f"/train_config_{i}" if len(train_cfgs) > 1 else ""
                    outdir = experiment_dir / f"{m_name}/{data_name}/{split_mode}_{split_value}{tail}"
                    try:
                        outdir.mkdir(parents=True, exist_ok=False)
                    except:
                        raise ValueError(f"Could not make directory {outdir}, ensure all config names are unique")
                    
                    data.set_split_and_scale(split_mode, split_value)
                    datasets = data.prepare_datasets(m_cfg)
                    data.save(outdir, save_scalers=True, save_preprocess_stats=True, save_scaled=False, save_unscaled=True, save_full=False)

                    model = LightOffModel(input_dims=data.input_dims, model_config=m_cfg)


def _load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
    

if __name__ == "__main__":
    main()