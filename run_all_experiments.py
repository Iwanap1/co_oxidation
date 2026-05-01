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
    experiment_dir.mkdir(parents=True, exist_ok=True)
    db = DB(os.getenv("MONGO"))
    pp = Preprocessor(database=db)
    data_cfgs = _load_json(DATA_CONFIGS)
    model_cfgs = _load_json(MODEL_CONFIGS)
    train_cfgs = _load_json(TRAIN_CONFIGS)

    for d_cfg in data_cfgs:
        data = Data(preprocessor=pp, data_config=d_cfg, data_config_name=DATA_CONFIGS.get("name", "unnamed"))


def _load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)