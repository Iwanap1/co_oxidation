import json

with open("configs/data_configs.json", "r") as f:
    data_configs = json.load(f)

with open("configs/model_configs.json", "r") as f:
    model_configs = json.load(f)

train_test_split_modes = ["Random_by_Material", "Remove_Metal", "Above_WHSV_Threshold"]

