from .preprocessor import Preprocessor
from .metals import Metal, METALS
from ..visualisation.dataset_analyser import DatasetOverlapAnalysis
from typing import Dict, Optional, Tuple, List, Any, Literal, Union
import pandas as pd
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import numpy as np

SplitMethod = Literal["Random_by_Material", "Random_by_Point", "Remove_Metal", "Above_WHSV_Threshold"]


class Data:
    def __init__(self, 
        preprocessor: Preprocessor, 
        data_config: Dict, 
        data_config_name: Optional[str]=None, 
        row_by_datapoint: bool=False,
        split_method: SplitMethod="Random_by_Material", 
        split_value: Union[Metal, int, float]=0.2,
        save_results: bool=True,
        data_outdir: Optional[Union[str, Path]]=None,
        train_outdir: Optional[Union[str, Path]]=None,
        analyse_data: bool=True
    ):
        if split_method == "Random_by_Point" and not row_by_datapoint:
            raise ValueError("split_method 'Random_by_Point' requires row_by_datapoint=True to be set in the config.")
        self.preprocessor = preprocessor
        self.config = data_config
        self.config_name = data_config_name or self.config.get("name", None)
        self.save_results = save_results
        self.split_method = split_method
        self.split_value = split_value
        self.train_outdir = Path(train_outdir) if train_outdir else None
        if data_outdir: 
            self.save_dir = Path(data_outdir)
        elif self.config_name and self.save_results:
            self.save_dir = Path(f"data_analysis/{self.config_name}")
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
        print(self.save_dir)

        self.full_dataframes, stats = self.prepare_merged_dataframes_from_config(row_by_datapoint=row_by_datapoint)
        self.X_cols = self.resolve_x_cols(stats, row_by_datapoint)
        self.analyse_dataset() if analyse_data else None

        train_reactions, test_reactions = self.resolve_split()

        self.scaled_train, self.scaled_test, self.scaler = self.scale_and_transform(
            x_cols=self.X_cols,
            train_df=train_reactions,
            test_df=test_reactions,
            outdir=self.train_outdir,
            save_scaled_dfs=True
        )
    

    def prepare_merged_dataframes_from_config(self, row_by_datapoint: bool=False, override_min_appearances: Optional[int]=None, override_min_papers: Optional[int]=None) -> Tuple[Dict[str, pd.DataFrame], Dict]:

        base_dfs = self.preprocessor.get_base_dataframes(self.config)
        preprocessed = {}
        preprocessing_stats = {}
        materials_df, preprocessing_stats["materials"] = self.preprocessor.preprocess_materials(base_dfs["materials"],config=self.config)
        preprocessed["reactions"], preprocessing_stats["reactions"] = self.preprocessor.preprocess_reactions(base_dfs["reactions"],config=self.config)
        preprocessed["h2_tpr"], preprocessing_stats["h2_tpr"] = self.preprocessor.preprocess_h2_tpr_peaks(base_dfs["h2_tpr_peaks"],config=self.config)
        preprocessed["o2_tpd"], preprocessing_stats["o2_tpd"] = self.preprocessor.preprocess_tpd_peaks(base_dfs["o2_tpd_peaks"],config=self.config,config_section="o2_tpd_peaks")
        preprocessed["co2_tpd"], preprocessing_stats["co2_tpd"] = self.preprocessor.preprocess_tpd_peaks(base_dfs["co2_tpd_peaks"],config=self.config,config_section="co2_tpd_peaks")
        preprocessed["osc"], preprocessing_stats["osc"] = self.preprocessor.preprocess_osc(base_dfs["osc"],config=self.config)
        merge_stats = {}
        niche_element_stats = {}
        results = {"all_materials": materials_df}
        final_counts = {}

        if self.config_name and self.save_results:
            data_dir = self.save_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True) if self.save_dir else None
            materials_df.to_csv(data_dir / "all_materials.csv", index=False)

        min_app = override_min_appearances if override_min_appearances is not None else self.config["material"]["element_min_appearances"]
        min_pap = override_min_papers if override_min_papers is not None else self.config["material"]["element_min_papers"]

        for key, df in preprocessed.items():
            if key == "reactions":
                merged, merge_stats[key] = self.preprocessor.merge_materials_and_reactions(materials_df, df)
                doi_col = "doi_material"
            else:
                merged, merge_stats[key] = self.preprocessor.merge_characterisation_with_materials(materials_df, df)
                doi_col = "doi_x"

            results[key], niche_element_stats[key] = self.preprocessor.filter_niche_elements(merged, min_appearances=min_app, min_papers=min_pap, doi_col=doi_col)

            if row_by_datapoint and key == "reactions":
                results[key] = self.preprocessor.row_by_temperature(results[key])

            final_counts[key] = len(results[key])
            if self.config_name and self.save_results:
                results[key].to_csv(data_dir / f"{key}_merged.csv", index=False)

        stats = {
            "base_preprocess": preprocessing_stats,
            "materials_merge": merge_stats,
            "niche_elements_filter": niche_element_stats,
            "final_counts": final_counts
        }
        if self.config_name and self.save_results:
            with open(self.save_dir / "stats.json", "w") as f:
                json.dump(stats, f, indent=4)

        return results, stats
    
    def resolve_x_cols(self, stats: Dict, row_by_datapoint: bool) -> List[str]:
        base_set = self.config["x_cols_minus_T_and_elements"]
        element_set = stats["niche_elements_filter"]["reactions"]["kept_elements"]
        x_cols = base_set + element_set
        if row_by_datapoint:
            x_cols.append("temperature")
        return x_cols

    def analyse_dataset(self):
        analyser = DatasetOverlapAnalysis(preprocessor=self.preprocessor)
        analyser.full_analysis_from_merged_dataframes(
            self.full_dataframes, 
            output_path=self.save_dir if self.save_results else None,
            feature_cols=self.X_cols,
            show_figures=False
        )
    

    def scale_and_transform(
        self,
        x_cols: List[str],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        scaler: Optional[Any] = None,
        outdir: Optional[Path] = None,
        save_scaled_dfs: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        
        if scaler is None:
            scaler = StandardScaler()

        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df[x_cols] = scaler.fit_transform(train_df[x_cols])
        test_df[x_cols] = scaler.transform(test_df[x_cols])

        if outdir is not None:
            with open(outdir / "scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)
            if save_scaled_dfs:
                train_df.to_csv(outdir / "train_df_scaled.csv", index=False)
                test_df.to_csv(outdir / "test_df_scaled.csv", index=False)

        return train_df, test_df, scaler
    

    def convert_to_tensors(
        self,
        scaled_df: pd.DataFrame,
        x_cols: List[str],
        y_cols: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        missing_x = [col for col in x_cols if col not in scaled_df.columns]
        missing_y = [col for col in y_cols if col not in scaled_df.columns]

        if missing_x:
            raise KeyError(f"Missing x_cols in dataframe: {missing_x}")
        if missing_y:
            raise KeyError(f"Missing y_cols in dataframe: {missing_y}")

        X = torch.tensor(
            scaled_df[x_cols].to_numpy(dtype="float32", copy=True),
            dtype=torch.float32
        )

        y_arr = scaled_df[y_cols].to_numpy(dtype="float32", copy=True)
        if len(y_cols) == 1:
            y_arr = y_arr[:, 0]

        y = torch.tensor(y_arr, dtype=torch.float32)

        return X, y


    def resolve_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.split_method == "Random_by_Material":
            train, test, split_stats = self.split_by_material(self.full_dataframes["reactions"], test_size=self.split_value, seed=42)
        elif self.split_method == "Random_by_Point":
            train, test, split_stats = self.split_by_point(self.full_dataframes["reactions"], test_size=self.split_value)
        elif self.split_method == "Remove_Metal":
            if self.split_value is None or not isinstance(self.split_value, str):
                raise ValueError("For 'Remove_Metal' split method, split_value must be a string representing the metal to remove.")
            train = self.full_dataframes["reactions"][~self.full_dataframes["reactions"]["_id_material"].str.contains(self.split_value, case=False)].reset_index(drop=True)
            test = self.full_dataframes["reactions"][self.full_dataframes["reactions"]["_id_material"].str.contains(self.split_value, case=False)].reset_index(drop=True)
            split_stats = {
                "removed_metal": self.split_value,
                "n_reactions_removed": len(test),
                "n_reactions_remaining": len(train)
            }
        elif self.split_method == "Above_WHSV_Threshold":
            if self.split_value is None or not isinstance(self.split_value, (int, float)):
                raise ValueError("For 'Above_WHSV_Threshold' split method, split_value must be a number representing the WHSV threshold.")
            train = self.full_dataframes["reactions"][self.full_dataframes["reactions"]["flow_mL_h_g"] <= self.split_value].reset_index(drop=True)
            test = self.full_dataframes["reactions"][self.full_dataframes["reactions"]["flow_mL_h_g"] > self.split_value].reset_index(drop=True)
            split_stats = {
                "whsv_threshold": self.split_value,
                "n_reactions_above_threshold": len(test),
                "n_reactions_below_threshold": len(train)
            }
        else:
            raise NotImplementedError(f"Split method '{self.split_method}' not implemented yet.")
        
        if self.train_outdir:
            self.train_outdir.mkdir(parents=True, exist_ok=True)
            train.to_csv(self.train_outdir / "train_reactions.csv", index=False)
            test.to_csv(self.train_outdir / "test_reactions.csv", index=False)
            with open(self.train_outdir / "split_stats.json", "w") as f:
                json.dump(split_stats, f, indent=4)
        
        return train, test


    def split_by_point(self, merged_df: pd.DataFrame, test_size: float = 0.2, seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        split_details = {"seed": seed, "test_fraction": test_size}
        train_df_split, test_df_split = train_test_split(merged_df, test_size=test_size, random_state=seed)
        split_details["n_reactions_train"] = len(train_df_split)
        split_details["n_reactions_test"] = len(test_df_split)
        return train_df_split.reset_index(drop=True), test_df_split.reset_index(drop=True), split_details


    def split_by_material(
        self,
        merged_df: pd.DataFrame,
        test_size: float = 0.2,
        material_col: str = "_id_material",
        seed: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        
        split_details = {"seed": seed, "test_fraction": test_size}

        if material_col not in merged_df.columns:
            raise ValueError(
                f"Expected material id column '{material_col}' not found in dataframe. "
                f"Columns are: {merged_df.columns.tolist()}"
            )

        material_ids = merged_df[material_col].unique()

        train_materials, test_materials = train_test_split(material_ids, test_size=test_size, random_state=seed)

        train_df_split = merged_df[merged_df[material_col].isin(train_materials)].reset_index(drop=True)
        test_df_split = merged_df[merged_df[material_col].isin(test_materials)].reset_index(drop=True)

        split_details["n_materials_total"] = len(material_ids)
        split_details["n_materials_train"] = len(train_materials)
        split_details["n_materials_test"] = len(test_materials)
        split_details["n_reactions_train"] = len(train_df_split)
        split_details["n_reactions_test"] = len(test_df_split)

        return train_df_split, test_df_split, split_details
    

