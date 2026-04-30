from .preprocessor import Preprocessor
from .featurise_elements import Metal
from ..visualisation.dataset_analyser import DatasetOverlapAnalysis
from typing import Dict, Optional, Tuple, List, Any, Literal, Union
import pandas as pd
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import numpy as np
import warnings

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
        analyse_data: bool=False
    ):
        """Key Attributes:
-   .full_dataframes        raw merged + featurised dfs
-   .clean_dataframes       same as full without the irrelevant columns as per the data config
-   .feature_cols           cols to scale/use per df
-   .train_dataframes       unscaled train dfs
-   .test_dataframes        unscaled test dfs
-   .scaled_train_dfs       scaled train dfs
-   .scaled_test_dfs        scaled test dfs
-   .scalers                one scaler per df
-   .prepare_datasets()     returns TensorDatasets for each net in the contrastive learning NN
"""

        if split_method == "Random_by_Point" and not row_by_datapoint:
            raise ValueError("split_method 'Random_by_Point' requires row_by_datapoint=True to be set in the config.")
        self.preprocessor = preprocessor
        self.config = data_config
        self.config_name = data_config_name or self.config.get("name", None)
        self.save_results = save_results
        self.split_method = split_method
        self.split_value = split_value
        self.train_outdir = Path(train_outdir) if train_outdir else None
        self.target_cols = {
            "reactions": self.config["reactions"].get("target_cols", ["conversion"]),
            "h2_tpr": self.config["h2_tpr"].get("target_cols", ["temp"]), 
            "osc": self.config["osc"].get("target_cols", ["value_O_umol_per_g_catalyst"]),
        }

        if data_outdir: 
            self.save_dir = Path(data_outdir)
        elif self.config_name and self.save_results:
            self.save_dir = Path(f"data_analysis/{self.config_name}")
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.save_dir = None
        print(self.save_dir)

        self.full_dataframes, stats = self.prepare_merged_dataframes_from_config(
            row_by_datapoint=row_by_datapoint
        )

        self.feature_cols = self.resolve_feature_cols(row_by_datapoint)

        self.full_dataframes = self.drop_missing_feature_rows(self.full_dataframes, self.feature_cols)
        self.clean_dataframes = self.select_model_columns(self.full_dataframes, self.feature_cols)

        self.analyse_dataset() if analyse_data else None

        self.train_dataframes, self.test_dataframes = self.resolve_split_all()

        self.scaled_train_dfs, self.scaled_test_dfs, self.scalers = self.scale_and_transform(
            train_dfs=self.train_dataframes,
            test_dfs=self.test_dataframes,
            feature_cols=self.feature_cols,
            outdir=self.train_outdir,
            save_scaled_dfs=True,
        )
    
    def prepare_merged_dataframes_from_config(self, row_by_datapoint: bool=False, override_min_appearances: Optional[int]=None, override_min_papers: Optional[int]=None) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        base_dfs = self.preprocessor.get_base_dataframes(self.config)
        preprocessed = {}
        preprocessing_stats = {}
        materials_df, preprocessing_stats["materials"] = self.preprocessor.preprocess_materials(base_dfs["materials"],config=self.config)

        preprocessed["reactions"], preprocessing_stats["reactions"] = self.preprocessor.preprocess_reactions(base_dfs["reactions"],config=self.config)
        preprocessed["h2_tpr"], preprocessing_stats["h2_tpr"] = self.preprocessor.preprocess_h2_tpr_peaks(base_dfs["h2_tpr_peaks"],config=self.config)
        preprocessed["osc"], preprocessing_stats["osc"] = self.preprocessor.preprocess_osc(base_dfs["osc"],config=self.config)

        if self.config.get("o2_tpd_peaks") is not None:
            preprocessed["o2_tpd"], preprocessing_stats["o2_tpd"] = self.preprocessor.preprocess_tpd_peaks(base_dfs["o2_tpd_peaks"],config=self.config,config_section="o2_tpd_peaks")
        
        if self.config.get("co2_tpd_peaks") is not None:
            preprocessed["co2_tpd"], preprocessing_stats["co2_tpd"] = self.preprocessor.preprocess_tpd_peaks(base_dfs["co2_tpd_peaks"],config=self.config,config_section="co2_tpd_peaks")

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
            results[key] = self.preprocessor.convert_metals_to_dopant_features(
                results[key],
                self.config,
                config_section=key,
            )

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
    
    def _dopant_cols(self, df: pd.DataFrame) -> List[str]:
        return [
            c for c in df.columns
            if c.startswith("dopant_") or c == "n_dopants"
        ]

    def _check_cols(self, df: pd.DataFrame, cols: List[str], name: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} columns missing: {missing}")

    def resolve_feature_cols(self, row_by_datapoint: bool) -> Dict[str, List[str]]:
        material_cols = self.config["material"]["feature_cols_minus_elements"]

        feature_cols = {}

        reaction_df = self.full_dataframes["reactions"]
        conversion_cols = (
            material_cols
            + self.config["reactions"]["feature_cols_minus_conversion_temp"]
            + self._dopant_cols(reaction_df)
        )

        if row_by_datapoint:
            conversion_cols.append("temperature")

        self._check_cols(reaction_df, conversion_cols, "conversion")
        feature_cols["reactions"] = conversion_cols

        if "h2_tpr" in self.full_dataframes:
            tpr_df = self.full_dataframes["h2_tpr"]
            tpr_cols = (
                material_cols
                + self.config["h2_tpr"].get("feature_cols", [])
                + self._dopant_cols(tpr_df)
            )
            self._check_cols(tpr_df, tpr_cols, "tpr")
            feature_cols["h2_tpr"] = tpr_cols

        if "osc" in self.full_dataframes:
            osc_df = self.full_dataframes["osc"]
            osc_cols = (
                material_cols
                + self.config["osc"].get("feature_cols", [])
                + self._dopant_cols(osc_df)
            )
            self._check_cols(osc_df, osc_cols, "osc")
            feature_cols["osc"] = osc_cols

        return feature_cols

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
        train_dfs: Dict[str, pd.DataFrame],
        test_dfs: Dict[str, pd.DataFrame],
        feature_cols: Dict[str, List[str]],
        scalers: Optional[Dict[str, Any]] = None,
        outdir: Optional[Path] = None,
        save_scaled_dfs: bool = False,
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, Any]]:

        if scalers is None:
            scalers = {}

        scaled_train = {}
        scaled_test = {}

        for name, train_df in train_dfs.items():
            if name not in feature_cols:
                scaled_train[name] = train_df.copy()
                scaled_test[name] = test_dfs[name].copy()
                continue

            cols = feature_cols[name]

            train_df = train_df.copy()
            test_df = test_dfs[name].copy()

            scaler = scalers.get(name, StandardScaler())

            train_df[cols] = scaler.fit_transform(train_df[cols])
            test_df[cols] = scaler.transform(test_df[cols])

            scalers[name] = scaler
            scaled_train[name] = train_df
            scaled_test[name] = test_df

        if outdir is not None:
            outdir.mkdir(parents=True, exist_ok=True)

            with open(outdir / "scalers.pkl", "wb") as f:
                pickle.dump(scalers, f)

            if save_scaled_dfs:
                for name, df in scaled_train.items():
                    df.to_csv(outdir / f"{name}_train_scaled.csv", index=False)
                for name, df in scaled_test.items():
                    df.to_csv(outdir / f"{name}_test_scaled.csv", index=False)

        return scaled_train, scaled_test, scalers

    def resolve_split_all(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        train_reactions, test_reactions = self.resolve_split()

        material_col = "_id_material"

        train_materials = set(train_reactions[material_col])
        test_materials = set(test_reactions[material_col])

        train_dfs = {"reactions": train_reactions.reset_index(drop=True)}
        test_dfs = {"reactions": test_reactions.reset_index(drop=True)}

        for name, df in self.full_dataframes.items():
            if name in ["all_materials", "reactions"]:
                continue

            if material_col not in df.columns:
                continue

            train_dfs[name] = df[df[material_col].isin(train_materials)].reset_index(drop=True)
            test_dfs[name] = df[df[material_col].isin(test_materials)].reset_index(drop=True)

        return train_dfs, test_dfs

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
    
    def drop_missing_feature_rows(
        self,
        dfs: Dict[str, pd.DataFrame],
        feature_cols: Dict[str, List[str]],
    ) -> Dict[str, pd.DataFrame]:
        cleaned = {}

        for name, df in dfs.items():
            df = df.copy()

            if name not in feature_cols:
                cleaned[name] = df
                continue

            cols = feature_cols[name] + self.target_cols.get(name, [])
            cols = [c for c in cols if c in df.columns]

            n_before = len(df)
            nan_mask = df[cols].isna().any(axis=1)
            n_nan_rows = int(nan_mask.sum())

            if n_nan_rows > 0:
                # optional: show which columns are problematic
                nan_counts = df.loc[nan_mask, cols].isna().sum()
                nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

                warnings.warn(
                    f"[{name}] Dropping {n_nan_rows}/{n_before} rows with NaNs "
                    f"in selected feature/target columns.\n"
                    f"Top offending columns:\n{nan_counts.head(10)}"
                )

            cleaned[name] = df.loc[~nan_mask].reset_index(drop=True)

        return cleaned

    def select_model_columns(
        self,
        dfs: Dict[str, pd.DataFrame],
        feature_cols: Dict[str, List[str]],
    ) -> Dict[str, pd.DataFrame]:
        
        id_cols = {
            "reactions": ["_id_material", "material_id", "_id_reaction"],
            "h2_tpr": ["_id", "material_id"],
            "osc": ["_id", "material_id"],
        }

        selected = {}

        for name, df in dfs.items():
            if name not in feature_cols:
                selected[name] = df.copy()
                continue

            cols = (
                id_cols.get(name, [])
                + feature_cols[name]
                + self.target_cols[name]
            )

            cols = [c for c in dict.fromkeys(cols) if c in df.columns]
            selected[name] = df[cols].copy()

        return selected

    def prepare_datasets(self, model_config):
        """prepares datasets specific to the model, i.e. WHSV hybridisation model wants the flow_mL_h_g seperate from other features"""
        return NotImplemented
    