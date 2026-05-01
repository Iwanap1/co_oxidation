from .preprocessor import Preprocessor
from .featurise_elements import Metal, METALS
from ..visualisation.dataset_analyser import DatasetOverlapAnalysis
from typing import Dict, Optional, Tuple, List, Any, Literal, Union
import pandas as pd
from pathlib import Path
import json, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import numpy as np
import warnings
import torch

SplitMethod = Literal["Random_by_Material", "Random_by_Point", "Remove_Metal", "Above_WHSV_Threshold"]

class Data:
    def __init__(self, 
        preprocessor: Preprocessor, 
        data_config: Dict, 
        data_config_name: Optional[str]=None, 
        row_by_datapoint: bool=False
    ):
        """Key Attributes from init:
-   .full_dataframes        raw merged + featurised dfs
-   .clean_dataframes       same as full without the irrelevant columns as per the data config
-   .feature_cols           cols to scale/use per df
After running .set_split():
-   .train_dataframes       unscaled train dfs
-   .test_dataframes        unscaled test dfs
-   .scaled_train_dfs       scaled train dfs
-   .scaled_test_dfs        scaled test dfs
-   .scalers                one scaler per df
-   .prepare_datasets()     returns TensorDatasets for each net in the contrastive learning NN
"""
        self.preprocessor = preprocessor
        self.config = data_config
        self.config_name = data_config_name or self.config.get("name", None)
        self.target_cols = {
            "reactions": self.config["reactions"].get("target_cols", ["conversion"]),
            "h2_tpr": self.config["h2_tpr"].get("target_cols", ["temp"]), 
            "osc": self.config["osc"].get("target_cols", ["value_O_umol_per_g_catalyst"]),
        }
        self.full_dataframes, self.stats = self._prepare_merged_dataframes_from_config(
            row_by_datapoint=row_by_datapoint
        )
        self.feature_cols = self._resolve_feature_cols(row_by_datapoint)
        self.full_dataframes = self._drop_missing_feature_rows(self.full_dataframes, self.feature_cols)
        self.clean_dataframes = self._select_model_columns(self.full_dataframes, self.feature_cols)

    # Main functionality

    def analyse_dataset(self, output_path: Union[str, Path], show_figures: bool=True):
        raise NotImplementedError("DatasetOverlapAnalysis not yet updated to new config style")
        analyser = DatasetOverlapAnalysis(preprocessor=self.preprocessor)
        analyser.full_analysis_from_merged_dataframes(
            self.full_dataframes, 
            output_path=output_path,
            feature_cols=self.X_cols,
            show_figures=show_figures
        )

    def set_split_and_scale(self, split_method: Optional[SplitMethod] = None, split_value: Optional[Union[Metal, int, float]] = None) -> None:
        """
        Split is defined on reactions only
        For auxiliary datasets (OSC / TPR):

        Include in train if:
            - material ∈ train_materials
            OR
            - material ∉ reactions at all and does not contain forbidden metal

        Include in test if:
            - material ∈ test_materials
            OR
            - material ∉ reactions and does contain forbidden metal
            
        Scaler will scale all the feature cols. If a column is needed unscaled (due to hybridisation) this will be resolved when calling prepare_datasets
        """
        
        if split_method is not None:
            self.split_method = split_method

        if split_value is not None:
            self.split_value = split_value

        self.train_dataframes, self.test_dataframes = self._resolve_split_all()

        self.scaled_train_dfs, self.scaled_test_dfs, self.scalers = self._scale_and_transform(
            train_dfs=self.train_dataframes,
            test_dfs=self.test_dataframes,
            feature_cols=self.feature_cols
        )

    def prepare_datasets(self, model_config: Dict) -> Dict[str, Dict[str, Any]]:
        """
        Create model-config-specific TensorDatasets. Will use unscaled data if that column is not in the features

        Returns:
            {
                "train": {
                    "reactions": {
                        "dataset": TensorDataset(...),
                        "tensor_names": [...],
                        "feature_names": {...},
                        "n": ...
                    },
                    "h2_tpr": {...},
                    "osc": {...},
                },
                "test": {...}
            }

        For reactions, possible tensors are:
            - conversion_features (scaled)
            - whsv (unscaled)
            - p_co (unscaled)
            - p_o2 (unscaled)
            - target (unscaled)

        For h2_tpr:
            - tpr_features (scaled)
            - ramp_rate (scaled)
            - target (unscaled)

        For osc:
            - osc_features (scaled)
            - target (unscaled)
        """

        prepared = {"train": {}, "test": {}}
        tensor_cols_for_input_dims = {}

        conversion_cols = self._resolve_conversion_input_cols(model_config)
        try:
            dfs = [("train", self.scaled_train_dfs, self.train_dataframes),("test", self.scaled_test_dfs, self.test_dataframes)]
        except:
            raise ValueError("You need to run set_split_and_scale first")

        for split_name, scaled_dfs, raw_dfs in dfs:
            # Reactions
            rxn_df = scaled_dfs["reactions"]
            raw_rxn_df = raw_dfs["reactions"]

            reaction_tensor_cols = {"conversion_features": conversion_cols}
            if split_name == "train":
                tensor_cols_for_input_dims["reactions"] = reaction_tensor_cols.copy()

            if model_config.get("hybridise_whsv", False):
                if "flow_mL_h_g" not in raw_rxn_df.columns:
                    raise KeyError("hybridise_whsv=True but 'flow_mL_h_g' is missing.")

                rxn_df["flow_mL_h_g"] = raw_rxn_df["flow_mL_h_g"].to_numpy()
                reaction_tensor_cols["whsv"] = ["flow_mL_h_g"]

            if model_config.get("hybridise_pressures", False):
                pressure_cols = ["gas_co_content", "gas_o2_content"]
                missing = [c for c in pressure_cols if c not in raw_rxn_df.columns]

                if missing: raise KeyError(f"hybridise_pressures=True but pressure columns are missing: {missing}")

                for c in pressure_cols:
                    rxn_df[c] = raw_rxn_df[c].to_numpy()

                reaction_tensor_cols["p_co"] = ["gas_co_content"]
                reaction_tensor_cols["p_o2"] = ["gas_o2_content"]

            reaction_tensor_cols["target"] = self.target_cols["reactions"]

            prepared[split_name]["reactions"] = self._make_named_tensor_dataset(rxn_df, reaction_tensor_cols)

            # H2-TPR
            if model_config.get("tpr_net") is not None:
                if "h2_tpr" not in scaled_dfs:
                    raise KeyError("model_config contains tpr_net but no h2_tpr dataframe exists.")

                tpr_df = scaled_dfs["h2_tpr"]

                tpr_tensor_cols = {
                    "tpr_features": self._resolve_tpr_input_cols(model_config)
                }

                if model_config["tpr_net"].get("condition_tpr_with_ramp_rate", False):
                    if "ramp_rate_C_min" not in tpr_df.columns:
                        raise KeyError(
                            "condition_tpr_with_ramp_rate=True but 'ramp_rate_C_min' is missing."
                        )
                    tpr_tensor_cols["ramp_rate"] = ["ramp_rate_C_min"]

                tpr_tensor_cols["target"] = self.target_cols["h2_tpr"]

                prepared[split_name]["h2_tpr"] = self._make_named_tensor_dataset(
                    tpr_df,
                    tpr_tensor_cols,
                )

                if split_name == "train":
                    tensor_cols_for_input_dims["h2_tpr"] = tpr_tensor_cols.copy()
                
            # OSC
            if model_config.get("osc_net") is not None:
                if "osc" not in scaled_dfs: raise KeyError("model_config contains osc_net but no osc dataframe exists.")
                osc_tensor_cols = { "osc_features": self.feature_cols["osc"], "target": self.target_cols["osc"]}
                prepared[split_name]["osc"] = self._make_named_tensor_dataset(scaled_dfs["osc"], osc_tensor_cols)
                if split_name == "train": tensor_cols_for_input_dims["osc"] = osc_tensor_cols.copy()

        self.input_dims = self._resolve_input_dims(
            model_config=model_config,
            tensor_cols_by_dataset=tensor_cols_for_input_dims,
        )
        return prepared

    def save(
        self,
        outdir: Union[str, Path],
        save_scalers: bool = True,
        save_preprocess_stats: bool = True,
        save_split_stats: bool = True,
        save_scaled: bool = False,
        save_unscaled: bool = False,
        save_full: bool = False,
    ) -> None:
        """
        Save current state of Data object.

        Args:
            outdir: directory to save into
            save_scalers: save fitted scalers
            save_scaled: save scaled train/test dfs
            save_preprocess_stats: save stats about filtrations from the Preprocessor
            save_unscaled: save unscaled train/test dfs
            save_full: save full (pre-clean) dataframes
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        meta = {
            "config_name": self.config_name,
            "split_method": getattr(self, "split_method", None),
            "split_value": getattr(self, "split_value", None),
            "feature_cols": self.feature_cols,
            "target_cols": self.target_cols,
        }

        with open(outdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=4)

        if save_preprocess_stats:
            with open(outdir / "preprocess_stats.json", "w") as f:
                json.dump(self.stats, f, indent=4)
            
        if save_split_stats:
            with open(outdir / "train_test_split_stats.json", "w") as f:
                json.dump(self.split_stats, f, indent=4)

        if save_scalers and hasattr(self, "scalers"):
            with open(outdir / "scalers.pkl", "wb") as f:
                pickle.dump(self.scalers, f)

        if save_scaled and hasattr(self, "scaled_train_dfs"):
            scaled_dir = outdir / "scaled"
            scaled_dir.mkdir(exist_ok=True)

            for name, df in self.scaled_train_dfs.items():
                df.to_csv(scaled_dir / f"{name}_train.csv", index=False)

            for name, df in self.scaled_test_dfs.items():
                df.to_csv(scaled_dir / f"{name}_test.csv", index=False)

        if save_unscaled and hasattr(self, "train_dataframes"):
            unscaled_dir = outdir / "unscaled"
            unscaled_dir.mkdir(exist_ok=True)

            for name, df in self.train_dataframes.items():
                df.to_csv(unscaled_dir / f"{name}_train.csv", index=False)

            for name, df in self.test_dataframes.items():
                df.to_csv(unscaled_dir / f"{name}_test.csv", index=False)

        if save_full:
            full_dir = outdir / "full"
            full_dir.mkdir(exist_ok=True)

            for name, df in self.full_dataframes.items():
                df.to_csv(full_dir / f"{name}.csv", index=False)

    # Helpers

    def _prepare_merged_dataframes_from_config(self, row_by_datapoint: bool=False, override_min_appearances: Optional[int]=None, override_min_papers: Optional[int]=None) -> Tuple[Dict[str, pd.DataFrame], Dict]:
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

        stats = {
            "base_preprocess": preprocessing_stats,
            "materials_merge": merge_stats,
            "niche_elements_filter": niche_element_stats,
            "final_counts": final_counts
        }

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

    def _resolve_feature_cols(self, row_by_datapoint: bool) -> Dict[str, List[str]]:
        feature_cols = {}

        reaction_df = self.full_dataframes["reactions"]
        reaction_material_cols = self._resolve_material_feature_cols_for_dataset("reactions")

        conversion_cols = (
            reaction_material_cols
            + self.config["reactions"]["feature_cols_minus_conversion_temp"]
            + self._dopant_cols(reaction_df)
        )

        if row_by_datapoint:
            conversion_cols.append("temperature")

        conversion_cols = list(dict.fromkeys(conversion_cols))
        self._check_cols(reaction_df, conversion_cols, "conversion")
        feature_cols["reactions"] = conversion_cols

        if "h2_tpr" in self.full_dataframes:
            tpr_df = self.full_dataframes["h2_tpr"]
            tpr_material_cols = self._resolve_material_feature_cols_for_dataset("h2_tpr")

            tpr_cols = (
                tpr_material_cols
                + self.config["h2_tpr"].get("feature_cols", [])
                + self._dopant_cols(tpr_df)
            )

            tpr_cols = list(dict.fromkeys(tpr_cols))
            self._check_cols(tpr_df, tpr_cols, "tpr")
            feature_cols["h2_tpr"] = tpr_cols

        if "osc" in self.full_dataframes:
            osc_df = self.full_dataframes["osc"]
            osc_material_cols = self._resolve_material_feature_cols_for_dataset("osc")

            osc_cols = (
                osc_material_cols
                + self.config["osc"].get("feature_cols", [])
                + self._dopant_cols(osc_df)
            )

            osc_cols = list(dict.fromkeys(osc_cols))
            self._check_cols(osc_df, osc_cols, "osc")
            feature_cols["osc"] = osc_cols

        return feature_cols

    def _scale_and_transform(
        self,
        train_dfs: Dict[str, pd.DataFrame],
        test_dfs: Dict[str, pd.DataFrame],
        feature_cols: Dict[str, List[str]],
        scalers: Optional[Dict[str, Any]] = None
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
            
        return scaled_train, scaled_test, scalers

    def _resolve_split_all(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        train_reactions, test_reactions = self._resolve_split()

        reaction_material_col = "_id_material"
        train_materials = set(train_reactions[reaction_material_col])
        test_materials = set(test_reactions[reaction_material_col])
        all_reaction_materials = train_materials | test_materials
        train_dfs = {"reactions": train_reactions.reset_index(drop=True)}
        test_dfs = {"reactions": test_reactions.reset_index(drop=True)}
        for name, df in self.clean_dataframes.items():
            if name in ["all_materials", "reactions"]:
                continue
            material_col = "_id_material" if "_id_material" in df.columns else "material_id"
            if material_col not in df.columns:
                warnings.warn(f"[{name}] No material id column found; skipping split.")
                continue
            material_ids = df[material_col]

            overlap_train_mask = material_ids.isin(train_materials)
            overlap_test_mask = material_ids.isin(test_materials)
            aux_only_mask = ~material_ids.isin(all_reaction_materials)

            if self.split_method == "Remove_Metal":
                # Ensure that no entries in the auxillary train sets (OSC and TPR) contain the excluded metal to avoid leakage
                metal = self.split_value

                if metal not in df.columns:
                    raise KeyError(
                        f"[{name}] Cannot enforce Remove_Metal='{metal}' because "
                        f"column '{metal}' is missing. Keep element columns as metadata."
                    )

                metal_mask = pd.to_numeric(df[metal], errors="coerce").fillna(0).gt(0)

                # overlapping materials follow reaction split
                # aux-only materials containing removed metal go to test, not train
                train_mask = (overlap_train_mask | aux_only_mask) & ~metal_mask
                test_mask = overlap_test_mask | metal_mask

            elif self.split_method in ["Random_by_Material", "Random_by_Point", "Above_WHSV_Threshold"]:
                # Don't need to worry about leaking entries into auxillory datasets (OSC and TPR) for any of these
                train_mask = overlap_train_mask | aux_only_mask
                test_mask = overlap_test_mask

            else:
                raise NotImplementedError(f"Split method '{self.split_method}' not implemented.")

            n_aux_only = int(aux_only_mask.sum())
            n_aux_only_test = int((aux_only_mask & test_mask).sum())

            if n_aux_only > 0:
                warnings.warn(
                    f"[{name}] Found {n_aux_only} auxiliary-only rows. "
                    f"{n_aux_only_test} assigned to test under split_method={self.split_method}."
                )

            train_dfs[name] = df.loc[train_mask].reset_index(drop=True)
            test_dfs[name] = df.loc[test_mask].reset_index(drop=True)

        return train_dfs, test_dfs

    def _resolve_split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.split_method == "Random_by_Material":
            train, test, self.split_stats = self._split_by_material(self.clean_dataframes["reactions"], test_size=self.split_value, seed=42)
        elif self.split_method == "Random_by_Point":
            train, test, self.split_stats = self._split_by_point(self.clean_dataframes["reactions"], test_size=self.split_value)
        elif self.split_method == "Remove_Metal":
            if self.split_value is None or not isinstance(self.split_value, str):
                raise ValueError(
                    "For 'Remove_Metal' split method, split_value must be a string representing the metal to remove."
                )

            reaction_df = self.clean_dataframes["reactions"]
            metal = self.split_value

            if metal not in reaction_df.columns:
                raise KeyError(
                    f"Remove_Metal='{metal}' requires column '{metal}' in reactions dataframe."
                )

            metal_mask = pd.to_numeric(reaction_df[metal], errors="coerce").fillna(0).gt(0)

            train = reaction_df.loc[~metal_mask].reset_index(drop=True)
            test = reaction_df.loc[metal_mask].reset_index(drop=True)
            self.split_stats = {
                "removed_metal": metal,
                "n_reactions_removed": len(test),
                "n_reactions_remaining": len(train),
            }
        elif self.split_method == "Above_WHSV_Threshold":
            if self.split_value is None or not isinstance(self.split_value, (int, float)):
                raise ValueError("For 'Above_WHSV_Threshold' split method, split_value must be a number representing the WHSV threshold.")
            train = self.clean_dataframes["reactions"][self.clean_dataframes["reactions"]["flow_mL_h_g"] <= self.split_value].reset_index(drop=True)
            test = self.clean_dataframes["reactions"][self.clean_dataframes["reactions"]["flow_mL_h_g"] > self.split_value].reset_index(drop=True)
            self.split_stats = {
                "whsv_threshold": self.split_value,
                "n_reactions_above_threshold": len(test),
                "n_reactions_below_threshold": len(train)
            }
        else:
            raise NotImplementedError(f"Split method '{self.split_method}' not implemented yet.")
        
        return train, test

    def _split_by_point(self, merged_df: pd.DataFrame, test_size: float = 0.2, seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        split_details = {"seed": seed, "test_fraction": test_size}
        train_df_split, test_df_split = train_test_split(merged_df, test_size=test_size, random_state=seed)
        split_details["n_reactions_train"] = len(train_df_split)
        split_details["n_reactions_test"] = len(test_df_split)
        return train_df_split.reset_index(drop=True), test_df_split.reset_index(drop=True), split_details

    def _split_by_material(self, merged_df: pd.DataFrame, test_size: float = 0.2, material_col: str = "_id_material", seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        
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
    
    def _drop_missing_feature_rows(
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
            self._check_cols(df, cols, name)

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

    def _select_model_columns(
        self,
        dfs: Dict[str, pd.DataFrame],
        feature_cols: Dict[str, List[str]],
    ) -> Dict[str, pd.DataFrame]:
        
        id_cols = {
            "reactions": ["_id_material", "material_id", "_id_reaction"] + METALS,
            "h2_tpr": ["_id", "material_id"] + METALS,
            "osc": ["_id", "material_id"] + METALS,
        }

        selected = {}

        for name, df in dfs.items():
            if name not in feature_cols:
                selected[name] = df.copy()
                continue

            required_cols = feature_cols[name] + self.target_cols[name]
            self._check_cols(df, required_cols, name)

            cols = (
                id_cols.get(name, [])
                + feature_cols[name]
                + self.target_cols[name]
            )

            cols = [c for c in dict.fromkeys(cols) if c in df.columns]
            selected[name] = df[cols].copy()

        return selected
    
    def _to_tensor(self, df: pd.DataFrame, cols: List[str], device: Optional[Union[str, torch.device]] = None,) -> torch.Tensor:
        if len(cols) == 0:
            return torch.empty((len(df), 0), dtype=torch.float32)
        return torch.tensor(
            df[cols].to_numpy(dtype=np.float32),
            dtype=torch.float32,
            device=device
        )
    
    def _resolve_conversion_input_cols(self, model_config: Dict) -> List[str]:
        conv_cfg = model_config.get("conversion_net", {})

        include_material_features = conv_cfg.get("include_material_features", True)

        if not include_material_features:
            cols = []
        else:
            included_features = conv_cfg.get("included_features", "all")

            if included_features == "all":
                cols = list(self.feature_cols["reactions"])
            elif isinstance(included_features, list):
                allowed = set(self.feature_cols["reactions"])
                missing = [c for c in included_features if c not in allowed]
                if missing:
                    raise KeyError(
                        f"conversion_net.included_features contains columns not in "
                        f"reaction feature columns: {missing}"
                    )
                cols = list(included_features)
            else:
                raise ValueError(
                    "conversion_net.included_features must be either 'all' or a list of columns"
                )

        if model_config.get("hybridise_whsv", False):
            cols = [c for c in cols if c != "flow_mL_h_g"]

        if model_config.get("hybridise_pressures", False):
            cols = [c for c in cols if c not in ["gas_co_content", "gas_o2_content"]]

        return cols

    def _resolve_material_feature_cols_for_dataset(self, dataset_name: str) -> List[str]:
        """
        Dataset-specific material feature columns.

        Fallback:
            material.feature_cols_minus_elements

        Override:
            reactions.material_feature_cols
            h2_tpr.material_feature_cols
            osc.material_feature_cols
        """
        fallback = self.config["material"].get("feature_cols_minus_elements", [])

        if dataset_name == "reactions":
            section = self.config["reactions"]
        else:
            section = self.config.get(dataset_name, {})

        return section.get("material_feature_cols", fallback)    

    def _make_named_tensor_dataset(self, df: pd.DataFrame, tensor_cols: Dict[str, List[str]]) -> Dict[str, Any]:
        tensors = []
        tensor_names = []
        feature_names = {}

        for name, cols in tensor_cols.items():
            tensor_names.append(name)
            feature_names[name] = cols
            tensors.append(self._to_tensor(df, cols))

        dataset = TensorDataset(*tensors)

        return {
            "dataset": dataset,
            "tensor_names": tensor_names,
            "feature_names": feature_names,
            "n": len(df),
        }
    
    def _resolve_tpr_input_cols(self, model_config: Dict) -> List[str]:
        cols = list(self.feature_cols["h2_tpr"])
        tpr_cfg = model_config.get("tpr_net") or {}

        if tpr_cfg.get("condition_tpr_with_ramp_rate", False):
            cols = [c for c in cols if c != "ramp_rate_C_min"]

        return cols

    def _restore_unscaled_physical_cols(
        self,
        scaled_dfs: Dict[str, pd.DataFrame],
        raw_dfs: Dict[str, pd.DataFrame],
        cols_by_dataset: Dict[str, List[str]],
    ) -> Dict[str, pd.DataFrame]:
        out = {}

        for name, df in scaled_dfs.items():
            df = df.copy()

            for col in cols_by_dataset.get(name, []):
                if col in df.columns and col in raw_dfs[name].columns:
                    df[f"{col}_unscaled"] = raw_dfs[name][col].to_numpy()

            out[name] = df

    def _resolve_input_dims(
        self,
        model_config: Dict,
        tensor_cols_by_dataset: Dict[str, Dict[str, List[str]]],
    ) -> Dict[str, int]:
        input_dims = {}

        rxn_cols = tensor_cols_by_dataset["reactions"]
        input_dims["conversion"] = len(rxn_cols.get("conversion_features", []))

        if model_config.get("osc_net") is not None:
            input_dims["osc"] = len(tensor_cols_by_dataset["osc"]["osc_features"])

        if model_config.get("tpr_net") is not None:
            input_dims["tpr"] = len(tensor_cols_by_dataset["h2_tpr"]["tpr_features"])

        return input_dims