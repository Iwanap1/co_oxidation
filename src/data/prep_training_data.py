from .preprocessor import Preprocessor
from .metals import Metal
from typing import Dict, Optional, Tuple, List, Any, Literal, Union
import pandas as pd
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler
import torch

SplitMethod = Literal["Random_by_Material", "Random_by_Point", "Remove_Element", "Above_WHSV_Threshold"]


class Data:
    def __init__(self, 
        preprocessor: Preprocessor, 
        config: Dict, 
        config_name: Optional[str]=None, 
        split_method: SplitMethod="Random_by_Material", 
        split_value: Optional[Union[Metal, int]]=None,
        save_results: bool=True,
        data_outdir: Optional[Union[str, Path]]=None,
        analyse_data: bool=True
    ):
        self.preprocessor = preprocessor
        self.config = config
        self.config_name = config_name
        self.save_results = save_results
        if data_outdir: 
            self.save_dir = Path(data_outdir)
        elif self.config_name and self.save_results:
            self.save_dir = Path(f"data_analysis/{self.config_name}")
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.dataframes, _ = self.prepare_merged_dataframes_from_config()
        



    def prepare_merged_dataframes_from_config(
        self,
        override_min_appearances: Optional[int]=None,
        override_min_papers: Optional[int]=None
    ) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        
        base_dfs = self.preprocessor.get_base_dataframes(self.config)
        preprocessed = {}
        preprocessing_stats = {}
        materials_df, preprocessing_stats["materials"] = self.preprocessor.preprocess_materials(base_dfs["materials"],config=data_config)
        preprocessed["reactions"], preprocessing_stats["reactions"] = self.preprocessor.preprocess_reactions(base_dfs["reactions"],config=data_config)
        preprocessed["h2_tpr"], preprocessing_stats["h2_tpr"] = self.preprocessor.preprocess_h2_tpr_peaks(base_dfs["h2_tpr_peaks"],config=data_config)
        preprocessed["o2_tpd"], preprocessing_stats["o2_tpd"] = self.preprocessor.preprocess_tpd_peaks(base_dfs["o2_tpd_peaks"],config=data_config,config_section="o2_tpd_peaks")
        preprocessed["co2_tpd"], preprocessing_stats["co2_tpd"] = self.preprocessor.preprocess_tpd_peaks(base_dfs["co2_tpd_peaks"],config=data_config,config_section="co2_tpd_peaks")
        preprocessed["osc"], preprocessing_stats["osc"] = self.preprocessor.preprocess_osc(base_dfs["osc"],config=data_config)
        merge_stats = {}
        niche_element_stats = {}
        results = {"all_materials": materials_df}
        final_counts = {}
        if self.config_name and self.save_results:
            materials_df.to_csv(self.save_dir / "all_materials.csv", index=False)

        min_app = override_min_appearances if override_min_appearances is not None else self.config["material"]["element_min_appearances"]
        min_pap = override_min_papers if override_min_papers is not None else self.config["material"]["element_min_papers"]

        for key, df in preprocessed.items():
            if key == "reactions":
                merged, merge_stats[key] = self.preprocessor.merge_materials_and_reactions(materials_df, df)
                doi_col = "doi_material"
            else:
                merged, merge_stats[key] = self.preprocessor.merge_characterisation_with_materials(materials_df, df)
                doi_col = "doi_x"

            results[key], niche_element_stats[key] = self.preprocessor.filter_niche_elements(
                merged,
                min_appearances=min_app,
                min_papers=min_pap,
                doi_col=doi_col
            )
            final_counts[key] = len(results[key])
            if self.config_name and self.save_results:
                results[key].to_csv(self.save_dir / f"{key}_merged.csv", index=False)

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

        rng = np.random.default_rng(seed)
        rng.shuffle(material_ids)

        n_test = int(len(material_ids) * test_size)

        test_materials = set(material_ids[:n_test])
        train_materials = set(material_ids[n_test:])

        train_df_split = merged_df[merged_df[material_col].isin(train_materials)].reset_index(drop=True)
        test_df_split = merged_df[merged_df[material_col].isin(test_materials)].reset_index(drop=True)

        split_details["n_materials_total"] = len(material_ids)
        split_details["n_materials_train"] = len(train_materials)
        split_details["n_materials_test"] = len(test_materials)
        split_details["n_reactions_train"] = len(train_df_split)
        split_details["n_reactions_test"] = len(test_df_split)

        return train_df_split, test_df_split, split_details