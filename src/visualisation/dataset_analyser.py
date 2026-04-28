from ..db import DB
from ..data.preprocessor import Preprocessor
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.constants import ELEMENTS
from venny4py.venny4py import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DatasetOverlapAnalysis:
    def __init__(self, database: Optional[DB]=None, preprocessor: Optional[Preprocessor]=None):
        if database and not preprocessor:
            self.database = database
            self.preprocessor = Preprocessor(self.database)
        elif preprocessor and not database:
            self.database = preprocessor.database
            self.preprocessor = preprocessor

    def full_analysis_from_data_config(
        self,
        data_config: Dict,
        override_min_appearances: Optional[int]=None,
        override_min_papers: Optional[int]=None,
        output_path: Optional[Path] = None,
        show_figures: bool = True
    ) -> Dict:
        merged_dataframes, preprocessing_stats = self.prepare_merged_dataframes_from_config(data_config, override_min_appearances=override_min_appearances, override_min_papers=override_min_papers)
        self.full_analysis_from_merged_dataframes(
            merged_dataframes,
            output_path=output_path,
            show_figures=show_figures,
            feature_cols=data_config["x_cols_minus_T_and_elements"] + [e for e in merged_dataframes["all_materials"].columns if e in ELEMENTS]
        )
        return preprocessing_stats

    def full_analysis_from_merged_dataframes(
        self,
        merged_dataframes: Dict[str, pd.DataFrame],
        output_path: Optional[Path] = None,
        show_figures: bool = True,
        feature_cols: List[str] = [],
        only_pca_reactions: bool = True
    ) -> None:
        if output_path is None:
            output_path = None
        elif isinstance(output_path, str):
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        
        self.element_frequencies_in_dataset_bar(
            merged_dataframes,
            output_path=output_path,
            show_figures=show_figures
        )
        self.material_overlap_venn_diagram(merged_dataframes, save_path=output_path)
        _, _ = self.pca_overlap(merged_dataframes=merged_dataframes, feature_cols=feature_cols, output_path=output_path, show_figures=show_figures, pca_on_reaction_mats_only=only_pca_reactions)

    def prepare_merged_dataframes_from_config(
        self,
        data_config: Dict,
        override_min_appearances: Optional[int]=None,
        override_min_papers: Optional[int]=None
    ) -> Tuple[Dict[str, pd.DataFrame], Dict]:
        base_dfs = self.preprocessor.get_base_dataframes(data_config)
        preprocessed = {}
        preprocessing_stats = {}
        materials_df, preprocessing_stats["materials"] = self.preprocessor.preprocess_materials(
            base_dfs["materials"],
            config=data_config
        )
        preprocessed["reactions"], preprocessing_stats["reactions"] = self.preprocessor.preprocess_reactions(
            base_dfs["reactions"],
            config=data_config
        )
        preprocessed["h2_tpr"], preprocessing_stats["h2_tpr"] = self.preprocessor.preprocess_h2_tpr_peaks(
            base_dfs["h2_tpr_peaks"],
            config=data_config
        )
        preprocessed["o2_tpd"], preprocessing_stats["o2_tpd"] = self.preprocessor.preprocess_tpd_peaks(
            base_dfs["o2_tpd_peaks"],
            config=data_config,
            config_section="o2_tpd_peaks"
        )
        preprocessed["co2_tpd"], preprocessing_stats["co2_tpd"] = self.preprocessor.preprocess_tpd_peaks(
            base_dfs["co2_tpd_peaks"],
            config=data_config,
            config_section="co2_tpd_peaks"
        )
        preprocessed["osc"], preprocessing_stats["osc"] = self.preprocessor.preprocess_osc(
            base_dfs["osc"],
            config=data_config
        )

        merge_stats = {}
        niche_element_stats = {}
        results = {"all_materials": materials_df}
        final_counts = {}

        min_app = (
            override_min_appearances
            if override_min_appearances is not None
            else data_config["material"]["element_min_appearances"]
        )
        min_pap = (
            override_min_papers
            if override_min_papers is not None
            else data_config["material"]["element_min_papers"]
        )

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

        stats = {
            "base_preprocess": preprocessing_stats,
            "materials_merge": merge_stats,
            "niche_elements_filter": niche_element_stats,
            "final_counts": final_counts
        }

        return results, stats

    def element_frequencies_in_dataset_bar(
        self,
        merged_dataframes: Dict[str, pd.DataFrame],
        output_path: Optional[Path] = None,
        show_figures: bool = True
    ) -> None:
        """
        Plot top-10 non-Ce element presence percentages.

        For each characterisation dataset, plot:
            - that dataset
            - reactions
        for the top 10 elements ranked by the characterisation dataset.

        For reactions, plot:
            - reactions
            - all characterisation datasets
        for the top 10 elements ranked by reactions.

        Percentage presence:
            100 * (# rows where element > 0) / (total rows in dataset)
        """

        materials_df = merged_dataframes["all_materials"]
        element_cols = [c for c in ELEMENTS if c in materials_df.columns and c != "Ce"]

        if not element_cols:
            return

        dataset_names = [name for name in merged_dataframes.keys() if name != "all_materials"]
        if "reactions" not in dataset_names:
            return

        char_names = [n for n in dataset_names if n != "reactions"]

        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

        def compute_percentages(df: pd.DataFrame) -> Dict[str, float]:
            if df.empty:
                return {el: 0.0 for el in element_cols}

            total_rows = len(df)
            out = {}
            for el in element_cols:
                if el not in df.columns:
                    out[el] = 0.0
                    continue
                col = pd.to_numeric(df[el], errors="coerce").fillna(0)
                out[el] = 100.0 * float((col > 0).sum()) / float(total_rows)
            return out

        pct_by_dataset = {
            name: compute_percentages(df)
            for name, df in merged_dataframes.items()
            if name != "all_materials"
        }

        def plot_grouped_barh(
            ranking_dataset: str,
            shown_datasets: list[str],
            title: str,
            filename: str,
        ) -> None:
            ranking_pcts = pct_by_dataset[ranking_dataset]

            top_items = sorted(
                [(el, pct) for el, pct in ranking_pcts.items() if pct > 0],
                key=lambda x: x[1],
                reverse=True
            )[:10]

            if not top_items:
                return

            top_elements = [el for el, _ in top_items][::-1]

            n_groups = len(shown_datasets)
            y = np.arange(len(top_elements))
            bar_block = 0.8
            height = bar_block / n_groups

            fig_height = max(4, 0.5 * len(top_elements) + 1.5)
            fig, ax = plt.subplots(figsize=(9, fig_height))

            for i, ds_name in enumerate(shown_datasets):
                offset = (i - (n_groups - 1) / 2) * height
                values = [pct_by_dataset[ds_name].get(el, 0.0) for el in top_elements]
                ax.barh(
                    y + offset,
                    values,
                    height=height,
                    label=ds_name
                )

            ax.set_yticks(y)
            ax.set_yticklabels(top_elements)
            ax.set_xlabel("Percentage of materials (%)")
            ax.set_ylabel("Element")
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()

            if output_path is not None:
                fig.savefig(output_path / filename, dpi=300, bbox_inches="tight")

            if show_figures:
                plt.show()
            else:
                plt.close(fig)

        # Characterisation plots: each one vs reactions
        for char_name in char_names:
            if merged_dataframes[char_name].empty:
                continue

            plot_grouped_barh(
                ranking_dataset=char_name,
                shown_datasets=[char_name, "reactions"],
                title=f"Top 10 Dopants in {char_name} vs reactions",
                filename=f"top10_non_ce_elements_{char_name}_vs_reactions.png",
            )

        # Reactions plot: reactions vs all characterisation datasets
        shown_for_reactions = ["reactions"] + [n for n in char_names if not merged_dataframes[n].empty]
        if shown_for_reactions:
            plot_grouped_barh(
                ranking_dataset="reactions",
                shown_datasets=shown_for_reactions,
                title="Top 10 Dopants in reactions vs characterisation datasets",
                filename="top10_non_ce_elements_reactions_vs_characterisation.png",
            )

    def material_overlap_venn_diagram(self, merged_dataframes: Dict[str, pd.DataFrame], save_path: Optional[Path]=None):
        sets = {}

        for key, df in merged_dataframes.items():
            if key == "co2_tpd" or key == 'all_materials':
                continue 
            try:
                n = "_id_material" if key == "reactions" else "_id"
                sets[key] = set(df[n].values)
            except:
                continue

        if save_path is None:
            venny4py(sets=sets)
        else:
            venny4py(sets=sets, out=save_path)
            plt.close()
                
    def pca_overlap(
        self,
        merged_dataframes: Dict[str, pd.DataFrame],
        feature_cols: Optional[List[str]] = None,
        output_path: Optional[Path] = None,
        show_figures: bool = True,
        pca_on_reaction_mats_only: bool = True
    ):
        if feature_cols is None:
            feature_cols = []

        if pca_on_reaction_mats_only:
            reference_name = "reactions"
            reference_label = "Light Off"
        else:
            reference_name = "all_materials"
            reference_label = "All Materials"

        if reference_name not in merged_dataframes:
            raise KeyError(
                f"Reference dataframe '{reference_name}' not found. "
                f"Available keys: {list(merged_dataframes.keys())}"
            )

        reference_df = merged_dataframes[reference_name].copy()

        feature_cols = [c for c in feature_cols if c in reference_df.columns]
        if not feature_cols:
            raise ValueError(
                f"You probably forgot to pass feature_cols or they don't match the columns "
                f"in the {reference_name} dataframe."
            )

        def prepare_features(
            df: pd.DataFrame,
            columns: list[str],
            fill_values: Optional[pd.Series] = None,
            dataset_name: Optional[str] = None,
            print_nans: bool = False
        ):
            X = df.reindex(columns=columns, fill_value=0).copy()

            # force numeric; non-numeric values become NaN
            for col in columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

            # 👉 PRINT NaNs BEFORE filling
            if print_nans:
                nan_counts = X.isna().sum()
                nan_cols = nan_counts[nan_counts > 0].sort_values(ascending=False)

                if not nan_cols.empty:
                    print(f"\nNaNs in {dataset_name}:")
                    for col, count in nan_cols.items():
                        print(f"{col}: {count} ({count / len(X):.2%})")

            if fill_values is None:
                # fit imputation values on reference set only
                fill_values = X.median(axis=0, numeric_only=True)
                fill_values = fill_values.fillna(0)

            X = X.fillna(fill_values)
            return X, fill_values

        # Fit imputation/scaler/PCA on chosen reference population
        reference_X, fill_values = prepare_features(reference_df, feature_cols, fill_values=None)

        scaler = StandardScaler()
        scaled_reference_X = scaler.fit_transform(reference_X)

        pca = PCA(n_components=2)
        reference_pca = pca.fit_transform(scaled_reference_X)

        def transform_dataset(name: str):
            df = merged_dataframes[name]
            if df.empty:
                return None
            X, _ = prepare_features(df, feature_cols, fill_values=fill_values)
            scaled = scaler.transform(X)
            return pca.transform(scaled)

        preferred_order = ["reactions", "h2_tpr", "o2_tpd", "co2_tpd", "osc"]
        color_map = {
            "reactions": "tab:blue",
            "h2_tpr": "tab:red",
            "o2_tpd": "tab:green",
            "co2_tpd": "tab:cyan",
            "osc": "tab:pink",
        }
        label_map = {
            "reactions": "Light Off",
            "h2_tpr": "H2-TPR",
            "o2_tpd": "O2-TPD",
            "co2_tpd": "CO2-TPD",
            "osc": "OSC",
        }

        dataset_keys = [
            k for k in preferred_order
            if k in merged_dataframes and k != reference_name and not merged_dataframes[k].empty
        ]

        if not dataset_keys:
            return scaler, pca

        transformed = {
            k: v
            for k in dataset_keys
            if (v := transform_dataset(k)) is not None
        }

        # If PCA is fit on all_materials, it is useful to also plot reactions if present
        reactions_pca = None
        if not pca_on_reaction_mats_only and "reactions" in merged_dataframes:
            reactions_pca = transform_dataset("reactions")

        n = len(dataset_keys)
        ncols = 2 if n > 1 else 1
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 5 * nrows),
            sharex=True,
            sharey=True
        )

        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for ax, key in zip(axes, dataset_keys):
            data = transformed[key]
            color = color_map.get(key, "tab:orange")
            name = label_map.get(key, key)

            ax.scatter(
                reference_pca[:, 0],
                reference_pca[:, 1],
                c="lightgrey",
                alpha=0.4,
                label=reference_label,
                s=30,
            )

            if reactions_pca is not None and key != "reactions":
                ax.scatter(
                    reactions_pca[:, 0],
                    reactions_pca[:, 1],
                    c="tab:blue",
                    alpha=0.5,
                    label="Light Off",
                    s=30,
                )

            if len(data) > 0:
                ax.scatter(
                    data[:, 0],
                    data[:, 1],
                    c=color,
                    alpha=0.8,
                    label=name,
                    s=40,
                )

            if pca_on_reaction_mats_only:
                ax.set_title(f"{name} vs Light-Off PCA space")
            else:
                ax.set_title(f"{name} vs All-Materials PCA space")

            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.legend()

        for ax in axes[len(dataset_keys):]:
            ax.axis("off")

        fig.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            filename = (
                "pca_overlap_reaction_space.png"
                if pca_on_reaction_mats_only
                else "pca_overlap_all_materials_space.png"
            )
            fig.savefig(output_path / filename, dpi=300, bbox_inches="tight")

        if show_figures:
            plt.show()
        else:
            plt.close(fig)

        return scaler, pca