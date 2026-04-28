from src.db import Database
from src.constants import ELEMENTS
import pandas as pd
from typing import Tuple, Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np
import pickle
import re
import torch


class Preprocessor:
    def __init__(self, database: Database):
        self.database = database

    def get_base_dataframes(
        self,
        config: Optional[Dict] = None,
        base_material_filter: Optional[Dict] = None,
        base_reaction_filter: Optional[Dict] = None
    ):
        
        if config is not None:
            mat_filter = config["material"].get("base_filter", {})
            rxn_filter = config.get("reaction", {}).get("base_filter", None)
            tpr_filter = config.get("h2_tpr_peaks", {}).get("base_filter", None)
            otpd_filter = config.get("o2_tpd_peaks", {}).get("base_filter", None)
            cotpd_filter = config.get("co2_tpd_peaks", {}).get("base_filter", None)
            osc_filter = config.get("osc", {}).get("base_filter", None)
            return {
                "materials": pd.DataFrame(list(self.database.collections["materials"].find(mat_filter))),
                "reactions": pd.DataFrame(list(self.database.collections["reactions"].find(rxn_filter))) if rxn_filter is not None else None,
                "h2_tpr_peaks": pd.DataFrame(list(self.database.collections["h2_tpr_peaks"].find(tpr_filter))) if tpr_filter is not None else None,
                "o2_tpd_peaks": pd.DataFrame(list(self.database.collections["o2_tpd_peaks"].find(otpd_filter))) if otpd_filter is not None else None,
                "co2_tpd_peaks": pd.DataFrame(list(self.database.collections["co2_tpd_peaks"].find(cotpd_filter))) if cotpd_filter is not None else None,
                "osc": pd.DataFrame(list(self.database.collections["osc"].find(osc_filter))) if osc_filter is not None else None,
            }

        if base_material_filter is None:     
            base_material_filter = {}
        if base_reaction_filter is None:
            base_reaction_filter = {}

        all_materials = pd.DataFrame(list(self.database.collections["materials"].find(base_material_filter)))
        all_reactions = pd.DataFrame(list(self.database.collections["reactions"].find(base_reaction_filter)))
        return all_materials, all_reactions

    def add_synthesis_method_flags(
        self,
        materials_df: pd.DataFrame,
        synthesis_col: str = "synthesis",
        drop_original_synthesis_col: bool = False,
        method_map: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        materials_df = materials_df.copy()

        if method_map is None:
            method_map = {
                "solid_state": ["solid_state", "solid state", "solid-state"],
                "sol_gel": ["sol_gel", "sol-gel", "sol gel", "pechini"],
                "hydrothermal": ["hydrothermal"],
                "precipitation": ["precipitation", "co-precipitation", "coprecipitation", "co precipitation"],
                "pyrolysis": ["pyrolysis", "spray pyrolysis", "flame pyrolysis", "combustion"],
            }

        stats = {
            "synthesis_column_used": synthesis_col,
            "drop_original_synthesis_col": drop_original_synthesis_col,
            "synthesis_method_columns_added": list(method_map.keys()),
            "rows_with_missing_synthesis": 0,
            "positive_counts_by_method": {k: 0 for k in method_map},
        }

        if synthesis_col not in materials_df.columns:
            for method_name in method_map:
                materials_df[method_name] = 0
            stats["rows_with_missing_synthesis"] = len(materials_df)
            return materials_df, stats

        synthesis_series = materials_df[synthesis_col].fillna("").astype(str).str.strip().str.lower()
        stats["rows_with_missing_synthesis"] = int((synthesis_series == "").sum())

        for method_name, aliases in method_map.items():
            alias_pattern = "|".join(re.escape(a.lower()) for a in aliases)
            materials_df[method_name] = synthesis_series.str.contains(
                alias_pattern,
                regex=True,
                na=False
            ).astype(int)
            stats["positive_counts_by_method"][method_name] = int(materials_df[method_name].sum())

        if drop_original_synthesis_col and synthesis_col in materials_df.columns:
            materials_df = materials_df.drop(columns=[synthesis_col])

        return materials_df, stats

    def process_element_columns(
        self,
        materials_df: pd.DataFrame,
        minimum_Ce_content: float = 0.0,
        allowed_elements: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Process only element columns that are already meaningfully present in the dataframe.
        - coerce those columns to numeric
        - fill missing values in those columns with 0
        - optionally drop rows containing disallowed elements
        - normalize element fractions row-wise to sum to 1
        - optionally apply a minimum normalized Ce-content filter
        """
        materials_df = materials_df.copy()

        candidate_element_cols = [el for el in ELEMENTS if el in materials_df.columns]

        element_cols_present_before_processing = []
        element_cols_ignored_because_empty = []

        for el in candidate_element_cols:
            numeric_col = pd.to_numeric(materials_df[el], errors="coerce")
            has_any_real_content = ((numeric_col.notna()) & (numeric_col != 0)).any()

            if has_any_real_content:
                element_cols_present_before_processing.append(el)
            else:
                element_cols_ignored_because_empty.append(el)

        element_cols = element_cols_present_before_processing.copy()

        number_of_missing_element_values_defaulted_to_zero = {}
        for el in element_cols:
            numeric_col = pd.to_numeric(materials_df[el], errors="coerce")
            number_of_missing_element_values_defaulted_to_zero[el] = int(numeric_col.isna().sum())
            materials_df[el] = numeric_col.fillna(0.0)

        rows_dropped_due_to_disallowed_elements = 0
        rows_with_nonzero_disallowed_element = {}
        disallowed_elements_checked = []

        if allowed_elements is not None:
            allowed_elements = list(dict.fromkeys(allowed_elements))
            disallowed_elements_checked = [el for el in element_cols if el not in allowed_elements]

            if disallowed_elements_checked:
                disallowed_mask_df = materials_df[disallowed_elements_checked].gt(0)

                rows_with_nonzero_disallowed_element = {
                    el: int(disallowed_mask_df[el].sum()) for el in disallowed_elements_checked
                }

                allowed_mask = ~disallowed_mask_df.any(axis=1)
                rows_dropped_due_to_disallowed_elements = int((~allowed_mask).sum())
                materials_df = materials_df.loc[allowed_mask].copy()

        rows_dropped_due_to_zero_total_element_content = 0
        rows_dropped_due_to_minimum_Ce_content = 0

        if element_cols:
            element_sum_before_normalization = materials_df[element_cols].sum(axis=1)
            zero_sum_mask = element_sum_before_normalization <= 0
            rows_dropped_due_to_zero_total_element_content = int(zero_sum_mask.sum())

            if rows_dropped_due_to_zero_total_element_content > 0:
                materials_df = materials_df.loc[~zero_sum_mask].copy()
                element_sum_before_normalization = materials_df[element_cols].sum(axis=1)

            if len(materials_df) > 0:
                materials_df[element_cols] = materials_df[element_cols].div(element_sum_before_normalization, axis=0)

            if "Ce" in element_cols and minimum_Ce_content > 0:
                ce_mask = materials_df["Ce"] >= minimum_Ce_content
                rows_dropped_due_to_minimum_Ce_content = int((~ce_mask).sum())
                materials_df = materials_df.loc[ce_mask].copy()

        stats = {
            "element_columns_processed": element_cols,
            "element_columns_ignored_because_empty": element_cols_ignored_because_empty,
            "number_of_missing_element_values_defaulted_to_zero": number_of_missing_element_values_defaulted_to_zero,
            "allowed_elements": allowed_elements,
            "disallowed_elements_checked": disallowed_elements_checked,
            "rows_with_nonzero_disallowed_element": rows_with_nonzero_disallowed_element,
            "rows_dropped_due_to_disallowed_elements": rows_dropped_due_to_disallowed_elements,
            "rows_dropped_due_to_zero_total_element_content": rows_dropped_due_to_zero_total_element_content,
            "minimum_Ce_content": minimum_Ce_content,
            "rows_dropped_due_to_minimum_Ce_content": rows_dropped_due_to_minimum_Ce_content,
        }

        return materials_df, stats

    def preprocess_materials(
        self,
        materials_df: pd.DataFrame,
        config: Optional[Dict] = None,
        default_to_mean_cols: Optional[List[str]] = None,
        default_to_zero_cols: Optional[List[str]] = None,
        cannot_be_zero_or_none: Optional[List[str]] = None,
        minimum_Ce_content: Optional[float] = None,
        add_synthesis_flags: Optional[bool] = None,
        synthesis_col: Optional[str] = None,
        drop_original_synthesis_col: Optional[bool] = None,
        synthesis_method_map: Optional[Dict[str, List[str]]] = None,
        allowed_elements: Optional[List[str]] = None,
        allow_supported_samples: Optional[bool] = None,
        default_deposited_fields_to_zero: Optional[bool] = None,
        convert_phase_flag_to_binary: Optional[bool] = None,
        phase_flag_output_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:

        resolved = self._resolve_from_config(
            config,
            "material",
            default_to_mean_cols=default_to_mean_cols,
            default_to_zero_cols=default_to_zero_cols,
            cannot_be_zero_or_none=cannot_be_zero_or_none,
            minimum_Ce_content=minimum_Ce_content,
            add_synthesis_flags=add_synthesis_flags,
            synthesis_col=synthesis_col,
            drop_original_synthesis_col=drop_original_synthesis_col,
            synthesis_method_map=synthesis_method_map,
            allowed_elements=allowed_elements,
            allow_supported_samples=allow_supported_samples,
            default_deposited_fields_to_zero=default_deposited_fields_to_zero,
            convert_phase_flag_to_binary=convert_phase_flag_to_binary,
            phase_flag_output_col=phase_flag_output_col,
        )

        default_to_mean_cols = resolved["default_to_mean_cols"]
        default_to_zero_cols = resolved["default_to_zero_cols"]
        cannot_be_zero_or_none = resolved["cannot_be_zero_or_none"]
        minimum_Ce_content = resolved["minimum_Ce_content"]
        add_synthesis_flags = resolved["add_synthesis_flags"]
        synthesis_col = resolved["synthesis_col"]
        drop_original_synthesis_col = resolved["drop_original_synthesis_col"]
        synthesis_method_map = resolved["synthesis_method_map"]
        allowed_elements = resolved["allowed_elements"]
        allow_supported_samples = resolved["allow_supported_samples"]
        default_deposited_fields_to_zero = resolved["default_deposited_fields_to_zero"]
        convert_phase_flag_to_binary = resolved["convert_phase_flag_to_binary"]
        phase_flag_output_col = resolved["phase_flag_output_col"]

        if default_to_mean_cols is None:
            default_to_mean_cols = ["calcination_time"]
        if default_to_zero_cols is None:
            default_to_zero_cols = ["pretreatment_temp", "pretreatment_o2_vol%", "pretreatment_time_h"]
        if cannot_be_zero_or_none is None:
            cannot_be_zero_or_none = ["calcination_temp", "Sbet"]
        if minimum_Ce_content is None:
            minimum_Ce_content = 0.0
        if add_synthesis_flags is None:
            add_synthesis_flags = False
        if synthesis_col is None:
            synthesis_col = "synthesis"
        if drop_original_synthesis_col is None:
            drop_original_synthesis_col = False
        if allow_supported_samples is None:
            allow_supported_samples = True
        if default_deposited_fields_to_zero is None:
            default_deposited_fields_to_zero = True
        if convert_phase_flag_to_binary is None:
            convert_phase_flag_to_binary = True
        if phase_flag_output_col is None:
            phase_flag_output_col = "single_phase_binary"


        if default_to_mean_cols is None:
            default_to_mean_cols = ["calcination_time"]
        if default_to_zero_cols is None:
            default_to_zero_cols = ["pretreatment_temp", "pretreatment_o2_vol%", "pretreatment_time_h"]
        if cannot_be_zero_or_none is None:
            cannot_be_zero_or_none = ["calcination_temp", "Sbet"]

        materials_df = materials_df.copy()
        n_before = len(materials_df)
        materials_df.replace("", pd.NA, inplace=True)

        deposit_stats = None
        materials_df, deposit_stats = self.process_deposit_fields(
            materials_df=materials_df,
            allow_supported_samples=allow_supported_samples,
            default_deposited_fields_to_zero=default_deposited_fields_to_zero,
        )

        phase_flag_stats = None
        if convert_phase_flag_to_binary:
            materials_df, phase_flag_stats = self.process_phase_flag(
                materials_df=materials_df,
                phase_flag_col="phase_flag",
                output_col=phase_flag_output_col,
            )

        synthesis_stats = None
        if add_synthesis_flags:
            materials_df, synthesis_stats = self.add_synthesis_method_flags(
                materials_df=materials_df,
                synthesis_col=synthesis_col,
                drop_original_synthesis_col=drop_original_synthesis_col,
                method_map=synthesis_method_map,
            )

        materials_df, element_stats = self.process_element_columns(
            materials_df=materials_df,
            minimum_Ce_content=minimum_Ce_content,
            allowed_elements=allowed_elements,
        )

        number_of_values_defaulted_to_mean = {}
        number_of_values_defaulted_to_zero = {}

        for col in default_to_mean_cols:
            if col in materials_df.columns:
                n_missing = int(materials_df[col].isna().sum())
                number_of_values_defaulted_to_mean[col] = n_missing
                if n_missing > 0:
                    materials_df[col] = materials_df[col].fillna(materials_df[col].mean())
            else:
                number_of_values_defaulted_to_mean[col] = 0

        for col in default_to_zero_cols:
            if col in materials_df.columns:
                n_missing = int(materials_df[col].isna().sum())
                number_of_values_defaulted_to_zero[col] = n_missing
                if n_missing > 0:
                    materials_df[col] = materials_df[col].fillna(0)
            else:
                number_of_values_defaulted_to_zero[col] = 0


        dropped_due_to_missing_values = {}
        dropped_due_to_zero_or_negative_values = {}

        for col in cannot_be_zero_or_none:
            if col not in materials_df.columns:
                dropped_due_to_missing_values[col] = len(materials_df)
                dropped_due_to_zero_or_negative_values[col] = 0
                continue

            col_series = materials_df[col]

            dropped_due_to_missing_values[col] = int(col_series.isna().sum())

            if pd.api.types.is_numeric_dtype(col_series):
                dropped_due_to_zero_or_negative_values[col] = int(
                    col_series.notna().sum() - col_series.gt(0).sum()
                )
            else:
                dropped_due_to_zero_or_negative_values[col] = int(
                    (col_series.fillna("").astype(str).str.strip() == "").sum()
                )

        existing_critical_cols = [col for col in cannot_be_zero_or_none if col in materials_df.columns]
        if existing_critical_cols:
            materials_df = materials_df.dropna(subset=existing_critical_cols)
            mask = pd.Series(True, index=materials_df.index)

            for col in existing_critical_cols:
                if pd.api.types.is_numeric_dtype(materials_df[col]):
                    mask &= materials_df[col].notna() & materials_df[col].gt(0)
                else:
                    mask &= materials_df[col].fillna("").astype(str).str.strip().ne("")

            materials_df = materials_df[mask].copy()

        stats = {
            "total_rows_before_preprocessing": n_before,
            "total_rows_after_preprocessing": len(materials_df),
            "total_rows_dropped": n_before - len(materials_df),
            "columns_filled_with_mean": default_to_mean_cols,
            "number_of_values_defaulted_to_mean": number_of_values_defaulted_to_mean,
            "columns_filled_with_zero": default_to_zero_cols,
            "number_of_values_defaulted_to_zero": number_of_values_defaulted_to_zero,
            "dropped_due_to_missing_values": dropped_due_to_missing_values,
            "dropped_due_to_zero_or_negative_values": dropped_due_to_zero_or_negative_values,
            "cannot_be_zero_or_none": cannot_be_zero_or_none,
            "add_synthesis_flags": add_synthesis_flags,
            "synthesis_stats": synthesis_stats,
            "element_stats": element_stats,
            "allow_supported_samples": allow_supported_samples,
            "default_deposited_fields_to_zero": default_deposited_fields_to_zero,
            "deposit_stats": deposit_stats,
            "convert_phase_flag_to_binary": convert_phase_flag_to_binary,
            "phase_flag_stats": phase_flag_stats
        }

        return materials_df, stats
    
    def process_deposit_fields(
        self,
        materials_df: pd.DataFrame,
        allow_supported_samples: bool = True,
        default_deposited_fields_to_zero: bool = True,
        deposit_col: str = "deposit",
        deposited_prefix: str = "deposited_",
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Handle supported/deposited samples.

        Behaviour
        ---------
        - If allow_supported_samples is False:
            drop all rows where `deposit_col` is not null.
        - If allow_supported_samples is True:
            flatten keys from the deposit dict into columns prefixed with `deposited_`.
        - If default_deposited_fields_to_zero is True:
            for rows with null deposit, set all flattened deposited_* columns to 0.

        Notes
        -----
        - Only keys actually observed in non-null deposit dicts are flattened.
        - Nested deposit dicts are not recursively flattened.
        """
        materials_df = materials_df.copy()

        stats = {
            "allow_supported_samples": allow_supported_samples,
            "default_deposited_fields_to_zero": default_deposited_fields_to_zero,
            "deposit_column_used": deposit_col,
            "deposit_prefix": deposited_prefix,
            "rows_with_nonnull_deposit_before_processing": 0,
            "rows_dropped_due_to_nonnull_deposit": 0,
            "deposited_columns_created": [],
            "number_of_missing_deposited_values_defaulted_to_zero": {},
        }

        if deposit_col not in materials_df.columns:
            return materials_df, stats

        deposit_nonnull_mask = materials_df[deposit_col].notna()
        stats["rows_with_nonnull_deposit_before_processing"] = int(deposit_nonnull_mask.sum())

        if not allow_supported_samples:
            stats["rows_dropped_due_to_nonnull_deposit"] = int(deposit_nonnull_mask.sum())
            materials_df = materials_df.loc[~deposit_nonnull_mask].copy()
            return materials_df, stats

        # Collect all keys seen across non-null deposit dicts
        deposit_keys = set()
        for val in materials_df.loc[deposit_nonnull_mask, deposit_col]:
            if isinstance(val, dict):
                deposit_keys.update(val.keys())

        deposit_keys = sorted(deposit_keys)
        deposited_cols = [f"{deposited_prefix}{k}" for k in deposit_keys]
        stats["deposited_columns_created"] = deposited_cols

        if not deposit_keys:
            return materials_df, stats

        # Flatten deposit dicts into deposited_* columns
        for key in deposit_keys:
            new_col = f"{deposited_prefix}{key}"
            materials_df[new_col] = materials_df[deposit_col].apply(
                lambda d: d.get(key, pd.NA) if isinstance(d, dict) else pd.NA
            )

        # Default missing deposited fields to zero if requested
        if default_deposited_fields_to_zero:
            for col in deposited_cols:
                n_missing = int(materials_df[col].isna().sum())
                stats["number_of_missing_deposited_values_defaulted_to_zero"][col] = n_missing
                if n_missing > 0:
                    materials_df[col] = materials_df[col].fillna(0)
        else:
            for col in deposited_cols:
                stats["number_of_missing_deposited_values_defaulted_to_zero"][col] = 0

        return materials_df, stats

    def preprocess_reactions(
        self,
        reactions_df: pd.DataFrame,
        config: Optional[Dict] = None,
        oxygen_content_in_air: Optional[float] = None,
        convert_ghsv_to_whsv_with_assumed_density: Optional[float] = None,
        default_to_mean_cols: Optional[List[str]] = None,
        cannot_be_zero_or_none: Optional[List[str]] = None,
        must_be_zero_or_none: Optional[List[str]] = None,
        minimum_number_datapoints: Optional[int] = None,
        convert_percentages_to_fractions: Optional[bool] = None,
        maximum_fractional_conversion: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        
        resolved = self._resolve_from_config(
            config,
            "reaction",
            oxygen_content_in_air=oxygen_content_in_air,
            convert_ghsv_to_whsv_with_assumed_density=convert_ghsv_to_whsv_with_assumed_density,
            default_to_mean_cols=default_to_mean_cols,
            cannot_be_zero_or_none=cannot_be_zero_or_none,
            must_be_zero_or_none=must_be_zero_or_none,
            minimum_number_datapoints=minimum_number_datapoints,
            convert_percentages_to_fractions=convert_percentages_to_fractions,
            maximum_fractional_conversion=maximum_fractional_conversion,
        )

        oxygen_content_in_air = resolved["oxygen_content_in_air"]
        convert_ghsv_to_whsv_with_assumed_density = resolved["convert_ghsv_to_whsv_with_assumed_density"]
        default_to_mean_cols = resolved["default_to_mean_cols"]
        cannot_be_zero_or_none = resolved["cannot_be_zero_or_none"]
        must_be_zero_or_none = resolved["must_be_zero_or_none"]
        minimum_number_datapoints = resolved["minimum_number_datapoints"]
        convert_percentages_to_fractions = resolved["convert_percentages_to_fractions"]
        maximum_fractional_conversion = resolved["maximum_fractional_conversion"]

        if oxygen_content_in_air is None:
            oxygen_content_in_air = 0.2
        if default_to_mean_cols is None:
            default_to_mean_cols = []
        if cannot_be_zero_or_none is None:
            cannot_be_zero_or_none = []
        if must_be_zero_or_none is None:
            must_be_zero_or_none = []
        if minimum_number_datapoints is None:
            minimum_number_datapoints = 1
        if convert_percentages_to_fractions is None:
            convert_percentages_to_fractions = False

        reactions_df = reactions_df.copy()
        reactions_df.replace("", pd.NA, inplace=True)

        n_before = len(reactions_df)

        stats = {
            "total_rows_before_preprocessing": n_before,
            "oxygen_content_in_air_used": oxygen_content_in_air,
            "convert_ghsv_to_whsv_with_assumed_density_used": convert_ghsv_to_whsv_with_assumed_density,
            "convert_percentages_to_fractions": convert_percentages_to_fractions,
            "maximum_fractional_conversion": maximum_fractional_conversion,
        }

        gas_o2_content_added_from_air = {
            "rows_with_air_content": 0,
            "rows_where_o2_was_missing_before": 0,
            "rows_where_o2_was_present_before": 0,
        }

        if oxygen_content_in_air is not None and "gas_air_content" in reactions_df.columns:
            air_mask = reactions_df["gas_air_content"].notna()

            if "gas_o2_content" not in reactions_df.columns:
                reactions_df["gas_o2_content"] = pd.NA

            o2_missing_mask = reactions_df["gas_o2_content"].isna()

            gas_o2_content_added_from_air["rows_with_air_content"] = int(air_mask.sum())
            gas_o2_content_added_from_air["rows_where_o2_was_missing_before"] = int((air_mask & o2_missing_mask).sum())
            gas_o2_content_added_from_air["rows_where_o2_was_present_before"] = int((air_mask & ~o2_missing_mask).sum())

            reactions_df["gas_o2_content"] = (
                reactions_df["gas_o2_content"].fillna(0)
                + reactions_df["gas_air_content"].fillna(0) * oxygen_content_in_air
            )

        stats["gas_o2_content_added_from_air"] = gas_o2_content_added_from_air

        flow_conversion_stats = {
            "rows_converted": 0,
            "rows_where_flow_missing_before": 0,
            "rows_where_flow_overwritten": 0,
            "assumed_density_used": convert_ghsv_to_whsv_with_assumed_density,
        }

        if (
            convert_ghsv_to_whsv_with_assumed_density is not None
            and "flow_h-1" in reactions_df.columns
        ):
            ghsv_mask = reactions_df["flow_h-1"].notna()

            if "flow_mL_h_g" not in reactions_df.columns:
                reactions_df["flow_mL_h_g"] = pd.NA

            flow_missing_mask = reactions_df["flow_mL_h_g"].isna()

            flow_conversion_stats["rows_converted"] = int(ghsv_mask.sum())
            flow_conversion_stats["rows_where_flow_missing_before"] = int((ghsv_mask & flow_missing_mask).sum())
            flow_conversion_stats["rows_where_flow_overwritten"] = int((ghsv_mask & ~flow_missing_mask).sum())

            reactions_df.loc[ghsv_mask, "flow_mL_h_g"] = (
                reactions_df.loc[ghsv_mask, "flow_h-1"]
                / convert_ghsv_to_whsv_with_assumed_density
            )

        stats["flow_mL_h_g_defaulted_from_flow_h_minus_1"] = flow_conversion_stats

        # -----------------------------------------
        # Convert percentage-based quantities to fractions
        # -----------------------------------------
        percentage_conversion_stats = {
            "gas_co_content_rows_converted": 0,
            "gas_o2_content_rows_converted": 0,
            "conversion_rows_converted": 0,
            "conversion_points_seen": 0,
            "conversion_points_clipped": 0,
        }

        if convert_percentages_to_fractions:
            for col in ["gas_co_content", "gas_o2_content"]:
                if col in reactions_df.columns:
                    mask = reactions_df[col].notna()
                    percentage_conversion_stats[f"{col}_rows_converted"] = int(mask.sum())
                    reactions_df.loc[mask, col] = pd.to_numeric(
                        reactions_df.loc[mask, col], errors="coerce"
                    ) / 100.0

            if "conversion" in reactions_df.columns:
                def _convert_conversion_list(vals):
                    if not isinstance(vals, (list, tuple)):
                        return vals, 0, 0

                    arr = pd.to_numeric(pd.Series(vals), errors="coerce").to_numpy(dtype=float)
                    n_points = len(arr)
                    n_clipped = 0

                    arr = arr / 100.0

                    if maximum_fractional_conversion is not None:
                        n_clipped = int(np.sum(arr > maximum_fractional_conversion))
                        arr = np.minimum(arr, maximum_fractional_conversion)

                    return arr.tolist(), n_points, n_clipped

                mask = reactions_df["conversion"].apply(lambda x: isinstance(x, (list, tuple)))
                percentage_conversion_stats["conversion_rows_converted"] = int(mask.sum())

                processed = reactions_df.loc[mask, "conversion"].apply(_convert_conversion_list)
                reactions_df.loc[mask, "conversion"] = processed.apply(lambda x: x[0])

                percentage_conversion_stats["conversion_points_seen"] = int(
                    processed.apply(lambda x: x[1]).sum()
                )
                percentage_conversion_stats["conversion_points_clipped"] = int(
                    processed.apply(lambda x: x[2]).sum()
                )

        stats["percentage_conversion"] = percentage_conversion_stats

        number_of_values_defaulted_to_mean = {}
        for col in default_to_mean_cols:
            if col not in reactions_df.columns:
                number_of_values_defaulted_to_mean[col] = 0
                continue

            n_missing = int(reactions_df[col].isna().sum())
            number_of_values_defaulted_to_mean[col] = n_missing

            if n_missing > 0:
                reactions_df[col] = reactions_df[col].fillna(reactions_df[col].mean())

        stats["columns_filled_with_mean"] = default_to_mean_cols
        stats["number_of_values_defaulted_to_mean"] = number_of_values_defaulted_to_mean

        dropped_due_to_missing_values = {}
        dropped_due_to_zero_or_negative_values = {}

        for col in cannot_be_zero_or_none:
            if col not in reactions_df.columns:
                dropped_due_to_missing_values[col] = len(reactions_df)
                dropped_due_to_zero_or_negative_values[col] = 0
                continue

            dropped_due_to_missing_values[col] = int(reactions_df[col].isna().sum())
            dropped_due_to_zero_or_negative_values[col] = int(
                reactions_df[col].notna().sum() - reactions_df[col].gt(0).sum()
            )

        dropped_due_to_nonzero_forbidden_values = {}
        for col in must_be_zero_or_none:
            if col not in reactions_df.columns:
                dropped_due_to_nonzero_forbidden_values[col] = 0
                continue

            dropped_due_to_nonzero_forbidden_values[col] = int(
                reactions_df[col].fillna(0).ne(0).sum()
            )

        if "temps" in reactions_df.columns and "conversion" in reactions_df.columns:
            datapoint_mask = reactions_df.apply(
                lambda row: (
                    isinstance(row["temps"], (list, tuple))
                    and isinstance(row["conversion"], (list, tuple))
                    and min(len(row["temps"]), len(row["conversion"])) >= minimum_number_datapoints
                ),
                axis=1,
            )
        else:
            datapoint_mask = pd.Series(False, index=reactions_df.index)

        rows_dropped_due_to_too_few_datapoints = int((~datapoint_mask).sum())
        keep_mask = pd.Series(True, index=reactions_df.index)

        for col in cannot_be_zero_or_none:
            if col in reactions_df.columns:
                keep_mask &= reactions_df[col].notna() & reactions_df[col].gt(0)
            else:
                keep_mask &= False

        for col in must_be_zero_or_none:
            if col in reactions_df.columns:
                keep_mask &= reactions_df[col].fillna(0).eq(0)

        keep_mask &= datapoint_mask

        reactions_df = reactions_df[keep_mask].copy()

        stats.update({
            "total_rows_after_preprocessing": len(reactions_df),
            "total_rows_dropped": n_before - len(reactions_df),
            "cannot_be_zero_or_none": cannot_be_zero_or_none,
            "must_be_zero_or_none": must_be_zero_or_none,
            "minimum_number_datapoints": minimum_number_datapoints,
            "rows_dropped_due_to_too_few_datapoints": rows_dropped_due_to_too_few_datapoints,
            "dropped_due_to_missing_values": dropped_due_to_missing_values,
            "dropped_due_to_zero_or_negative_values": dropped_due_to_zero_or_negative_values,
            "dropped_due_to_nonzero_forbidden_values": dropped_due_to_nonzero_forbidden_values,
        })

        return reactions_df, stats

    def _resolve_from_config(
        self,
        config: Optional[Dict],
        section: str,
        **kwargs,
    ) -> Dict:
        """
        Resolve preprocessor kwargs from config[section], while letting any explicit
        non-None kwargs override the config.

        Example:
            resolved = self._resolve_from_config(
                config, "reaction",
                oxygen_content_in_air=oxygen_content_in_air,
                default_to_mean_cols=default_to_mean_cols,
            )
        """
        section_cfg = {}
        if config is not None:
            section_cfg = config.get(section, {})

        resolved = {}
        for key, value in kwargs.items():
            if value is not None:
                resolved[key] = value
            else:
                resolved[key] = section_cfg.get(key, None)

        return resolved

    def merge_materials_and_reactions(
        self,
        materials_df: pd.DataFrame,
        reactions_df: pd.DataFrame,
        drop_missing_rxns: bool = True,
        row_by_datapoint: bool = False
    ) -> Tuple[pd.DataFrame, Dict]:

        materials_df = materials_df.copy()
        reactions_df = reactions_df.copy()

        n_materials_before = len(materials_df)
        n_reactions_before = len(reactions_df)

        if "material_id" not in materials_df.columns:
            if "_id" in materials_df.columns:
                materials_df["material_id"] = materials_df["_id"]
            else:
                raise KeyError("materials_df must contain either 'material_id' or '_id'")

        if "material_id" not in reactions_df.columns:
            raise KeyError("reactions_df must contain 'material_id'")

        material_ids_before = set(materials_df["material_id"].dropna())
        reaction_material_ids_before = set(reactions_df["material_id"].dropna())

        materials_without_reactions = material_ids_before - reaction_material_ids_before
        reactions_without_materials = reaction_material_ids_before - material_ids_before
        matched_material_ids = material_ids_before & reaction_material_ids_before

        merged_df = materials_df.merge(
            reactions_df,
            on="material_id",
            how="inner",
            suffixes=("_material", "_reaction"),
        )

        stats = {
            "n_materials_before_merge": n_materials_before,
            "n_reactions_before_merge": n_reactions_before,
            "n_unique_material_ids_in_materials": len(material_ids_before),
            "n_unique_material_ids_in_reactions": len(reaction_material_ids_before),
            "n_unique_material_ids_matched": len(matched_material_ids),
            "n_material_ids_dropped_from_materials_due_to_missing_reactions": len(materials_without_reactions),
            "n_material_ids_dropped_from_reactions_due_to_missing_materials": len(reactions_without_materials),
            "n_rows_after_merge": len(merged_df),
            "drop_missing_rxns": drop_missing_rxns,
        }
        if row_by_datapoint:
            merged_df = self.row_by_temperature(merged_df)

        return merged_df, stats

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
    
    def process_phase_flag(
        self,
        materials_df: pd.DataFrame,
        phase_flag_col: str = "phase_flag",
        output_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Convert phase_flag values:
        - 'single' -> 1
        - 'multi'  -> 0

        If output_col is None, overwrite phase_flag in place.
        """
        materials_df = materials_df.copy()

        if output_col is None:
            output_col = phase_flag_col

        stats = {
            "phase_flag_column_used": phase_flag_col,
            "phase_flag_output_column": output_col,
            "rows_with_missing_phase_flag": 0,
            "rows_mapped_single_to_1": 0,
            "rows_mapped_multi_to_0": 0,
            "rows_unmapped_phase_flag": 0,
        }

        if phase_flag_col not in materials_df.columns:
            return materials_df, stats

        raw = materials_df[phase_flag_col]
        norm = raw.fillna("").astype(str).str.strip().str.lower()

        stats["rows_with_missing_phase_flag"] = int((norm == "").sum())
        stats["rows_mapped_single_to_1"] = int((norm == "single").sum())
        stats["rows_mapped_multi_to_0"] = int((norm == "multi").sum())
        stats["rows_unmapped_phase_flag"] = int(
            (~norm.isin(["", "single", "multi"])).sum()
        )

        mapped = norm.map({"single": 1, "multi": 0})

        # Preserve true missing values as pd.NA rather than mapping "" -> NaN float
        mapped = mapped.astype("Int64")

        materials_df[output_col] = mapped

        return materials_df, stats

    def row_by_temperature(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Flattens a merged_df such that each row is a single temperature–conversion pair
        instead of a full light-off curve.

        Assumes:
            - merged_df["temps"] is a list-like
            - merged_df["conversion"] is a list-like of same length

        Returns:
            DataFrame with:
                - one row per (temp, conversion)
                - all other columns duplicated
                - new columns: "temperature", "conversion"
        """

        df = merged_df.copy()
        valid_mask = df.apply(self._valid_row, axis=1)
        df = df[valid_mask].copy()
        df["temp_conv_pairs"] = df.apply(
            lambda row: list(zip(row["temps"], row["conversion"])),
            axis=1
        )
        df = df.explode("temp_conv_pairs", ignore_index=True)
        df[["temperature", "conversion"]] = pd.DataFrame(
            df["temp_conv_pairs"].tolist(),
            index=df.index
        )
        df = df.drop(columns=["temps", "temp_conv_pairs"])
        return df

    def _valid_row(self, row):
        return (
            isinstance(row["temps"], (list, tuple))
            and isinstance(row["conversion"], (list, tuple))
            and len(row["temps"]) == len(row["conversion"])
            and len(row["temps"]) > 0
        )
    
    def element_statistics(
        self,
        materials_df: pd.DataFrame,
        element_cols: Optional[List[str]] = None,
        doi_col: str = "doi",
    ) -> pd.DataFrame:
        """
        Compute element statistics:
            - total number of appearances (rows where element > 0)
            - number of unique papers (unique DOIs where element > 0)

        Returns:
            DataFrame indexed by element with columns:
                - n_materials
                - n_papers
        """

        df = materials_df.copy()

        # Determine which element columns to use
        if element_cols is None:
            element_cols = [col for col in df.columns if col in ELEMENTS]

        stats = []

        for el in element_cols:

            if el not in df.columns:
                continue

            col = pd.to_numeric(df[el], errors="coerce").fillna(0)

            mask = col > 0

            n_materials = int(mask.sum())

            if doi_col in df.columns:
                n_papers = int(df.loc[mask, doi_col].dropna().nunique())
            else:
                n_papers = 0

            stats.append({
                "element": el,
                "n_materials": n_materials,
                "n_papers": n_papers,
            })

        stats_df = pd.DataFrame(stats)

        if not stats_df.empty:
            stats_df = stats_df.sort_values("n_materials", ascending=False).reset_index(drop=True)

        return stats_df
    
    def filter_niche_elements(
        self,
        preprocessed_materials_df: pd.DataFrame,
        min_appearances: int,
        min_papers: int,
        doi_col: str = "doi",
        material_id_col: str = "material_id",
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove rows containing niche elements.

        An element is considered niche if:
        - it appears in fewer than min_appearances unique materials, or
        - it appears in fewer than min_papers unique DOIs

        If the input is a merged dataframe with multiple rows per material,
        element frequencies are computed on unique surviving materials only.
        """
        df = preprocessed_materials_df.copy()

        element_cols = [col for col in df.columns if col in ELEMENTS]

        if material_id_col in df.columns:
            material_level_df = df.drop_duplicates(subset=[material_id_col]).copy()
        else:
            material_level_df = df.copy()

        element_counts_df = self.element_statistics(
            materials_df=material_level_df,
            element_cols=element_cols,
            doi_col=doi_col,
        )

        if element_counts_df.empty:
            stats = {
                "min_appearances": min_appearances,
                "min_papers": min_papers,
                "n_rows_before": len(df),
                "n_rows_after": len(df),
                "n_rows_removed": 0,
                "n_unique_materials_before": material_level_df[material_id_col].nunique() if material_id_col in material_level_df.columns else len(material_level_df),
                "n_unique_materials_after": material_level_df[material_id_col].nunique() if material_id_col in material_level_df.columns else len(material_level_df),
                "removed_elements": [],
                "kept_elements": [],
                "element_value_counts": {},
                "element_statistics": [],
            }
            return df, stats

        niche_mask = (
            (element_counts_df["n_materials"] < min_appearances)
            | (element_counts_df["n_papers"] < min_papers)
        )

        removed_elements = element_counts_df.loc[niche_mask, "element"].tolist()
        kept_elements = element_counts_df.loc[~niche_mask, "element"].tolist()

        element_value_counts = {
            row["element"]: {
                "n_materials": int(row["n_materials"]),
                "n_papers": int(row["n_papers"]),
            }
            for _, row in element_counts_df.iterrows()
        }

        if removed_elements:
            has_removed_element_mask = df[removed_elements].fillna(0).gt(0).any(axis=1)
            filtered_df = df.loc[~has_removed_element_mask].copy()
        else:
            has_removed_element_mask = pd.Series(False, index=df.index)
            filtered_df = df.copy()

        if material_id_col in filtered_df.columns:
            n_unique_materials_after = filtered_df[material_id_col].nunique()
        else:
            n_unique_materials_after = len(filtered_df)

        stats = {
            "min_appearances": min_appearances,
            "min_papers": min_papers,
            "n_rows_before": len(df),
            "n_rows_after": len(filtered_df),
            "n_rows_removed": int(has_removed_element_mask.sum()),
            "n_unique_materials_before": material_level_df[material_id_col].nunique() if material_id_col in material_level_df.columns else len(material_level_df),
            "n_unique_materials_after": n_unique_materials_after,
            "removed_elements": removed_elements,
            "kept_elements": kept_elements,
            "element_value_counts": element_value_counts,
            "element_statistics": element_counts_df.to_dict(orient="records"),
        }

        return filtered_df, stats

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

    def prepare_neural_ode_curves(
        self,
        train_curves_df: pd.DataFrame,
        test_curves_df: pd.DataFrame,
        feature_cols: List[str],
        temp_scaler: Optional[StandardScaler] = None,
        feature_scaler: Optional[StandardScaler] = None,
        use_first_conversion_as_x0: bool = True,
        default_x0: float = 0.0,
        sort_temperatures: bool = True,
        drop_duplicate_temperatures: bool = True,
        insert_0_conv_at_temp: Optional[float] = None,
        insert_0_conv_only_if_below_min_temp: bool = True,
        insert_0_conv_offset_below_min_temp: Optional[float] = None,
        insert_1_conv_offset_above_max_temp: Optional[float] = None,
        insert_1_conv_min_final_conversion: Optional[float] = None,
        insert_1_conv_value: float = 1.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler, Dict]:
        """
        Prepare curve-level dataframes for Neural ODE training.

        Optionally inserts a synthetic anchor point with 0% conversion at a chosen
        temperature, typically below the first observed temperature of a curve.

        Returns
        -------
        train_prepared_df, test_prepared_df, feature_scaler, temp_scaler, stats
        """
        if sum([
            insert_0_conv_at_temp is not None,
            insert_0_conv_offset_below_min_temp is not None
        ]) > 1:
            raise ValueError("Only one insertion mode can be active.")
        train_df = train_curves_df.copy()
        test_df = test_curves_df.copy()

        n_inserted_train = 0
        n_inserted_test = 0

        def _sort_clean_and_optionally_insert(row):
            temps = row["temps"]
            conv = row["conversion"]

            if not isinstance(temps, (list, tuple)) or not isinstance(conv, (list, tuple)):
                return None, None, None, False, False

            if len(temps) != len(conv) or len(temps) == 0:
                return None, None, None, False, False

            temps = np.asarray(temps, dtype=float)
            conv = np.asarray(conv, dtype=float)

            if sort_temperatures:
                order = np.argsort(temps)
                temps = temps[order]
                conv = conv[order]

            if drop_duplicate_temperatures:
                _, unique_idx = np.unique(temps, return_index=True)
                unique_idx = np.sort(unique_idx)
                temps = temps[unique_idx]
                conv = conv[unique_idx]

            if len(temps) == 0:
                return None, None, None, False, False

            # ✅ initialise mask HERE
            observed_mask = np.ones(len(temps), dtype=bool)

            original_min_temp = float(np.min(temps))
            original_max_temp = float(np.max(temps))
            original_final_conv = float(conv[-1])

            inserted_low = False
            inserted_high = False

            # --- LOW anchor ---
            anchor_temp = None

            if insert_0_conv_at_temp is not None:
                candidate = float(insert_0_conv_at_temp)
                if insert_0_conv_only_if_below_min_temp:
                    if candidate < original_min_temp:
                        anchor_temp = candidate
                else:
                    anchor_temp = candidate

            elif insert_0_conv_offset_below_min_temp is not None:
                delta = float(insert_0_conv_offset_below_min_temp)
                candidate = original_min_temp - delta
                if np.isfinite(candidate):
                    anchor_temp = candidate

            if anchor_temp is not None:
                temps = np.insert(temps, 0, anchor_temp)
                conv = np.insert(conv, 0, 0.0)
                observed_mask = np.insert(observed_mask, 0, False)
                inserted_low = True

            # --- HIGH anchor ---
            if insert_1_conv_offset_above_max_temp is not None:
                should_insert = True

                if insert_1_conv_min_final_conversion is not None:
                    should_insert = (
                        original_final_conv >= float(insert_1_conv_min_final_conversion)
                    )

                if should_insert:
                    delta = float(insert_1_conv_offset_above_max_temp)
                    anchor_temp_high = original_max_temp + delta

                    temps = np.append(temps, anchor_temp_high)
                    conv = np.append(conv, float(insert_1_conv_value))
                    observed_mask = np.append(observed_mask, False)
                    inserted_high = True

            # --- FINAL CLEANUP ---
            if sort_temperatures:
                order = np.argsort(temps)
                temps = temps[order]
                conv = conv[order]
                observed_mask = observed_mask[order]

            if drop_duplicate_temperatures:
                _, unique_idx = np.unique(temps, return_index=True)
                unique_idx = np.sort(unique_idx)
                temps = temps[unique_idx]
                conv = conv[unique_idx]
                observed_mask = observed_mask[unique_idx]

            return (
                temps.tolist(),
                conv.tolist(),
                observed_mask.tolist(),
                inserted_low,
                inserted_high,
            )

        for df_name, df in [("train", train_df), ("test", test_df)]:
            cleaned = df.apply(_sort_clean_and_optionally_insert, axis=1)
            df["temps"] = cleaned.apply(lambda x: x[0])
            df["conversion"] = cleaned.apply(lambda x: x[1])
            df["observed_mask"] = cleaned.apply(lambda x: x[2])
            df["inserted_0_conv_anchor"] = cleaned.apply(lambda x: x[3])
            df["inserted_1_conv_anchor"] = cleaned.apply(lambda x: x[4])

            if df_name == "train":
                n_inserted_train = int(df["inserted_0_conv_anchor"].sum())
                n_inserted_high_train = int(df["inserted_1_conv_anchor"].sum())
            else:
                n_inserted_test = int(df["inserted_0_conv_anchor"].sum())
                n_inserted_high_test = int(df["inserted_1_conv_anchor"].sum())

        valid_train = train_df["temps"].notna() & train_df["conversion"].notna()
        valid_test = test_df["temps"].notna() & test_df["conversion"].notna()
        train_df = train_df.loc[valid_train].copy()
        test_df = test_df.loc[valid_test].copy()

        # Fit feature scaler on train only
        if feature_scaler is None:
            feature_scaler = StandardScaler()

        train_df[feature_cols] = feature_scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = feature_scaler.transform(test_df[feature_cols])

        # Fit temperature scaler on all train temperatures only
        if temp_scaler is None:
            temp_scaler = StandardScaler()

        train_temps_flat = np.concatenate(
            [np.asarray(t, dtype=float) for t in train_df["temps"].tolist()]
        )
        temp_scaler.fit(train_temps_flat.reshape(-1, 1))

        def _scale_temps(temp_list):
            arr = np.asarray(temp_list, dtype=float).reshape(-1, 1)
            return temp_scaler.transform(arr).ravel().tolist()

        train_df["temps_scaled"] = train_df["temps"].apply(_scale_temps)
        test_df["temps_scaled"] = test_df["temps"].apply(_scale_temps)

        if use_first_conversion_as_x0:
            train_df["x0"] = train_df["conversion"].apply(lambda x: float(x[0]))
            test_df["x0"] = test_df["conversion"].apply(lambda x: float(x[0]))
        else:
            train_df["x0"] = float(default_x0)
            test_df["x0"] = float(default_x0)

        stats = {
            "n_train_curves_before": len(train_curves_df),
            "n_test_curves_before": len(test_curves_df),
            "n_train_curves_after": len(train_df),
            "n_test_curves_after": len(test_df),
            "feature_cols": feature_cols,
            "use_first_conversion_as_x0": use_first_conversion_as_x0,
            "default_x0": default_x0,
            "sort_temperatures": sort_temperatures,
            "drop_duplicate_temperatures": drop_duplicate_temperatures,
            "insert_0_conv_offset_below_min_temp": insert_0_conv_offset_below_min_temp,
            "insert_0_conv_at_temp": insert_0_conv_at_temp,
            "insert_0_conv_only_if_below_min_temp": insert_0_conv_only_if_below_min_temp,
            "n_train_0_conv_points_inserted": n_inserted_train,
            "n_test_0_conv_points_inserted": n_inserted_test,
            "n_train_1_conv_points_inserted": n_inserted_high_train,
            "n_test_1_conv_points_inserted": n_inserted_high_test,
        }

        return train_df, test_df, feature_scaler, temp_scaler, stats
    
    def prepare_global_range_neural_ode_curves(
        self,
        train_curves_df: pd.DataFrame,
        test_curves_df: pd.DataFrame,
        feature_cols: List[str],
        temp_scaler: Optional[StandardScaler] = None,
        feature_scaler: Optional[StandardScaler] = None,
        sort_temperatures: bool = True,
        drop_duplicate_temperatures: bool = True,
        low_enabled: bool = False,
        low_offset_below_min_temp: Optional[float] = None,
        low_spacing: Optional[float] = None,
        low_lower_bound_temp: float = 0.0,
        low_only_if_first_observed_conversion_below: Optional[float] = None,
        low_value: float = 0.0,
        high_enabled: bool = False,
        high_offset_above_max_temp: Optional[float] = None,
        high_spacing: Optional[float] = None,
        high_upper_bound_temp: Optional[float] = None,
        high_only_if_final_observed_conversion_above: Optional[float] = None,
        high_value: float = 1.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, StandardScaler, Dict]:
        """
        Prepare curve-level dataframes for global-range Neural ODE training.

        This function supports adding multiple synthetic anchor points below the
        first observed temperature and/or above the last observed temperature.

        Low-anchor semantics
        --------------------
        If enabled, the last synthetic point before the first observed point is:
            T_last_low = Tmin_obs - low_offset_below_min_temp

        Then additional low synthetic points are added at regular spacing:
            T_last_low - low_spacing, T_last_low - 2*low_spacing, ...
        down to (and including, if hit exactly) low_lower_bound_temp.

        High-anchor semantics
        ---------------------
        If enabled, the first synthetic point after the last observed point is:
            T_first_high = Tmax_obs + high_offset_above_max_temp

        Then additional high synthetic points are added at regular spacing:
            T_first_high + high_spacing, T_first_high + 2*high_spacing, ...
        up to high_upper_bound_temp if provided.

        Returns
        -------
        train_prepared_df, test_prepared_df, feature_scaler, temp_scaler, stats
        """

        train_df = train_curves_df.copy()
        test_df = test_curves_df.copy()

        if low_enabled:
            if low_offset_below_min_temp is None or low_spacing is None:
                raise ValueError(
                    "If low_enabled=True, both low_offset_below_min_temp and low_spacing must be provided."
                )
            if low_spacing <= 0:
                raise ValueError("low_spacing must be > 0.")
            if low_offset_below_min_temp < 0:
                raise ValueError("low_offset_below_min_temp must be >= 0.")

        if high_enabled:
            if high_offset_above_max_temp is None or high_spacing is None:
                raise ValueError(
                    "If high_enabled=True, both high_offset_above_max_temp and high_spacing must be provided."
                )
            if high_spacing <= 0:
                raise ValueError("high_spacing must be > 0.")
            if high_offset_above_max_temp < 0:
                raise ValueError("high_offset_above_max_temp must be >= 0.")

        def _make_low_synthetic_temps(
            observed_min_temp: float,
        ) -> np.ndarray:
            """
            Build low synthetic temperatures in ascending order.
            Example:
                observed_min_temp = 140
                offset = 30
                spacing = 10
                lower_bound = 0
            gives:
                [0, 10, 20, ..., 100, 110]
            """
            last_low = observed_min_temp - float(low_offset_below_min_temp)
            if not np.isfinite(last_low):
                return np.array([], dtype=float)

            temps_desc = []
            t = last_low
            lb = float(low_lower_bound_temp)

            # Build descending, then reverse to ascending
            while t >= lb - 1e-12:
                temps_desc.append(float(t))
                t -= float(low_spacing)

            temps_asc = np.array(list(reversed(temps_desc)), dtype=float)
            return temps_asc

        def _make_high_synthetic_temps(
            observed_max_temp: float,
        ) -> np.ndarray:
            """
            Build high synthetic temperatures in ascending order.
            Example:
                observed_max_temp = 320
                offset = 20
                spacing = 10
                upper_bound = 380
            gives:
                [340, 350, 360, 370, 380]
            """
            first_high = observed_max_temp + float(high_offset_above_max_temp)
            if not np.isfinite(first_high):
                return np.array([], dtype=float)

            temps = []
            t = first_high

            if high_upper_bound_temp is None:
                temps.append(float(t))
            else:
                ub = float(high_upper_bound_temp)
                while t <= ub + 1e-12:
                    temps.append(float(t))
                    t += float(high_spacing)

            return np.array(temps, dtype=float)

        def _sort_clean_and_insert(row):
            temps = row["temps"]
            conv = row["conversion"]

            if not isinstance(temps, (list, tuple)) or not isinstance(conv, (list, tuple)):
                return None, None, None, 0, 0

            if len(temps) != len(conv) or len(temps) == 0:
                return None, None, None, 0, 0

            temps = np.asarray(temps, dtype=float)
            conv = np.asarray(conv, dtype=float)

            if sort_temperatures:
                order = np.argsort(temps)
                temps = temps[order]
                conv = conv[order]

            if drop_duplicate_temperatures:
                _, unique_idx = np.unique(temps, return_index=True)
                unique_idx = np.sort(unique_idx)
                temps = temps[unique_idx]
                conv = conv[unique_idx]

            if len(temps) == 0:
                return None, None, None, 0, 0

            observed_mask = np.ones(len(temps), dtype=bool)

            observed_min_temp = float(np.min(temps))
            observed_max_temp = float(np.max(temps))
            first_observed_conversion = float(conv[0])
            final_observed_conversion = float(conv[-1])

            low_insert_count = 0
            high_insert_count = 0

            # -----------------------------
            # Low synthetic region
            # -----------------------------
            if low_enabled:
                low_ok = True
                if low_only_if_first_observed_conversion_below is not None:
                    low_ok = first_observed_conversion <= float(
                        low_only_if_first_observed_conversion_below
                    )

                if low_ok:
                    low_temps = _make_low_synthetic_temps(observed_min_temp)

                    # Keep only those strictly below the observed min
                    low_temps = low_temps[low_temps < observed_min_temp - 1e-12]

                    if low_temps.size > 0:
                        low_conv = np.full(low_temps.shape, float(low_value), dtype=float)
                        low_mask = np.zeros(low_temps.shape, dtype=bool)

                        temps = np.concatenate([low_temps, temps])
                        conv = np.concatenate([low_conv, conv])
                        observed_mask = np.concatenate([low_mask, observed_mask])
                        low_insert_count = int(low_temps.size)

            # -----------------------------
            # High synthetic region
            # -----------------------------
            if high_enabled:
                high_ok = True
                if high_only_if_final_observed_conversion_above is not None:
                    high_ok = final_observed_conversion >= float(
                        high_only_if_final_observed_conversion_above
                    )

                if high_ok:
                    high_temps = _make_high_synthetic_temps(observed_max_temp)

                    # Keep only those strictly above the observed max
                    high_temps = high_temps[high_temps > observed_max_temp + 1e-12]

                    if high_temps.size > 0:
                        high_conv = np.full(high_temps.shape, float(high_value), dtype=float)
                        high_mask = np.zeros(high_temps.shape, dtype=bool)

                        temps = np.concatenate([temps, high_temps])
                        conv = np.concatenate([conv, high_conv])
                        observed_mask = np.concatenate([observed_mask, high_mask])
                        high_insert_count = int(high_temps.size)

            # Final cleanup
            if sort_temperatures:
                order = np.argsort(temps)
                temps = temps[order]
                conv = conv[order]
                observed_mask = observed_mask[order]

            if drop_duplicate_temperatures:
                _, unique_idx = np.unique(temps, return_index=True)
                unique_idx = np.sort(unique_idx)
                temps = temps[unique_idx]
                conv = conv[unique_idx]
                observed_mask = observed_mask[unique_idx]

            return (
                temps.tolist(),
                conv.tolist(),
                observed_mask.tolist(),
                low_insert_count,
                high_insert_count,
            )

        n_train_low_inserted = 0
        n_test_low_inserted = 0
        n_train_high_inserted = 0
        n_test_high_inserted = 0

        for df_name, df in [("train", train_df), ("test", test_df)]:
            cleaned = df.apply(_sort_clean_and_insert, axis=1)

            df["temps"] = cleaned.apply(lambda x: x[0])
            df["conversion"] = cleaned.apply(lambda x: x[1])
            df["observed_mask"] = cleaned.apply(lambda x: x[2])
            df["n_low_synthetic_points"] = cleaned.apply(lambda x: x[3])
            df["n_high_synthetic_points"] = cleaned.apply(lambda x: x[4])

            if df_name == "train":
                n_train_low_inserted = int(df["n_low_synthetic_points"].sum())
                n_train_high_inserted = int(df["n_high_synthetic_points"].sum())
            else:
                n_test_low_inserted = int(df["n_low_synthetic_points"].sum())
                n_test_high_inserted = int(df["n_high_synthetic_points"].sum())

        valid_train = (
            train_df["temps"].notna()
            & train_df["conversion"].notna()
            & train_df["observed_mask"].notna()
        )
        valid_test = (
            test_df["temps"].notna()
            & test_df["conversion"].notna()
            & test_df["observed_mask"].notna()
        )
        train_df = train_df.loc[valid_train].copy()
        test_df = test_df.loc[valid_test].copy()

        if feature_scaler is None:
            feature_scaler = StandardScaler()

        train_df[feature_cols] = feature_scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = feature_scaler.transform(test_df[feature_cols])

        if temp_scaler is None:
            temp_scaler = StandardScaler()

        train_temps_flat = np.concatenate(
            [np.asarray(t, dtype=float) for t in train_df["temps"].tolist()]
        )
        temp_scaler.fit(train_temps_flat.reshape(-1, 1))

        def _scale_temps(temp_list):
            arr = np.asarray(temp_list, dtype=float).reshape(-1, 1)
            return temp_scaler.transform(arr).ravel().tolist()

        train_df["temps_scaled"] = train_df["temps"].apply(_scale_temps)
        test_df["temps_scaled"] = test_df["temps"].apply(_scale_temps)

        stats = {
            "n_train_curves_before": len(train_curves_df),
            "n_test_curves_before": len(test_curves_df),
            "n_train_curves_after": len(train_df),
            "n_test_curves_after": len(test_df),
            "feature_cols": feature_cols,
            "sort_temperatures": sort_temperatures,
            "drop_duplicate_temperatures": drop_duplicate_temperatures,
            "low_enabled": low_enabled,
            "low_offset_below_min_temp": low_offset_below_min_temp,
            "low_spacing": low_spacing,
            "low_lower_bound_temp": low_lower_bound_temp,
            "low_only_if_first_observed_conversion_below": low_only_if_first_observed_conversion_below,
            "low_value": low_value,
            "high_enabled": high_enabled,
            "high_offset_above_max_temp": high_offset_above_max_temp,
            "high_spacing": high_spacing,
            "high_upper_bound_temp": high_upper_bound_temp,
            "high_only_if_final_observed_conversion_above": high_only_if_final_observed_conversion_above,
            "high_value": high_value,
            "n_train_low_synthetic_points_inserted": n_train_low_inserted,
            "n_test_low_synthetic_points_inserted": n_test_low_inserted,
            "n_train_high_synthetic_points_inserted": n_train_high_inserted,
            "n_test_high_synthetic_points_inserted": n_test_high_inserted,
        }

        return train_df, test_df, feature_scaler, temp_scaler, stats

    def preprocess_h2_tpr_peaks(
        self,
        h2_tpr_df: pd.DataFrame,
        config: Optional[Dict] = None,
        default_to_mean_cols: Optional[List[str]] = None,
        default_to_zero_cols: Optional[List[str]] = None,
        cannot_be_zero_or_none: Optional[List[str]] = None,
        cannot_be_none: Optional[List[str]] = None,
        min_temps: Optional[int] = None,
        drop_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:

        resolved = self._resolve_from_config(
            config,
            "h2_tpr_peaks",
            default_to_mean_cols=default_to_mean_cols,
            default_to_zero_cols=default_to_zero_cols,
            cannot_be_zero_or_none=cannot_be_zero_or_none,
            cannot_be_none=cannot_be_none,
            min_temps=min_temps,
            drop_cols=drop_cols,
        )

        default_to_mean_cols = resolved["default_to_mean_cols"]
        default_to_zero_cols = resolved["default_to_zero_cols"]
        cannot_be_zero_or_none = resolved["cannot_be_zero_or_none"]
        cannot_be_none = resolved["cannot_be_none"]
        min_temps = resolved["min_temps"]
        drop_cols = resolved["drop_cols"]

        if default_to_mean_cols is None:
            default_to_mean_cols = []
        if default_to_zero_cols is None:
            default_to_zero_cols = ["pretreatment_temp", "pretreatment_time"]
        if cannot_be_zero_or_none is None:
            cannot_be_zero_or_none = ["ramp_rate_C_min"]
        if cannot_be_none is None:
            cannot_be_none = []
        if min_temps is None:
            min_temps = 1
        if drop_cols is None:
            drop_cols = []

        h2_tpr_df = h2_tpr_df.copy()
        h2_tpr_df.replace("", pd.NA, inplace=True)

        n_before = len(h2_tpr_df)

        stats = {
            "total_rows_before_preprocessing": n_before,
            "min_temps": min_temps,
            "drop_cols_requested": drop_cols,
        }

        # -------------------------------------------------
        # Convert boolean-like pretreatment_oxidising to 1/0
        # only when it is being used as a required-positive field
        # -------------------------------------------------
        boolean_conversion_stats = {
            "column_converted": False,
            "true_values_seen": 0,
            "false_values_seen": 0,
            "missing_values_seen": 0,
            "unrecognised_nonmissing_values": 0,
        }

        if "pretreatment_oxidising" in (cannot_be_zero_or_none + cannot_be_none):
            def _convert_bool_like(val):
                if pd.isna(val):
                    return pd.NA

                if isinstance(val, bool):
                    return int(val)

                if isinstance(val, (int, float)) and not pd.isna(val):
                    if val in (0, 1):
                        return int(val)

                sval = str(val).strip().lower()
                if sval in {"true", "t", "yes", "y", "1"}:
                    return 1
                if sval in {"false", "f", "no", "n", "0"}:
                    return 0

                return pd.NA

            original = h2_tpr_df["pretreatment_oxidising"].copy()
            converted = original.apply(_convert_bool_like)

            boolean_conversion_stats["column_converted"] = True
            boolean_conversion_stats["true_values_seen"] = int(
                original.apply(lambda x: (isinstance(x, bool) and x is True) or str(x).strip().lower() in {"true", "t", "yes", "y", "1"} if not pd.isna(x) else False).sum()
            )
            boolean_conversion_stats["false_values_seen"] = int(
                original.apply(lambda x: (isinstance(x, bool) and x is False) or str(x).strip().lower() in {"false", "f", "no", "n", "0"} if not pd.isna(x) else False).sum()
            )
            boolean_conversion_stats["missing_values_seen"] = int(original.isna().sum())
            boolean_conversion_stats["unrecognised_nonmissing_values"] = int(
                original.notna().sum() - converted.notna().sum()
            )

            h2_tpr_df["pretreatment_oxidising"] = converted

        stats["pretreatment_oxidising_conversion"] = boolean_conversion_stats

        # -------------------------
        # Fill columns with mean
        # -------------------------
        number_of_values_defaulted_to_mean = {}
        for col in default_to_mean_cols:
            if col not in h2_tpr_df.columns:
                number_of_values_defaulted_to_mean[col] = 0
                continue

            n_missing = int(h2_tpr_df[col].isna().sum())
            number_of_values_defaulted_to_mean[col] = n_missing

            if n_missing > 0:
                h2_tpr_df[col] = h2_tpr_df[col].fillna(h2_tpr_df[col].mean())

        stats["columns_filled_with_mean"] = default_to_mean_cols
        stats["number_of_values_defaulted_to_mean"] = number_of_values_defaulted_to_mean

        # -------------------------
        # Fill columns with zero
        # -------------------------
        number_of_values_defaulted_to_zero = {}
        for col in default_to_zero_cols:
            if col not in h2_tpr_df.columns:
                number_of_values_defaulted_to_zero[col] = 0
                continue

            n_missing = int(h2_tpr_df[col].isna().sum())
            number_of_values_defaulted_to_zero[col] = n_missing

            if n_missing > 0:
                h2_tpr_df[col] = h2_tpr_df[col].fillna(0)

        stats["columns_filled_with_zero"] = default_to_zero_cols
        stats["number_of_values_defaulted_to_zero"] = number_of_values_defaulted_to_zero

        # -------------------------
        # cannot_be_none validation
        # -------------------------
        keep_mask = pd.Series(True, index=h2_tpr_df.index)
        dropped_due_to_none_only = {}

        for col in cannot_be_none:
            if col not in h2_tpr_df.columns:
                dropped_due_to_none_only[col] = len(h2_tpr_df)
            else:
                dropped_due_to_none_only[col] = int(h2_tpr_df[col].isna().sum())

        # ------------------------------------------------
        # Scalar required-positive checks
        # Exclude list-like cols such as temps from this
        # ------------------------------------------------
        scalar_required_cols = [col for col in cannot_be_zero_or_none if col != "temps"]

        dropped_due_to_missing_values = {}
        dropped_due_to_zero_or_negative_values = {}

        for col in scalar_required_cols:
            if col not in h2_tpr_df.columns:
                dropped_due_to_missing_values[col] = len(h2_tpr_df)
                dropped_due_to_zero_or_negative_values[col] = 0
                continue

            dropped_due_to_missing_values[col] = int(h2_tpr_df[col].isna().sum())
            dropped_due_to_zero_or_negative_values[col] = int(
                h2_tpr_df[col].notna().sum() - h2_tpr_df[col].gt(0).sum()
            )

        # -------------------------
        # temps list validation
        # -------------------------
        temps_validation_stats = {
            "temps_column_present": "temps" in h2_tpr_df.columns,
            "rows_with_missing_temps": 0,
            "rows_with_non_list_temps": 0,
            "rows_with_too_few_temps": 0,
        }

        if "temps" in h2_tpr_df.columns:
            def _valid_temps(vals):
                # Check missing first (safe)
                if vals is None:
                    return False

                # Handle numpy arrays + lists + tuples
                if isinstance(vals, (list, tuple, np.ndarray)):
                    return len(vals) >= min_temps

                return False

            def _missing_temps(vals):
                return vals is None

            def _non_list_temps(vals):
                return (vals is not None) and (not isinstance(vals, (list, tuple, np.ndarray)))

            def _too_few_temps(vals):
                return isinstance(vals, (list, tuple, np.ndarray)) and len(vals) < min_temps

            temps_mask = h2_tpr_df["temps"].apply(_valid_temps)

            temps_validation_stats["rows_with_missing_temps"] = int(
                h2_tpr_df["temps"].apply(_missing_temps).sum()
            )
            temps_validation_stats["rows_with_non_list_temps"] = int(
                h2_tpr_df["temps"].apply(_non_list_temps).sum()
            )
            temps_validation_stats["rows_with_too_few_temps"] = int(
                h2_tpr_df["temps"].apply(_too_few_temps).sum()
            )
        else:
            temps_mask = pd.Series(False, index=h2_tpr_df.index)
            temps_validation_stats["rows_with_missing_temps"] = len(h2_tpr_df)

        rows_dropped_due_to_invalid_temps = int((~temps_mask).sum())
        stats["temps_validation"] = temps_validation_stats

        # -------------------------
        # Build keep mask
        # -------------------------
        for col in cannot_be_none:
            if col in h2_tpr_df.columns:
                keep_mask &= h2_tpr_df[col].notna()
            else:
                keep_mask &= False

        for col in scalar_required_cols:
            if col in h2_tpr_df.columns:
                keep_mask &= h2_tpr_df[col].notna() & h2_tpr_df[col].gt(0)
            else:
                keep_mask &= False

        if "temps" in cannot_be_zero_or_none or "temps" in h2_tpr_df.columns:
            keep_mask &= temps_mask

        h2_tpr_df = h2_tpr_df[keep_mask].copy()

        # -------------------------
        # Drop requested columns
        # -------------------------
        dropped_columns_that_existed = [col for col in drop_cols if col in h2_tpr_df.columns]
        if dropped_columns_that_existed:
            h2_tpr_df = h2_tpr_df.drop(columns=dropped_columns_that_existed)

        stats.update({
            "total_rows_after_preprocessing": len(h2_tpr_df),
            "total_rows_dropped": n_before - len(h2_tpr_df),
            "cannot_be_none": cannot_be_none,
            "dropped_due_to_none_only": dropped_due_to_none_only,
            "cannot_be_zero_or_none": cannot_be_zero_or_none,
            "scalar_required_positive_cols": scalar_required_cols,
            "rows_dropped_due_to_invalid_temps": rows_dropped_due_to_invalid_temps,
            "dropped_due_to_missing_values": dropped_due_to_missing_values,
            "dropped_due_to_zero_or_negative_values": dropped_due_to_zero_or_negative_values,
            "dropped_columns_that_existed": dropped_columns_that_existed,
        })

        return h2_tpr_df, stats

    def _clean_and_encode_pretreatment_gas_type(
        self,
        df: pd.DataFrame,
        remove_ambiguous_pretreatment_type: bool = True,
        numerical_pretreatment_type: bool = True,
        pretreatment_gas_type_col: str = "pretreatment_gas_type",
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean and optionally numerically encode pretreatment gas type.

        Allowed canonical values:
            - "oxidising"
            - "reducing"
            - "inert"

        Ambiguous values are values containing "/" or values whose normalized form
        is not one of the allowed canonical values.

        Behaviour
        ---------
        - If remove_ambiguous_pretreatment_type is True:
            rows with ambiguous / unrecognized pretreatment_gas_type are removed.
        - If numerical_pretreatment_type is True:
            * if exactly 2 canonical classes remain, create one binary column for one class
              (preferring "inert_pretreatment", then "reducing_pretreatment",
              then "oxidising_pretreatment")
            * if all 3 remain, create one-hot columns:
                ["inert_pretreatment", "reducing_pretreatment", "oxidising_pretreatment"]
            * the original pretreatment_gas_type column is dropped after encoding
        """
        df = df.copy()

        stats = {
            "pretreatment_gas_type_column_used": pretreatment_gas_type_col,
            "remove_ambiguous_pretreatment_type": remove_ambiguous_pretreatment_type,
            "numerical_pretreatment_type": numerical_pretreatment_type,
            "rows_with_missing_pretreatment_gas_type_before": 0,
            "rows_with_slash_in_pretreatment_gas_type": 0,
            "rows_with_unrecognized_pretreatment_gas_type": 0,
            "rows_dropped_due_to_ambiguous_pretreatment_gas_type": 0,
            "pretreatment_type_counts_after_cleaning": {},
            "pretreatment_type_encoding_mode": None,
            "pretreatment_type_columns_created": [],
        }

        if pretreatment_gas_type_col not in df.columns:
            return df, stats

        allowed = {"oxidising", "reducing", "inert"}

        def _normalize_gas_type(val):
            if pd.isna(val):
                return pd.NA

            sval = str(val).strip().lower()
            if sval == "":
                return pd.NA

            # explicitly ambiguous if slash present
            if "/" in sval:
                return "__AMBIGUOUS__"

            # some mild normalization
            alias_map = {
                "oxidizing": "oxidising",
                "oxidising": "oxidising",
                "reducing": "reducing",
                "reduction": "reducing",
                "inert": "inert",
            }
            sval = alias_map.get(sval, sval)

            if sval in allowed:
                return sval

            return "__AMBIGUOUS__"

        original = df[pretreatment_gas_type_col].copy()
        cleaned = original.apply(_normalize_gas_type)

        stats["rows_with_missing_pretreatment_gas_type_before"] = int(original.isna().sum())
        stats["rows_with_slash_in_pretreatment_gas_type"] = int(
            original.fillna("").astype(str).str.contains("/", regex=False).sum()
        )
        stats["rows_with_unrecognized_pretreatment_gas_type"] = int((cleaned == "__AMBIGUOUS__").sum())

        if remove_ambiguous_pretreatment_type:
            keep_mask = cleaned.notna() & (cleaned != "__AMBIGUOUS__")
            stats["rows_dropped_due_to_ambiguous_pretreatment_gas_type"] = int((~keep_mask).sum())
            df = df.loc[keep_mask].copy()
            cleaned = cleaned.loc[keep_mask].copy()
        else:
            # keep rows, but ambiguous values become NA so they can be handled downstream
            cleaned = cleaned.replace("__AMBIGUOUS__", pd.NA)

        df[pretreatment_gas_type_col] = cleaned

        type_counts = df[pretreatment_gas_type_col].value_counts(dropna=True).to_dict()
        stats["pretreatment_type_counts_after_cleaning"] = {
            str(k): int(v) for k, v in type_counts.items()
        }

        if not numerical_pretreatment_type:
            return df, stats

        present_types = sorted([t for t in df[pretreatment_gas_type_col].dropna().unique().tolist() if t in allowed])

        if len(present_types) == 2:
            # Prefer inert as the binary flag if present, otherwise reducing, then oxidising
            preferred_order = ["inert", "reducing", "oxidising"]
            positive_class = next(t for t in preferred_order if t in present_types)

            new_col = f"{positive_class}_pretreatment"
            df[new_col] = (df[pretreatment_gas_type_col] == positive_class).astype(int)

            stats["pretreatment_type_encoding_mode"] = "binary"
            stats["pretreatment_type_columns_created"] = [new_col]

            df = df.drop(columns=[pretreatment_gas_type_col])

        elif len(present_types) == 3:
            new_cols = []
            for gas_type in ["inert", "reducing", "oxidising"]:
                new_col = f"{gas_type}_pretreatment"
                df[new_col] = (df[pretreatment_gas_type_col] == gas_type).astype(int)
                new_cols.append(new_col)

            stats["pretreatment_type_encoding_mode"] = "one_hot"
            stats["pretreatment_type_columns_created"] = new_cols

            df = df.drop(columns=[pretreatment_gas_type_col])

        else:
            # len 0 or 1: no useful encoding needed
            stats["pretreatment_type_encoding_mode"] = "none"

        return df, stats

    def preprocess_tpd_peaks(
        self,
        tpd_df: pd.DataFrame,
        config: Optional[Dict] = None,
        config_section: str = "o2_tpd_peaks",
        default_to_mean_cols: Optional[List[str]] = None,
        default_to_zero_cols: Optional[List[str]] = None,
        cannot_be_zero_or_none: Optional[List[str]] = None,
        cannot_be_none: Optional[List[str]] = None,
        min_temps: Optional[int] = None,
        remove_ambiguous_pretreatment_type: Optional[bool] = None,
        numerical_pretreatment_type: Optional[bool] = None,
        drop_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:

        resolved = self._resolve_from_config(
            config,
            config_section,
            default_to_mean_cols=default_to_mean_cols,
            default_to_zero_cols=default_to_zero_cols,
            cannot_be_zero_or_none=cannot_be_zero_or_none,
            cannot_be_none=cannot_be_none,
            min_temps=min_temps,
            remove_ambiguous_pretreatment_type=remove_ambiguous_pretreatment_type,
            numerical_pretreatment_type=numerical_pretreatment_type,
            drop_cols=drop_cols,
        )

        default_to_mean_cols = resolved["default_to_mean_cols"]
        default_to_zero_cols = resolved["default_to_zero_cols"]
        cannot_be_zero_or_none = resolved["cannot_be_zero_or_none"]
        cannot_be_none = resolved["cannot_be_none"]
        min_temps = resolved["min_temps"]
        remove_ambiguous_pretreatment_type = resolved["remove_ambiguous_pretreatment_type"]
        numerical_pretreatment_type = resolved["numerical_pretreatment_type"]
        drop_cols = resolved["drop_cols"]

        if default_to_mean_cols is None:
            default_to_mean_cols = []
        if default_to_zero_cols is None:
            default_to_zero_cols = ["pretreatment_temp", "pretreatment_time"]
        if cannot_be_zero_or_none is None:
            cannot_be_zero_or_none = ["ramp_rate_C_min"]
        if cannot_be_none is None:
            cannot_be_none = []
        if min_temps is None:
            min_temps = 1
        if remove_ambiguous_pretreatment_type is None:
            remove_ambiguous_pretreatment_type = True
        if numerical_pretreatment_type is None:
            numerical_pretreatment_type = True
        if drop_cols is None:
            drop_cols = []

        tpd_df = tpd_df.copy()
        tpd_df.replace("", pd.NA, inplace=True)

        n_before = len(tpd_df)

        stats = {
            "total_rows_before_preprocessing": n_before,
            "min_temps": min_temps,
            "drop_cols_requested": drop_cols,
            "cannot_be_none": cannot_be_none,
            "cannot_be_zero_or_none": cannot_be_zero_or_none,
            "remove_ambiguous_pretreatment_type": remove_ambiguous_pretreatment_type,
            "numerical_pretreatment_type": numerical_pretreatment_type,
        }

        # -------------------------
        # Fill columns with mean
        # -------------------------
        number_of_values_defaulted_to_mean = {}
        for col in default_to_mean_cols:
            if col not in tpd_df.columns:
                number_of_values_defaulted_to_mean[col] = 0
                continue

            n_missing = int(tpd_df[col].isna().sum())
            number_of_values_defaulted_to_mean[col] = n_missing

            if n_missing > 0:
                tpd_df[col] = tpd_df[col].fillna(tpd_df[col].mean())

        stats["columns_filled_with_mean"] = default_to_mean_cols
        stats["number_of_values_defaulted_to_mean"] = number_of_values_defaulted_to_mean

        # -------------------------
        # Fill columns with zero
        # -------------------------
        number_of_values_defaulted_to_zero = {}
        for col in default_to_zero_cols:
            if col not in tpd_df.columns:
                number_of_values_defaulted_to_zero[col] = 0
                continue

            n_missing = int(tpd_df[col].isna().sum())
            number_of_values_defaulted_to_zero[col] = n_missing

            if n_missing > 0:
                tpd_df[col] = tpd_df[col].fillna(0)

        stats["columns_filled_with_zero"] = default_to_zero_cols
        stats["number_of_values_defaulted_to_zero"] = number_of_values_defaulted_to_zero

        # -------------------------
        # cannot_be_none validation
        # -------------------------
        dropped_due_to_none_only = {}
        for col in cannot_be_none:
            if col not in tpd_df.columns:
                dropped_due_to_none_only[col] = len(tpd_df)
            else:
                dropped_due_to_none_only[col] = int(tpd_df[col].isna().sum())

        # ------------------------------------------------
        # Scalar required-positive checks
        # ------------------------------------------------
        scalar_required_cols = [col for col in cannot_be_zero_or_none if col != "temps"]

        dropped_due_to_missing_values = {}
        dropped_due_to_zero_or_negative_values = {}

        for col in scalar_required_cols:
            if col not in tpd_df.columns:
                dropped_due_to_missing_values[col] = len(tpd_df)
                dropped_due_to_zero_or_negative_values[col] = 0
                continue

            dropped_due_to_missing_values[col] = int(tpd_df[col].isna().sum())
            dropped_due_to_zero_or_negative_values[col] = int(
                tpd_df[col].notna().sum() - tpd_df[col].gt(0).sum()
            )

        # -------------------------
        # temps validation
        # -------------------------
        def _is_sequence(vals):
            return isinstance(vals, (list, tuple, np.ndarray))

        def _valid_temps(vals):
            if vals is None:
                return False
            if not _is_sequence(vals):
                return False
            return len(vals) >= min_temps

        def _missing_temps(vals):
            return vals is None

        def _non_list_temps(vals):
            return (vals is not None) and (not _is_sequence(vals))

        def _too_few_temps(vals):
            return _is_sequence(vals) and len(vals) < min_temps

        temps_validation_stats = {
            "temps_column_present": "temps" in tpd_df.columns,
            "rows_with_missing_temps": 0,
            "rows_with_non_list_temps": 0,
            "rows_with_too_few_temps": 0,
        }

        if "temps" in tpd_df.columns:
            temps_series = tpd_df["temps"].apply(
                lambda x: None if (isinstance(x, float) and pd.isna(x)) else x
            )

            temps_mask = temps_series.apply(_valid_temps)

            temps_validation_stats["rows_with_missing_temps"] = int(
                temps_series.apply(_missing_temps).sum()
            )
            temps_validation_stats["rows_with_non_list_temps"] = int(
                temps_series.apply(_non_list_temps).sum()
            )
            temps_validation_stats["rows_with_too_few_temps"] = int(
                temps_series.apply(_too_few_temps).sum()
            )
        else:
            temps_mask = pd.Series(False, index=tpd_df.index)
            temps_validation_stats["rows_with_missing_temps"] = len(tpd_df)

        stats["temps_validation"] = temps_validation_stats

        # -------------------------
        # Build keep mask
        # -------------------------
        keep_mask = pd.Series(True, index=tpd_df.index)

        for col in cannot_be_none:
            if col in tpd_df.columns:
                keep_mask &= tpd_df[col].notna()
            else:
                keep_mask &= False

        for col in scalar_required_cols:
            if col in tpd_df.columns:
                keep_mask &= tpd_df[col].notna() & tpd_df[col].gt(0)
            else:
                keep_mask &= False

        keep_mask &= temps_mask

        rows_dropped_due_to_invalid_temps = int((~temps_mask).sum())

        tpd_df = tpd_df.loc[keep_mask].copy()

        # -------------------------
        # Clean / encode pretreatment gas type
        # -------------------------
        tpd_df, pretreatment_type_stats = self._clean_and_encode_pretreatment_gas_type(
            df=tpd_df,
            remove_ambiguous_pretreatment_type=remove_ambiguous_pretreatment_type,
            numerical_pretreatment_type=numerical_pretreatment_type,
            pretreatment_gas_type_col="pretreatment_gas_type",
        )

        # -------------------------
        # Drop requested columns
        # -------------------------
        dropped_columns_that_existed = [col for col in drop_cols if col in tpd_df.columns]
        if dropped_columns_that_existed:
            tpd_df = tpd_df.drop(columns=dropped_columns_that_existed)

        stats.update({
            "pretreatment_type_stats": pretreatment_type_stats,
            "scalar_required_positive_cols": scalar_required_cols,
            "dropped_due_to_none_only": dropped_due_to_none_only,
            "dropped_due_to_missing_values": dropped_due_to_missing_values,
            "dropped_due_to_zero_or_negative_values": dropped_due_to_zero_or_negative_values,
            "rows_dropped_due_to_invalid_temps": rows_dropped_due_to_invalid_temps,
            "dropped_columns_that_existed": dropped_columns_that_existed,
            "total_rows_after_preprocessing": len(tpd_df),
            "total_rows_dropped": n_before - len(tpd_df),
        })

        return tpd_df, stats
    
    def preprocess_osc(
        self,
        osc_df: pd.DataFrame,
        config: Optional[Dict] = None,
        default_to_mean_cols: Optional[List[str]] = None,
        default_to_zero_cols: Optional[List[str]] = None,
        cannot_be_zero_or_none: Optional[List[str]] = None,
        cannot_be_none: Optional[List[str]] = None,
        allowed_experiment_classes: Optional[List[str]] = None,
        tag_dynamic: Optional[bool] = None,
        dynamic_or_total_tag_name: Optional[str] = None,
        drop_cols: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict]:

        resolved = self._resolve_from_config(
            config,
            "osc",
            default_to_mean_cols=default_to_mean_cols,
            default_to_zero_cols=default_to_zero_cols,
            cannot_be_zero_or_none=cannot_be_zero_or_none,
            cannot_be_none=cannot_be_none,
            allowed_experiment_classes=allowed_experiment_classes,
            tag_dynamic=tag_dynamic,
            dynamic_or_total_tag_name=dynamic_or_total_tag_name,
            drop_cols=drop_cols,
        )

        default_to_mean_cols = resolved["default_to_mean_cols"]
        default_to_zero_cols = resolved["default_to_zero_cols"]
        cannot_be_zero_or_none = resolved["cannot_be_zero_or_none"]
        cannot_be_none = resolved["cannot_be_none"]
        allowed_experiment_classes = resolved["allowed_experiment_classes"]
        tag_dynamic = resolved["tag_dynamic"]
        dynamic_or_total_tag_name = resolved["dynamic_or_total_tag_name"]
        drop_cols = resolved["drop_cols"]

        if default_to_mean_cols is None:
            default_to_mean_cols = []
        if default_to_zero_cols is None:
            default_to_zero_cols = ["h2_vol%", "co_vol%", "o2_vol%"]
        if cannot_be_zero_or_none is None:
            cannot_be_zero_or_none = ["value_O_umol_per_g_catalyst", "value_raw"]
        if cannot_be_none is None:
            cannot_be_none = []
        if allowed_experiment_classes is None:
            allowed_experiment_classes = ["a", "b", "c", "d"]
        if tag_dynamic is None:
            tag_dynamic = False
        if dynamic_or_total_tag_name is None:
            dynamic_or_total_tag_name = "is_dynamic_capacity"
        if drop_cols is None:
            drop_cols = []

        osc_df = osc_df.copy()
        osc_df.replace("", pd.NA, inplace=True)

        n_before = len(osc_df)

        stats = {
            "total_rows_before_preprocessing": n_before,
            "cannot_be_none": cannot_be_none,
            "cannot_be_zero_or_none": cannot_be_zero_or_none,
            "allowed_experiment_classes_requested": allowed_experiment_classes,
            "tag_dynamic": tag_dynamic,
            "dynamic_or_total_tag_name": dynamic_or_total_tag_name,
            "drop_cols_requested": drop_cols,
        }

        # -------------------------
        # Fill columns with mean
        # -------------------------
        number_of_values_defaulted_to_mean = {}
        for col in default_to_mean_cols:
            if col not in osc_df.columns:
                number_of_values_defaulted_to_mean[col] = 0
                continue

            n_missing = int(osc_df[col].isna().sum())
            number_of_values_defaulted_to_mean[col] = n_missing

            if n_missing > 0:
                osc_df[col] = osc_df[col].fillna(osc_df[col].mean())

        stats["columns_filled_with_mean"] = default_to_mean_cols
        stats["number_of_values_defaulted_to_mean"] = number_of_values_defaulted_to_mean

        # -------------------------
        # Fill columns with zero
        # -------------------------
        number_of_values_defaulted_to_zero = {}
        for col in default_to_zero_cols:
            if col not in osc_df.columns:
                number_of_values_defaulted_to_zero[col] = 0
                continue

            n_missing = int(osc_df[col].isna().sum())
            number_of_values_defaulted_to_zero[col] = n_missing

            if n_missing > 0:
                osc_df[col] = osc_df[col].fillna(0)

        stats["columns_filled_with_zero"] = default_to_zero_cols
        stats["number_of_values_defaulted_to_zero"] = number_of_values_defaulted_to_zero

        # -------------------------
        # cannot_be_none validation
        # -------------------------
        dropped_due_to_none_only = {}
        for col in cannot_be_none:
            if col not in osc_df.columns:
                dropped_due_to_none_only[col] = len(osc_df)
            else:
                dropped_due_to_none_only[col] = int(osc_df[col].isna().sum())

        # -------------------------
        # cannot_be_zero_or_none validation
        # -------------------------
        dropped_due_to_missing_values = {}
        dropped_due_to_zero_or_negative_values = {}

        for col in cannot_be_zero_or_none:
            if col not in osc_df.columns:
                dropped_due_to_missing_values[col] = len(osc_df)
                dropped_due_to_zero_or_negative_values[col] = 0
                continue

            col_series = osc_df[col]
            dropped_due_to_missing_values[col] = int(col_series.isna().sum())

            if pd.api.types.is_numeric_dtype(col_series):
                dropped_due_to_zero_or_negative_values[col] = int(
                    col_series.notna().sum() - col_series.gt(0).sum()
                )
            else:
                # For non-numeric fields, only require non-empty string
                dropped_due_to_zero_or_negative_values[col] = int(
                    (col_series.fillna("").astype(str).str.strip() == "").sum()
                )

        # -------------------------
        # Normalize and filter measurement_class
        # -------------------------
        measurement_class_stats = {
            "column_present": "measurement_class" in osc_df.columns,
            "rows_missing_before_filter": 0,
            "rows_not_in_allowed_classes": 0,
            "counts_after_filter": {},
        }

        if "measurement_class" in osc_df.columns:
            osc_df["measurement_class"] = (
                osc_df["measurement_class"]
                .astype("string")
                .str.strip()
                .str.lower()
            )

            measurement_class_stats["rows_missing_before_filter"] = int(
                osc_df["measurement_class"].isna().sum()
            )

            allowed_set = set(str(x).strip().lower() for x in allowed_experiment_classes)
            allowed_mask = osc_df["measurement_class"].isin(allowed_set)

            measurement_class_stats["rows_not_in_allowed_classes"] = int((~allowed_mask).sum())

        else:
            allowed_mask = pd.Series(False, index=osc_df.index)
            measurement_class_stats["rows_missing_before_filter"] = len(osc_df)

        # -------------------------
        # Build keep mask
        # -------------------------
        keep_mask = pd.Series(True, index=osc_df.index)

        for col in cannot_be_none:
            if col in osc_df.columns:
                keep_mask &= osc_df[col].notna()
            else:
                keep_mask &= False

        for col in cannot_be_zero_or_none:
            if col not in osc_df.columns:
                keep_mask &= False
                continue

            if pd.api.types.is_numeric_dtype(osc_df[col]):
                keep_mask &= osc_df[col].notna() & osc_df[col].gt(0)
            else:
                keep_mask &= osc_df[col].fillna("").astype(str).str.strip().ne("")

        keep_mask &= allowed_mask

        osc_df = osc_df.loc[keep_mask].copy()

        measurement_class_stats["counts_after_filter"] = {
            str(k): int(v)
            for k, v in osc_df["measurement_class"].value_counts(dropna=True).to_dict().items()
        }

        # -------------------------
        # Tag dynamic vs total
        # a,c -> dynamic (1)
        # b,d -> total   (0)
        # -------------------------
        dynamic_tag_stats = {
            "tag_created": False,
            "dynamic_classes": ["a", "c"],
            "total_classes": ["b", "d"],
            "positive_count": 0,
            "zero_count": 0,
        }

        if tag_dynamic and "measurement_class" in osc_df.columns:
            dynamic_classes = {"a", "c"}
            total_classes = {"b", "d"}

            osc_df[dynamic_or_total_tag_name] = osc_df["measurement_class"].apply(
                lambda x: 1 if x in dynamic_classes else 0 if x in total_classes else pd.NA
            )

            dynamic_tag_stats["tag_created"] = True
            dynamic_tag_stats["positive_count"] = int((osc_df[dynamic_or_total_tag_name] == 1).sum())
            dynamic_tag_stats["zero_count"] = int((osc_df[dynamic_or_total_tag_name] == 0).sum())

        # -------------------------
        # Drop requested columns
        # -------------------------
        dropped_columns_that_existed = [col for col in drop_cols if col in osc_df.columns]
        if dropped_columns_that_existed:
            osc_df = osc_df.drop(columns=dropped_columns_that_existed)

        stats.update({
            "measurement_class_stats": measurement_class_stats,
            "dynamic_tag_stats": dynamic_tag_stats,
            "dropped_due_to_none_only": dropped_due_to_none_only,
            "dropped_due_to_missing_values": dropped_due_to_missing_values,
            "dropped_due_to_zero_or_negative_values": dropped_due_to_zero_or_negative_values,
            "dropped_columns_that_existed": dropped_columns_that_existed,
            "total_rows_after_preprocessing": len(osc_df),
            "total_rows_dropped": n_before - len(osc_df),
        })

        return osc_df, stats

    def merge_characterisation_with_materials(
        self,
        materials_df: pd.DataFrame,
        char_df: pd.DataFrame,
        material_id_col_in_char: str = "material_id",
        material_id_col_in_materials: str = "material_id",
        drop_material_id_in_char: bool = True,
    ) -> Tuple[pd.DataFrame, Dict]:

        materials_df = materials_df.copy()
        char_df = char_df.copy()

        # -------------------------
        # Ensure material_id exists in materials
        # -------------------------
        if material_id_col_in_materials not in materials_df.columns:
            if "_id" in materials_df.columns:
                materials_df[material_id_col_in_materials] = materials_df["_id"]
            else:
                raise KeyError("materials_df must contain 'material_id' or '_id'")

        if material_id_col_in_char not in char_df.columns:
            raise KeyError(f"char_df must contain '{material_id_col_in_char}'")

        n_materials_before = len(materials_df)
        n_char_before = len(char_df)

        char_id_present = "_id" in char_df.columns
        if char_id_present:
            char_df = char_df.drop(columns=["_id"])

        merged_df = materials_df.merge(
            char_df,
            left_on=material_id_col_in_materials,
            right_on=material_id_col_in_char,
            how="inner",
        )

        if drop_material_id_in_char and material_id_col_in_char in merged_df.columns:
            # Only drop if duplicated (avoid dropping the main one)
            if material_id_col_in_char != material_id_col_in_materials:
                merged_df = merged_df.drop(columns=[material_id_col_in_char])

        stats = {
            "n_materials_before_merge": n_materials_before,
            "n_characterisation_rows_before_merge": n_char_before,
            "n_rows_after_merge": len(merged_df),
            "n_unique_materials_after_merge": merged_df[material_id_col_in_materials].nunique(),
            "char_id_was_dropped": char_id_present,
        }

        return merged_df, stats