from .element_attributes import METALS, Supported_Attributes, Metal, DEFAULT_ATTRIBUTES, CUSTOM_FUNCS, FLOAT_ATTRIBUTES, METHOD_ATTRIBUTES, NON_FLOAT_VARIABLES, DEFAULT_OVERRIDES
from typing import List, Dict, Optional
import pandas as pd
from mendeleev import element
from mendeleev.models import Element
import warnings

class DopantFeaturiser:
    def __init__(
        self, 
        metals: List[Metal]=METALS, 
        n_allowed_dopants: int=2, 
        desired_features: List[Supported_Attributes]=DEFAULT_ATTRIBUTES,
        overrides: Dict[Metal, Dict[str, float]]=DEFAULT_OVERRIDES,
        feature_map: Optional[Dict[Metal, Dict[Supported_Attributes, float]]]=None
    ) -> None:
        
        self.metals = metals
        self.n_allowed_dopants = n_allowed_dopants
        self.desired_features = desired_features
        self.overrides = overrides
        if not feature_map:
            self.feature_map = self.get_feature_map()
        else:
            self.feature_map = feature_map

    
    def convert_features(self, df: pd.DataFrame, leave_ce: bool = True, include_n_dopants: bool=True, delete_old_features: bool=False) -> pd.DataFrame:
        """Converts metal-wise concentrations to dopant features"""
        df = df.copy()

        metals_in_df = [m for m in self.metals if m in df.columns]

        if leave_ce and "Ce" in metals_in_df:
            dopant_cols = [m for m in metals_in_df if m != "Ce"]
        else:
            dopant_cols = metals_in_df

        # Non-element columns are kept as-is.

        cols_to_drop = dopant_cols.copy() if delete_old_features else []

        if not leave_ce and "Ce" in metals_in_df:
            cols_to_drop.append("Ce")

        base_df = df.drop(columns=cols_to_drop)

        featurised_rows = []

        valid_indices = []
        featurised_rows = []

        n_dropped = 0

        for idx, row in df.iterrows():
            row_features = {}

            dopants = []
            for metal in dopant_cols:
                frac = row.get(metal, 0.0)
                if pd.notna(frac) and float(frac) > 0:
                    dopants.append((metal, float(frac)))

            if len(dopants) > self.n_allowed_dopants:
                n_dropped += 1
                continue

            dopants = sorted(dopants, key=lambda x: (-x[1], x[0]))

            if include_n_dopants:
                row_features["n_dopants"] = len(dopants)

            for i in range(self.n_allowed_dopants):
                slot = i + 1

                if i < len(dopants):
                    metal, frac = dopants[i]
                    attrs = self.feature_map[metal]

                    row_features[f"dopant_{slot}_element_fraction"] = frac

                    for attr_name, attr_value in attrs.items():
                        row_features[f"dopant_{slot}_{attr_name}"] = attr_value

                else:
                    row_features[f"dopant_{slot}_element_fraction"] = 0.0

                    example_attrs = next(iter(self.feature_map.values()))
                    for attr_name in example_attrs:
                        row_features[f"dopant_{slot}_{attr_name}"] = 0.0

            featurised_rows.append(row_features)
            valid_indices.append(idx)
        dopant_features_df = pd.DataFrame(featurised_rows, index=valid_indices)
        base_df = base_df.loc[valid_indices]
        if n_dropped > 0:
            warnings.warn(
                f"Dropped {n_dropped} rows with more than {self.n_allowed_dopants} dopants. These will not be tracked in the preprocessing stats."
            )

        return pd.concat([base_df, dopant_features_df], axis=1)
        
    
    def get_feature_map(self) -> Dict[Metal, Dict[Supported_Attributes, float]]:
        feature_map = {}
        for metal_string in self.metals:
            metal = element(metal_string)
            metal_dict = {}
            for attr in self.desired_features:
                metal_dict = self._resolve_attribute(metal, metal_dict, attr)
            feature_map[metal_string] = metal_dict
        return feature_map


    def _resolve_attribute(self, metal: Element, metal_dict: Dict[str, float], attr: str):
        override = self.overrides.get(metal.symbol, {}).get(attr, None)
        if override is not None:
            if attr == 'electronegativity_pauling':
                print(f"Overriding {attr} of {metal.symbol} using wikipidea as not in Mendeleev due to being ill-defined.")

            metal_dict[attr] = override
            return metal_dict

        if attr in CUSTOM_FUNCS:
            metal_dict = CUSTOM_FUNCS[attr](metal, metal_dict)
        elif attr in FLOAT_ATTRIBUTES:
            metal_dict[attr] = getattr(metal, attr)
        elif attr in METHOD_ATTRIBUTES:
            try:
                result = getattr(metal, attr)()
                metal_dict[attr] = float(result)
            except Exception:
                raise AttributeError (f"Cannot process Element attribute '{attr}' of metal {metal.symbol} because it is a method that either must have an argument or cannot be converted to a float, or does not exist for this metal. Please make a custom function")
        elif attr in NON_FLOAT_VARIABLES:
            raise AttributeError (f"Cannot process Element attribute '{attr}' of metal {metal.symbol} because it is a variable that cannot be converted to a float, or does not exist for this metal. Please make a custom function")
        else:
            raise NotImplementedError(f"Cannot process Element attribute '{attr}' of metal {metal.symbol} because it does not exist as a custom function or a mendeleev.models.Element attribute, or does not exist for this metal.")
        return metal_dict
    

