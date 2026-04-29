from .element_attributes import METALS, Supported_Attributes, Metal, DEFAULT_ATTRIBUTES, CUSTOM_FUNCS, FLOAT_ATTRIBUTES, METHOD_ATTRIBUTES, NON_FLOAT_VARIABLES
from typing import List, Dict
import pandas as pd
from mendeleev import element
from mendeleev.models import Element

class DopantFeaturiser:
    def __init__(
        self, 
        metals: List[Metal]=METALS, 
        n_allowed_dopants: int=2, 
        desired_attributes: List[Supported_Attributes]=DEFAULT_ATTRIBUTES
    ) -> None:
        self.metals = metals
        self.n_allowed_dopants = n_allowed_dopants
        self.desired_attributes = desired_attributes
        self.feature_map = self.get_feature_map()

    
    def convert_features(self, df: pd.DataFrame, leave_ce_alone: bool = True) -> pd.DataFrame:
        """
        Replace elemental composition columns with dopant-slot features.

        Example output columns:
            Ce
            dopant_1_element_fraction
            dopant_1_atomic_radius
            dopant_1_electronegativity_pauling
            dopant_1_can_be_1
            ...
            dopant_2_element_fraction
            ...

        Dopants are sorted by fraction descending within each row.
        """

        df = df.copy()

        metals_in_df = [m for m in self.metals if m in df.columns]

        if leave_ce_alone and "Ce" in metals_in_df:
            dopant_cols = [m for m in metals_in_df if m != "Ce"]
        else:
            dopant_cols = metals_in_df

        # Non-element columns are kept as-is.
        cols_to_drop = dopant_cols.copy()
        if not leave_ce_alone and "Ce" in metals_in_df:
            cols_to_drop.append("Ce")

        base_df = df.drop(columns=cols_to_drop)

        featurised_rows = []

        for _, row in df.iterrows():
            row_features = {}

            # Find present dopants in this material
            dopants = []
            for metal in dopant_cols:
                frac = row.get(metal, 0.0)

                if pd.notna(frac) and float(frac) > 0:
                    dopants.append((metal, float(frac)))

            # Sort by amount, then element name for deterministic ordering
            dopants = sorted(dopants, key=lambda x: (-x[1], x[0]))

            # Fill dopant slots
            for i in range(self.n_allowed_dopants):
                slot = i + 1

                if i < len(dopants):
                    metal, frac = dopants[i]
                    attrs = self.feature_map[metal]

                    row_features[f"dopant_{slot}_element_fraction"] = frac
                    
                    for attr_name, attr_value in attrs.items():
                        row_features[f"dopant_{slot}_{attr_name}"] = attr_value

                else:
                    # Empty dopant slot
                    row_features[f"dopant_{slot}_element_fraction"] = 0.0
                    row_features[f"dopant_{slot}_atomic_number"] = 0.0

                    # Use the first feature_map entry to know which attributes exist
                    example_attrs = next(iter(self.feature_map.values()))

                    for attr_name in example_attrs:
                        row_features[f"dopant_{slot}_{attr_name}"] = 0.0

            featurised_rows.append(row_features)

        dopant_features_df = pd.DataFrame(featurised_rows, index=df.index)

        return pd.concat([base_df, dopant_features_df], axis=1)
        
    
    def get_feature_map(self) -> Dict[Metal, Dict[Supported_Attributes, float]]:
        feature_map = {}
        for metal_string in self.metals:
            metal = element(metal_string)
            metal_dict = {}
            for attr in self.desired_attributes:
                metal_dict = self._resolve_attribute(metal, metal_dict, attr)
            feature_map[metal_string] = metal_dict
        return feature_map


    def _resolve_attribute(self, metal: Element, metal_dict: Dict[str, float], attr: str):
        if attr in CUSTOM_FUNCS:
            metal_dict = CUSTOM_FUNCS[attr](metal, metal_dict)
        elif attr in FLOAT_ATTRIBUTES:
            metal_dict[attr] = getattr(metal, attr)
        elif attr in METHOD_ATTRIBUTES:
            try:
                result = getattr(metal, attr)()
                metal_dict[attr] = float(result)
            except:
                raise AttributeError (f"Cannot process Element attribute {attr} because it is a method that either must have an argument or cannot be converted to a float. Please make a custom function")
        elif attr in NON_FLOAT_VARIABLES:
            raise AttributeError (f"Cannot process Element attribute {attr} because it is a variable that cannot be converted to a float. Please make a custom function")
        else:
            raise NotImplementedError(f"Cannot process Element attribute {attr} because it does not exist as a custom function or a mendeleev.models.Element attribute")
        return metal_dict
    

