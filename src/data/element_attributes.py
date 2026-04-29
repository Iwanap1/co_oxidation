from typing import Literal, Dict
from mendeleev.models import Element


Metal = Literal[
    'Al', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Sr', 'Y', 'Zr', 'Nb', 'Ru', 'Ag', 'Sn', 'La', 'Ce',
    'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Ho', 'Yb', 'Lu',
    'Hf', 'Bi', "Pm"
]

METALS = list(Metal.__args__)

Supported_Attributes = Literal[
    # Mendeleev variable attributes
    'abundance_crust', 'abundance_sea', 'atomic_number', 'atomic_radius', 'atomic_radius_rahm', 'atomic_volume', 'atomic_weight', 'atomic_weight_uncertainty', 
    'boiling_point', 'c6_gb', 'covalent_radius', 'covalent_radius_cordero', 'covalent_radius_pyykko', 'covalent_radius_pyykko_double', 'covalent_radius_pyykko_triple', 
    'density', 'dipole_polarizability', 'dipole_polarizability_unc', 'discovery_year', 'electron_affinity', 'electrons', 'en_ghosh', 'en_gunnarsson_lundqvist', 
    'en_miedema', 'en_pauling', 'en_robles_bartolotti', 'evaporation_heat', 'fusion_heat', 'glawe_number', 'heat_of_formation', 'is_radioactive', 'lattice_constant', 
    'mass', 'mass_number', 'melting_point', 'mendeleev_number', 'miedema_electron_density', 'miedema_molar_volume', 'molar_heat_capacity', 'neutrons', 'period', 
    'pettifor_number', 'political_stability_of_top_producer', 'political_stability_of_top_reserve_holder', 'price_per_kg', 'production_concentration', 'protons', 
    'relative_supply_risk', 'reserve_distribution', 'specific_heat', 'specific_heat_capacity', 'thermal_conductivity', 'vdw_radius', 'vdw_radius_alvarez', 
    'vdw_radius_mm3', 'vdw_radius_uff',
    # Mendeleev method attributes that return a float-like value
    'electronegativity', 'electronegativity_allen', 'electronegativity_allred_rochow', 'electronegativity_cottrell_sutton', 'electronegativity_ghosh',
    'electronegativity_gordy', 'electronegativity_li_xue', 'electronegativity_martynov_batsanov', 'electronegativity_mullay', 'electronegativity_mulliken', 
    'electronegativity_nagle', 'electronegativity_pauling', 'electronegativity_sanderson', 'electronegativity_scales'
    # Custom
    'oxidation_states_one_hot', 'first_ionisation_energy'
]


NON_FLOAT_VARIABLES = ['block', 'c6', 'cas', 'covalent_radius_bragg', 'cpk_color', 'description', 'discoverers', 'discovery_location', 'ec', 'econf', 'en_allen', 'en_mullay', 'gas_basicity', 'geochemical_class', 'goldschmidt_class', 'group', 'group_id', 'inchi', 'ionenergies', 'ionic_radii', 'is_monoisotopic', 'isotopes', 'jmol_color', 'lattice_structure', 'metadata', 'metallic_radius', 'metallic_radius_c12', 'molcas_gv_color', 'name', 'name_origin', 'nist_webbook_url', 'oxistates', 'phase_transitions', 'proton_affinity', 'recycling_rate', 'registry', 'scattering_factors', 'sconst', 'screening_constants', 'series', 'sources', 'substitutability', 'symbol', 'top_3_producers', 'top_3_reserve_holders', 'uses', 'vdw_radius_batsanov', 'vdw_radius_bondi', 'vdw_radius_dreiding', 'vdw_radius_rt', 'vdw_radius_truhlar']
METHOD_ATTRIBUTES = ['electronegativity', 'electronegativity_allen', 'electronegativity_allred_rochow', 'electronegativity_cottrell_sutton', 'electronegativity_ghosh', 'electronegativity_gordy', 'electronegativity_li_xue', 'electronegativity_martynov_batsanov', 'electronegativity_mullay', 'electronegativity_mulliken', 'electronegativity_nagle', 'electronegativity_pauling', 'electronegativity_sanderson', 'electronegativity_scales', 'electrophilicity', 'hardness', 'init_on_load', 'isotope', 'mass_str', 'nvalence', 'oxidation_states', 'oxides', 'softness', 'zeff']
FLOAT_ATTRIBUTES = ['abundance_crust', 'abundance_sea', 'atomic_number', 'atomic_radius', 'atomic_radius_rahm', 'atomic_volume', 'atomic_weight', 'atomic_weight_uncertainty', 'boiling_point', 'c6_gb', 'covalent_radius', 'covalent_radius_cordero', 'covalent_radius_pyykko', 'covalent_radius_pyykko_double', 'covalent_radius_pyykko_triple', 'density', 'dipole_polarizability', 'dipole_polarizability_unc', 'discovery_year', 'electron_affinity', 'electrons', 'en_ghosh', 'en_gunnarsson_lundqvist', 'en_miedema', 'en_pauling', 'en_robles_bartolotti', 'evaporation_heat', 'fusion_heat', 'glawe_number', 'heat_of_formation', 'is_radioactive', 'lattice_constant', 'mass', 'mass_number', 'melting_point', 'mendeleev_number', 'miedema_electron_density', 'miedema_molar_volume', 'molar_heat_capacity', 'neutrons', 'period', 'pettifor_number', 'political_stability_of_top_producer', 'political_stability_of_top_reserve_holder', 'price_per_kg', 'production_concentration', 'protons', 'relative_supply_risk', 'reserve_distribution', 'specific_heat', 'specific_heat_capacity', 'thermal_conductivity', 'vdw_radius', 'vdw_radius_alvarez', 'vdw_radius_mm3', 'vdw_radius_uff']
DEFAULT_ATTRIBUTES = ["atomic_radius", 'electronegativity_pauling', "protons", "oxidation_states_one_hot", "first_ionisation_energy"]

DEFAULT_OVERRIDES = {
    "Eu": {
        'electronegativity_pauling': 1.2 # https://en.wikipedia.org/wiki/Europium
    },
    "Pm": {
        'electronegativity_pauling': 1.13 # https://en.wikipedia.org/wiki/Promethium
    },
    "Tb": {
        'electronegativity_pauling': 1.21 # https://en.wikipedia.org/wiki/Electronegativities_of_the_elements_(data_page)
    },
    "Yb": {
        'electronegativity_pauling': 1.1 # https://en.wikipedia.org/wiki/Ytterbium
    },
}


def oxidation_states_one_hot(element: Element, element_dict: Dict[str, float]) -> Dict[str, float]:
    ox_states = element.oxidation_states()
    for i in range(1, 8):
        if i in ox_states:
            element_dict[f"can_be_{i}"] = 1
        else: 
            element_dict[f"can_be_{i}"] = 0

    return element_dict


def first_ionisation_energy(element: Element, element_dict: Dict[str, float]) -> Dict[str, float]:
    element_dict["first_ionisation_energy"] = element.ionenergies[1]
    return element_dict


CUSTOM_FUNCS = {
    'oxidation_states_one_hot': oxidation_states_one_hot, 
    'first_ionisation_energy': first_ionisation_energy
}