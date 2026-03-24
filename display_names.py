"""
Shared display-name dictionaries for scientific visualisations.

Maps internal tag / value identifiers to human-readable labels
suitable for figures, axes, legends, and tables.
"""

# ─── Tag (column) names → axis / title labels ────────────────────────────
TAG_DISPLAY = {
    'precursor_family':           'Precursor',
    'calcination_temperature_bin': 'Temperature range',
    'atmosphere_class':           'Atmosphere',
    'primary_route':              'Synthesis route',
    'defect_introduction_mode':   'Defect introduction',
    'dopant_class':               'Dopant class',
    'morphology_form':            'Morphology',
    'synthesis_method':           'Synthesis method',
    'duration_bin':               'Duration',
    'sample_type':                'Sample type',
    'temperature_C':              'Temperature (°C)',
    'defect_type':                'Defect type',
    'drs_instrument':             'DRS instrument',
    # v2_staged: backbone
    'backbone_method':            'Backbone method',
    'backbone_temperature_C':     'Backbone temperature (°C)',
    'backbone_heating_rate_C_min': 'Backbone heating rate (°C/min)',
    'backbone_duration_bin':      'Backbone duration',
    'backbone_atmosphere':        'Backbone atmosphere',
    # v2_staged: modification
    'is_post_processed':          'Post-processed?',
    'mod_method':                 'Modification method',
    'mod_temperature_C':          'Modification temperature (°C)',
    'mod_atmosphere':             'Modification atmosphere',
    'mod_atmosphere_class':       'Modification atmosphere (class)',
    # Transition-specific: mod_ prefixed columns from transitions_core
    'mod_mod_method':              'Post-proc. method',
    'mod_mod_atmosphere_class':    'Post-proc. atmosphere',
    'mod_duration_bin':           'Modification duration',
    'mod_agent':                  'Modification agent',
    'mod_notes':                  'Modification notes',
}

# ─── Tag value names → legend / tick labels ──────────────────────────────
VALUE_DISPLAY = {
    # precursor_family
    'melamine':      'Melamine',
    'urea':          'Urea',
    'thiourea':      'Thiourea',
    'cyanamide':     'Cyanamide / DCDA',
    'other':         'Other',

    # calcination_temperature_bin
    'lt520':         '< 520 °C',
    '520_560':       '520–560 °C',
    '560_600':       '560–600 °C',
    'gt560':         '> 560 °C',
    'gt600':         '> 600 °C',

    # atmosphere_class
    'air':              'Air / not specified',
    'N2':               'N₂',
    'inert':            'Inert (Ar)',
    'reducing':         'Reducing (H₂ / NH₃)',
    'other_atmosphere': 'Other atmosphere',
    'co2_generated':    'CO₂-generating',
    'unknown':          'Not specified',

    # primary_route
    'direct_thermal':               'Direct thermal',
    'hydro_solvothermal_pre':       'Hydro-/solvo-thermal',
    'supramolecular_preassembly':   'Supramolecular',
    'other_route':                  'Other route',
    'template_assisted':            'Template-assisted',
    'unknown_or_other':             'Other / not specified',

    # defect_introduction_mode
    'none_or_baseline':             'None (baseline)',
    'dopant_induced':               'Dopant-induced',
    'etching':                      'Etching (gas / chemical)',
    'gas_assisted_etching':         'Gas-assisted etching',
    'chemical_vapor_etching':       'Chemical vapor etching',
    'two_step_overcalcination':     'Two-step / over-calcination',

    # dopant_class
    'none':              'Undoped',
    'metal':             'Metal dopant',
    'nonmetal':          'Non-metal dopant',
    'codoped_or_multi':  'Co-doped / multi',

    # morphology_form
    'bulk':                  'Bulk',
    'nanosheets_ultrathin':  'Nanosheets / ultra-thin',
    'porous_holey':          'Porous / holey',
    'tubular':               'Tubular',
    '3d_macroporous':        '3-D macroporous',

    # synthesis_method / backbone_method
    'thermal_polymerization':    'Thermal polymerisation',
    'solvothermal_exfoliation':  'Solvothermal exfoliation',
    'solvothermal':              'Solvothermal',
    'supramolecular':            'Supramolecular assembly',

    # duration_bin
    'lt2h':   '< 2 h',
    '2_4h':   '2–4 h',
    '4_8h':   '4–8 h',
    'gt8h':   '> 8 h',

    # sample_type
    'reference': 'Reference',
    'modified':  'Modified',
    'doped':     'Doped',
    'defective': 'Defective',

    # mod_method (v2_staged)
    're_calcination':    'Re-calcination',
    'chemical_etching':  'Chemical etching',
    'gas_treatment':     'Gas treatment',
    'wet_impregnation':  'Wet impregnation',
    'ultrasonication':   'Ultrasonication',
    'ball_milling':      'Ball milling',
    'plasma_treatment':  'Plasma treatment',

    # mod_atmosphere_class (v2_staged)
    'liquid':   'Liquid medium',
    'vacuum':   'Vacuum',

    # transition types
    'post_processing':      'Post-processing',
    'synthesis_variation':  'Synthesis variation',
}

# ─── Spectral feature display ────────────────────────────────────────────
FEATURE_DISPLAY = {
    'E_g_eV':  r'$E_g$ (eV)',
    'E_u_meV': r'$E_u$ (meV)',
    'A_sub':   r'$A_{sub}$',
}


# ─── Helper functions ────────────────────────────────────────────────────

def tag_label(tag_name: str) -> str:
    """Return display label for a tag column name."""
    return TAG_DISPLAY.get(tag_name, tag_name)


def value_label(value: str) -> str:
    """Return display label for a tag value."""
    if not isinstance(value, str):
        return str(value)
    return VALUE_DISPLAY.get(value, value.replace('_', ' ').title())


def rename_series_values(series):
    """Return a copy of a pandas Series with values mapped through VALUE_DISPLAY."""
    return series.map(lambda v: value_label(str(v)) if pd.notna(v) else v)


def feature_label(feat_name: str) -> str:
    """Return display label for a spectral feature."""
    return FEATURE_DISPLAY.get(feat_name, feat_name)


# Allow 'from display_names import *'
import pandas as pd  # noqa: E402 – for rename_series_values type hint
__all__ = [
    'TAG_DISPLAY', 'VALUE_DISPLAY', 'FEATURE_DISPLAY',
    'tag_label', 'value_label', 'rename_series_values', 'feature_label',
]







