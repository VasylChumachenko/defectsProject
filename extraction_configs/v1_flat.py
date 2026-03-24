"""
v1_flat – Original flat schema.

All synthesis conditions in a single level per sample.
No distinction between backbone-formation temperature and modification temperature.
"""

VERSION = "v1_flat"
DESCRIPTION = (
    "Original flat schema: one set of conditions per sample. "
    "temperature_C captures the highest calcination temperature reported "
    "for that sample, which for two-step processes may be the modification "
    "temperature rather than the backbone-formation temperature."
)

# ── Tag schema ───────────────────────────────────────────────────────────────

TAG_NAMES = [
    "precursor_family",
    "calcination_temperature_bin",
    "atmosphere_class",
    "primary_route",
    "defect_introduction_mode",
    "dopant_class",
    "morphology_form",
]

TAG_ALLOWED = {
    "precursor_family": ["urea", "thiourea", "cyanamide", "melamine", "other"],
    "calcination_temperature_bin": ["lt520", "520_560", "560_600", "gt600"],
    "atmosphere_class": [
        "inert", "N2", "air", "reducing",
        "etching_reactive", "co2_generated", "unknown",
    ],
    "primary_route": [
        "direct_thermal", "hydro_solvothermal_pre",
        "supramolecular_preassembly", "template_assisted",
        "unknown_or_other",
    ],
    "defect_introduction_mode": [
        "none_or_baseline", "two_step_overcalcination",
        "chemical_vapor_etching", "gas_assisted_etching",
        "dopant_induced",
    ],
    "dopant_class": ["none", "nonmetal", "metal", "codoped_or_multi"],
    "morphology_form": [
        "bulk", "nanosheets_ultrathin", "porous_holey",
        "tubular", "3d_macroporous", "unknown",
    ],
}

TAG_DEFAULTS = {
    "precursor_family": "other",
    "calcination_temperature_bin": "520_560",
    "atmosphere_class": "unknown",
    "primary_route": "unknown_or_other",
    "defect_introduction_mode": "none_or_baseline",
    "dopant_class": "none",
    "morphology_form": "unknown",
}

EXTRA_CATEGORICAL = {
    "synthesis_method": {
        "allowed": [
            "thermal_polymerization", "solvothermal_exfoliation",
            "supramolecular", "other",
        ],
        "default": "other",
    },
    "duration_bin": {
        "allowed": ["lt2h", "2_4h", "4_8h", "gt8h"],
        "default": "2_4h",
    },
}

# ── Per-sample fields expected in LLM output ─────────────────────────────────

SAMPLE_FIELDS = [
    "sample_name",
    "file_match",
    "sample_type",
    "co_precursor",
    "dopant_element",
    "synthesis_method",
    "temperature_C",
    "heating_rate_C_min",
    "duration_bin",
    "atmosphere",
    "pre_treatment",
    "post_treatment",
    "defect_type",
    "defect_formation_method",
    "special_notes",
    # 7 tags
    "precursor_family",
    "calcination_temperature_bin",
    "atmosphere_class",
    "primary_route",
    "defect_introduction_mode",
    "dopant_class",
    "morphology_form",
]

SAMPLE_TYPE_VALUES = ["reference", "modified", "doped", "defective"]

# ── Prompt template ──────────────────────────────────────────────────────────

PROMPT_TEMPLATE = r"""You are an expert in g-C3N4 (graphitic carbon nitride) materials science.

ARTICLE TITLE: {title}

EXPERIMENTAL TEXT:
{text}

SPECTRAL DATA FILE NAMES for this article (lowercase, dash-separated):
{file_samples}

=== TASK ===
1. Extract synthesis conditions for EACH SAMPLE described in the article.
2. For each sample, assign a "file_match" — which file name from the list above corresponds to this sample. Use null if no match.
   MATCHING RULES:
   - File names are lowercase, dash-separated simplifications of article sample names.
   - Examples: "bulk g-C3N4" → "bulk-g-c3n4", "CN-MIX-1" → "cn-mix1", "Ns-g-C3N4" → "ns-g-c3n4".
   - CRITICAL: Each file name can be assigned to AT MOST ONE sample (no duplicates!).
   - CRITICAL: Only match when you are confident. If unsure, use null — a wrong match is worse than null.
   - Compare carefully: "CN-AT" ≠ "CN-T", "CN-MIX-1" ≠ "CN-AT". Match the FULL name, not a substring.
3. For each sample, assign 7 standardized tags (see rules below).
4. Extract the DRS instrument (UV-Vis spectrophotometer) used in this article.

=== SAMPLE FIELDS ===
For each sample provide:
- sample_name: identifier used in the paper (e.g. "CN-500", "pristine g-C3N4")
- file_match: matching file name from the list above, or null
- sample_type: "reference" | "modified"
- co_precursor: secondary precursor or additive used alongside the main precursor, or "none"
- dopant_element: element symbol (S, P, Fe, La...) or "none"
- synthesis_method: "thermal_polymerization" (includes calcination, thermal polycondensation, pyrolysis at high T) | "solvothermal_exfoliation" (hydrothermal, solvothermal, autoclave-based) | "supramolecular" (supramolecular preassembly, e.g. cyanuric acid + melamine) | "other"
- temperature_C: max calcination/polymerization temperature as number, or null
- heating_rate_C_min: heating rate as number, or null
- duration_bin: total synthesis duration bin: "lt2h" (<2 hours) | "2_4h" (2-4 hours) | "4_8h" (4-8 hours) | "gt8h" (>8 hours)
- atmosphere: gas atmosphere as string (N2, Ar, air, etc.) or "unknown"
- pre_treatment: any pre-treatment steps or "none"
- post_treatment: any post-treatment (exfoliation, etching, washing, etc.) or "none"
- defect_type: specific defect type or "none"
- defect_formation_method: how defects were introduced or "none"
- special_notes: any other important conditions or "none"

=== 7 STANDARDIZED TAGS (per sample) ===

1. precursor_family — main condensation precursor (the primary one that forms the g-C3N4 backbone)
   Allowed: urea, thiourea, cyanamide, melamine, other
   Rule: cyanamide includes dicyandiamide (DCDA). Identify the PRIMARY g-C3N4 precursor even when mixed with additives/dopant sources. E.g. if DCDA + (NH4)2S2O3, precursor_family=cyanamide (the (NH4)2S2O3 is a dopant source, not a precursor). If none of the listed → other

2. calcination_temperature_bin — highest calcination temp (NOT drying 60-105°C)
   Allowed: lt520 (<520°C), 520_560 (520-559°C), 560_600 (560-599°C), gt600 (≥600°C)
   Rule: MUST be consistent with temperature_C for this sample. 550→520_560, 560→560_600, 600→gt600. Use THIS sample's temperature, not the article's max.

3. atmosphere_class — gas during thermal treatment
   Allowed: inert, N2, air, reducing, etching_reactive, co2_generated, unknown
   Rule: NH3→etching_reactive, N2→N2, H2→reducing, Ar/He→inert, CO2/NaHCO3→co2_generated, air/ambient→air

4. primary_route — main synthesis path for THIS sample
   Allowed: direct_thermal, hydro_solvothermal_pre, supramolecular_preassembly, template_assisted, unknown_or_other
   Rule: autoclave/Teflon→hydro_solvothermal_pre, cyanuric acid+melamine→supramolecular_preassembly, template→template_assisted, else→direct_thermal

5. defect_introduction_mode — how defects were introduced in THIS sample
   Allowed: none_or_baseline, two_step_overcalcination, chemical_vapor_etching, gas_assisted_etching, dopant_induced
   Rule: Mg vapor→chemical_vapor_etching, two-step/re-calcined→two_step_overcalcination, NH3/CO2 etching→gas_assisted_etching, doping→dopant_induced, else→none_or_baseline

6. dopant_class — doping type for THIS sample
   Allowed: none, nonmetal, metal, codoped_or_multi
   Rule: P/S/B/F/O→nonmetal, Fe/Cu/Ag/La/Pt→metal, ≥2 elements→codoped_or_multi

7. morphology_form — declared morphology for THIS sample
   Allowed: bulk, nanosheets_ultrathin, porous_holey, tubular, 3d_macroporous, unknown
   Rule: ultrathin/nanosheet→nanosheets_ultrathin, porous/holey→porous_holey, 3D→3d_macroporous, tubular→tubular, else→bulk

=== OUTPUT FORMAT ===
Return ONLY valid JSON:

{{
  "samples": [
    {{
      "sample_name": "...",
      "file_match": "..." or null,
      "sample_type": "...",
      "co_precursor": "...",
      "dopant_element": "...",
      "synthesis_method": "...",
      "temperature_C": number or null,
      "heating_rate_C_min": number or null,
      "duration_bin": "...",
      "atmosphere": "...",
      "pre_treatment": "...",
      "post_treatment": "...",
      "defect_type": "...",
      "defect_formation_method": "...",
      "special_notes": "...",
      "precursor_family": "...",
      "calcination_temperature_bin": "...",
      "atmosphere_class": "...",
      "primary_route": "...",
      "defect_introduction_mode": "...",
      "dopant_class": "...",
      "morphology_form": "..."
    }}
  ],
  "drs_instrument": "brand and model of UV-Vis/DRS spectrophotometer, or 'unknown'",
  "general_notes": "overall synthesis approach or important context"
}}

IMPORTANT RULES:
1. Extract ALL distinct samples (including reference/pristine).
2. For temperature/concentration series, create separate entries for each variant.
3. Use null for unknown numeric fields (temperature_C, heating_rate_C_min).
4. "reference" = pristine/bulk/unmodified g-C3N4 for comparison.
5. "modified" = exfoliated, porous, nanosheet without doping.
6. "doped" = element doping. "defective" = intentional vacancy/defect creation.
7. All categorical fields (tags, synthesis_method, duration_bin) must use ONLY allowed values.
8. NEVER assign the same file_match to more than one sample. Verify no duplicates in your output.
9. duration_bin: estimate total thermal treatment time. If multiple steps, sum them. lt2h=<2h, 2_4h=2-4h, 4_8h=4-8h, gt8h=>8h.
10. calcination_temperature_bin MUST match temperature_C: 550→520_560, 560→560_600, 600→gt600 etc.
11. precursor_family = the main g-C3N4 backbone precursor. Dopant sources, additives, co-reactants are NOT the precursor.

JSON:"""











