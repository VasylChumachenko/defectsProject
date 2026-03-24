"""
v2_staged – Two-stage schema: backbone formation + optional modification.

Separates the g-C₃N₄ backbone-formation conditions from any subsequent
modification / post-treatment.  This resolves the ambiguity where e.g.
temperature_C = 300 °C actually refers to a post-synthesis reduction step
while the backbone was formed at 600 °C.

Key changes vs v1_flat
──────────────────────
• temperature_C             → backbone_temperature_C
• atmosphere                → backbone_atmosphere
• duration_bin              → backbone_duration_bin
• synthesis_method          → backbone_method
• NEW  is_post_processed    – boolean flag
• NEW  mod_temperature_C    – modification-step temperature
• NEW  mod_atmosphere       – modification-step atmosphere
• NEW  mod_atmosphere_class – classified modification atmosphere (TAG)
• NEW  mod_duration_bin     – modification-step duration
• NEW  mod_method           – how modification was performed (TAG)
• NEW  mod_agent            – chemical agent / medium used
• calcination_temperature_bin  now derived from backbone_temperature_C
• atmosphere_class             now derived from backbone_atmosphere

Key changes in v2_staged refresh
─────────────────────────────────
• sample_type collapsed to binary: "reference" | "modified"
• defect_introduction_mode REMOVED (ambiguous, low-signal tag)
• mod_method PROMOTED from EXTRA_CATEGORICAL to TAG_NAMES (8 tags total)
"""

VERSION = "v2_staged"
DESCRIPTION = (
    "Two-stage schema: backbone-formation conditions + optional modification. "
    "temperature fields are split into backbone_temperature_C (always = the "
    "g-C₃N₄ polycondensation temperature) and mod_temperature_C (the post-"
    "processing temperature, if different). Tags are derived from backbone."
)

# ── Tag schema (6 backbone/universal + 2 modification) ───────────────────────

TAG_NAMES = [
    "precursor_family",
    "calcination_temperature_bin",
    "atmosphere_class",
    "primary_route",
    "dopant_class",
    "morphology_form",
    "mod_method",
    "mod_atmosphere_class",
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
    "dopant_class": ["none", "nonmetal", "metal", "codoped_or_multi"],
    "morphology_form": [
        "bulk", "nanosheets_ultrathin", "porous_holey",
        "tubular", "3d_macroporous", "unknown",
    ],
    "mod_method": [
        "none",
        "re_calcination",
        "solvothermal",
        "chemical_etching",
        "gas_treatment",
        "wet_impregnation",
        "ultrasonication",
        "ball_milling",
        "plasma_treatment",
        "other",
    ],
    "mod_atmosphere_class": [
        "none", "inert", "N2", "air", "reducing",
        "etching_reactive", "liquid", "vacuum", "unknown",
    ],
}

TAG_DEFAULTS = {
    "precursor_family": "other",
    "calcination_temperature_bin": "520_560",
    "atmosphere_class": "unknown",
    "primary_route": "unknown_or_other",
    "dopant_class": "none",
    "morphology_form": "unknown",
    "mod_method": "none",
    "mod_atmosphere_class": "none",
}

EXTRA_CATEGORICAL = {
    "backbone_method": {
        "allowed": [
            "thermal_polymerization", "solvothermal",
            "supramolecular", "other",
        ],
        "default": "other",
    },
    "backbone_duration_bin": {
        "allowed": ["lt2h", "2_4h", "4_8h", "gt8h"],
        "default": "2_4h",
    },
    "mod_duration_bin": {
        "allowed": ["none", "lt2h", "2_4h", "4_8h", "gt8h"],
        "default": "none",
    },
}

# ── Per-sample fields expected in LLM output ─────────────────────────────────

SAMPLE_FIELDS = [
    "sample_name",
    "file_match",
    "sample_type",
    # Backbone formation
    "precursor",
    "co_precursor",
    "dopant_element",
    "backbone_method",
    "backbone_temperature_C",
    "backbone_heating_rate_C_min",
    "backbone_duration_bin",
    "backbone_atmosphere",
    # Post-processing / modification
    "is_post_processed",
    "mod_method",
    "mod_temperature_C",
    "mod_atmosphere",
    "mod_duration_bin",
    "mod_agent",
    "mod_notes",
    # Other
    "defect_type",
    "morphology_notes",
    "special_notes",
    # 6+2 tags (mod_method already listed above under modification)
    "precursor_family",
    "calcination_temperature_bin",
    "atmosphere_class",
    "primary_route",
    "dopant_class",
    "morphology_form",
    "mod_atmosphere_class",
]

SAMPLE_TYPE_VALUES = ["reference", "modified"]

# ── Legacy mapping (v2 → v1 column names) ────────────────────────────────────
# Used by downstream scripts that still expect v1_flat column names.
# After LLM extraction with v2, these legacy columns are computed automatically.

V2_TO_V1 = {
    # v1 name             ← v2 source
    "temperature_C":       "backbone_temperature_C",
    "heating_rate_C_min":  "backbone_heating_rate_C_min",
    "atmosphere":          "backbone_atmosphere",
    "synthesis_method":    "backbone_method",
    "duration_bin":        "backbone_duration_bin",
}

# ── Prompt template ──────────────────────────────────────────────────────────

PROMPT_TEMPLATE = r"""You are an expert in g-C3N4 (graphitic carbon nitride) materials science.

ARTICLE TITLE: {title}

EXPERIMENTAL TEXT:
{text}

SPECTRAL DATA FILE NAMES for this article (lowercase, dash-separated):
{file_samples}

=== TASK ===
1. Extract synthesis conditions for EACH SAMPLE described in the article.
2. Match each sample to a file name from the list above (or null if unsure).
3. For each sample, SEPARATE the backbone-formation step from any subsequent
   modification / post-treatment step.
4. Assign 6+2 standardized tags (see rules below).
5. Extract the DRS instrument used in this article.

=== KEY CONCEPT: BACKBONE vs MODIFICATION ===

Many g-C3N4 studies involve TWO stages:
  STAGE 1 — BACKBONE FORMATION: g-C3N4 is synthesized from a precursor
            (melamine, urea, DCDA, etc.) via thermal polycondensation.
            This is typically at 500–650°C in N2/air/Ar.
  STAGE 2 — MODIFICATION (optional): the already-formed g-C3N4 powder
            is processed further: re-heated under different gas, treated
            in solution, loaded with metal nanoparticles, exfoliated,
            plasma-treated, etc.
            Temperature can be very different (e.g. 200–400°C for reduction,
            room temperature for ultrasonication).

For REFERENCE samples — there is usually only Stage 1.
For MODIFIED/DOPED/DEFECTIVE samples — there may be Stage 1 + Stage 2.

CRITICAL RULES for deciding is_post_processed:
• is_post_processed = true  — when a SEPARATE second step was applied to
  already-formed g-C3N4 (e.g. re-calcination, wet impregnation, exfoliation,
  gas treatment, plasma treatment).
• is_post_processed = false — when the sample differs from reference only
  in its one-step synthesis conditions (different precursor, different T,
  different atmosphere, different additive).

HANDLING UNKNOWN BACKBONE CONDITIONS:
• If the article describes a modified sample but does NOT explicitly state
  how the starting g-C3N4 was made (backbone conditions), set all
  backbone_* fields to null / "unknown".  Do NOT guess or copy from
  other articles.
• If the article says "commercial g-C3N4 was purchased" or similar,
  set backbone_method = "other", backbone_temperature_C = null,
  backbone_atmosphere = "unknown".
• If the article says "g-C3N4 was prepared according to ref. [X]" without
  details, set backbone fields to null / "unknown".

=== SAMPLE FIELDS ===
For each sample provide:

--- Identification ---
- sample_name: identifier used in the paper (e.g. "CN-500", "pristine g-C3N4")
- file_match: matching file name from the list above, or null
- sample_type: "reference" | "modified"
  Rule: "reference" = pristine/unmodified g-C3N4 used as baseline.
        "modified" = ANY sample that differs from reference — includes doped,
        defective, exfoliated, post-treated, differently-synthesized variants.
        When in doubt → "modified".

--- Backbone formation (Stage 1) ---
- precursor: main precursor name as written in the paper (e.g. "melamine",
  "dicyandiamide"), or "unknown" if not stated
- co_precursor: secondary precursor / additive used IN backbone synthesis, or "none"
- dopant_element: element symbol (S, P, Fe, La...) or "none"
- backbone_method: "thermal_polymerization" | "solvothermal" | "supramolecular" | "other"
  Rule: thermal polymerization = calcination / polycondensation / pyrolysis at high T;
        solvothermal = hydrothermal / autoclave / Teflon-lined;
        supramolecular = supramolecular preassembly (e.g. cyanuric acid + melamine)
- backbone_temperature_C: the FINAL HOLDING (regime) calcination temperature (°C) or null
  CRITICAL RULES:
  • This is the MAXIMUM holding temperature at which the g-C3N4 backbone
    was actually formed — NOT intermediate ramp temperatures.
  • Heating programs often describe stepwise ramps: e.g. "heated to 300°C
    (2h), then to 500°C (2h), then to 600°C (4h)". In this case,
    backbone_temperature_C = 600 (the final regime temperature).
  • If the paper says "heated at 5°C/min to 550°C and held for 4h",
    backbone_temperature_C = 550.
  • UNIT CONVERSION: If the article reports temperature in Kelvin (K),
    CONVERT to °C by subtracting 273. Example: "823 K" → 550°C.
    Always output backbone_temperature_C in °C.
  • For two-step processes, this is the Stage 1 (backbone) final holding
    temperature (often same as the reference sample in that article).
  • If unknown → null.
- backbone_heating_rate_C_min: heating rate (°C/min) or null
- backbone_duration_bin: "lt2h" | "2_4h" | "4_8h" | "gt8h"
  If unknown → "2_4h" (default)
- backbone_atmosphere: gas atmosphere during backbone formation
  (e.g. "N2", "Ar", "air", "NH3"). If unknown → "unknown"

--- Modification / post-treatment (Stage 2, optional) ---
- is_post_processed: true if a SEPARATE modification step was applied, false otherwise
- mod_method: "none" | "re_calcination" | "solvothermal" | "chemical_etching" | "gas_treatment" | "wet_impregnation" | "ultrasonication" | "ball_milling" | "plasma_treatment" | "other"
  Rules:
    re-heating at different T/atmosphere        → "re_calcination"
    hydro/solvothermal post-treatment           → "solvothermal"
    KOH/HCl/acid etching                       → "chemical_etching"
    H2/NH3/CO2 gas-phase treatment              → "gas_treatment"
    metal nanoparticle loading via wet chemistry → "wet_impregnation"
    liquid-phase exfoliation / sonication        → "ultrasonication"
    mechanical grinding / ball milling           → "ball_milling"
    Ar/N2/O2 plasma treatment                    → "plasma_treatment"
- mod_temperature_C: FINAL HOLDING temperature of the modification step (°C) or null
  Same rule: use the regime/holding temperature, not intermediate ramps.
  If the article reports in Kelvin (K), convert to °C (subtract 273).
  If modification is at room temperature (e.g. ultrasonication) → null
- mod_atmosphere: gas or medium during modification (e.g. "H2", "Ar", "water",
  "ethanol", "KOH solution") or "none"
- mod_duration_bin: "none" | "lt2h" | "2_4h" | "4_8h" | "gt8h"
- mod_agent: chemical agent used (H2, NH3, KOH, H2PtCl6, CuCl2...) or "none"
- mod_notes: brief description of what the modification does, or "none"

--- Other ---
- defect_type: specific defect type if reported (nitrogen vacancy, carbon vacancy,
  cyano defect, etc.) or "none"
- morphology_notes: declared morphology (nanosheets, porous, tubular, etc.) or "none"
- special_notes: any other important conditions or "none"

=== 6+2 STANDARDIZED TAGS (per sample) ===

Tags 1–4 describe the BACKBONE formation, NOT the modification step.
Tags 5–6 describe the sample in general.
Tags 7–8 describe the MODIFICATION step.

1. precursor_family — main condensation precursor for the g-C3N4 backbone
   Allowed: urea, thiourea, cyanamide, melamine, other
   Rule: cyanamide includes dicyandiamide (DCDA). If DCDA + additive → cyanamide.
   If backbone precursor is unknown → "other"

2. calcination_temperature_bin — BACKBONE final holding temperature
   Allowed: lt520 (<520°C), 520_560, 560_600, gt600
   Rule: MUST match backbone_temperature_C (the final regime temperature).
   550→520_560, 560→560_600, 600→gt600.
   CRITICAL: Use backbone_temperature_C, NOT mod_temperature_C.
   If backbone_temperature_C is null → "520_560" (default).

3. atmosphere_class — gas during BACKBONE thermal treatment
   Allowed: inert, N2, air, reducing, etching_reactive, co2_generated, unknown
   Rule: Derived from backbone_atmosphere.
   NH3 → etching_reactive, N2 → N2, H2 → reducing, Ar/He → inert, air → air.
   If backbone_atmosphere is unknown → "unknown".

4. primary_route — main synthesis path for backbone formation
   Allowed: direct_thermal, hydro_solvothermal_pre, supramolecular_preassembly,
            template_assisted, unknown_or_other

5. dopant_class — doping type
   Allowed: none, nonmetal, metal, codoped_or_multi
   Rule: P/S/B/F/O → nonmetal, Fe/Cu/Ag/La/Pt → metal, ≥2 → codoped_or_multi

6. morphology_form — declared morphology
   Allowed: bulk, nanosheets_ultrathin, porous_holey, tubular, 3d_macroporous, unknown

7. mod_method — how modification/post-treatment was performed
   Allowed: none, re_calcination, solvothermal, chemical_etching, gas_treatment,
            wet_impregnation, ultrasonication, ball_milling, plasma_treatment, other
   Rules:
     is_post_processed=false                       → "none"
     re-heating at different T/atmosphere           → "re_calcination"
     hydro/solvothermal post-treatment              → "solvothermal"
     KOH/HCl/acid etching                          → "chemical_etching"
     H2/NH3/CO2 gas-phase treatment                 → "gas_treatment"
     metal nanoparticle loading via wet chemistry    → "wet_impregnation"
     liquid-phase exfoliation / sonication           → "ultrasonication"
     mechanical grinding / ball milling              → "ball_milling"
     Ar/N2/O2 plasma treatment                       → "plasma_treatment"

8. mod_atmosphere_class — classified atmosphere/medium during MODIFICATION
   Allowed: none, inert, N2, air, reducing, etching_reactive, liquid, vacuum, unknown
   Rule: Derived from mod_atmosphere.
   If is_post_processed=false or mod_atmosphere="none" → "none".
   H2 → reducing, NH3 → etching_reactive, Ar → inert, water/ethanol/KOH → liquid.

=== FILE MATCHING RULES ===
- File names are lowercase, dash-separated simplifications of article sample names.
- Examples: "bulk g-C3N4" → "bulk-g-c3n4", "CN-MIX-1" → "cn-mix1"
- CRITICAL: Each file name can be assigned to AT MOST ONE sample (no duplicates!).
- Only match when confident. Wrong match is worse than null.

=== OUTPUT FORMAT ===
Return ONLY valid JSON:

{{
  "samples": [
    {{
      "sample_name": "...",
      "file_match": "..." or null,
      "sample_type": "reference" or "modified",
      "precursor": "...",
      "co_precursor": "...",
      "dopant_element": "...",
      "backbone_method": "...",
      "backbone_temperature_C": number or null,
      "backbone_heating_rate_C_min": number or null,
      "backbone_duration_bin": "...",
      "backbone_atmosphere": "...",
      "is_post_processed": true/false,
      "mod_method": "...",
      "mod_temperature_C": number or null,
      "mod_atmosphere": "...",
      "mod_duration_bin": "...",
      "mod_agent": "...",
      "mod_notes": "...",
      "defect_type": "...",
      "morphology_notes": "...",
      "special_notes": "...",
      "precursor_family": "...",
      "calcination_temperature_bin": "...",
      "atmosphere_class": "...",
      "primary_route": "...",
      "dopant_class": "...",
      "morphology_form": "...",
      "mod_atmosphere_class": "..."
    }}
  ],
  "drs_instrument": "brand and model of UV-Vis/DRS spectrophotometer, or 'unknown'",
  "general_notes": "overall synthesis approach or important context"
}}

IMPORTANT RULES:
1. Extract ALL distinct samples (including reference/pristine).
2. For temperature/concentration series, create separate entries for each variant.
3. Use null for unknown numeric fields.
4. "reference" = pristine/unmodified g-C3N4 used as baseline for comparison.
5. "modified" = ANY sample that differs from reference (doped, defective,
   exfoliated, post-treated, or differently-synthesized variant).
6. All categorical fields must use ONLY allowed values.
8. NEVER assign the same file_match to more than one sample.
9. For two-step processes: backbone_* = Stage 1, mod_* = Stage 2.
10. For one-step processes: is_post_processed=false, mod_*=null/none.
11. calcination_temperature_bin MUST match backbone_temperature_C.
12. If backbone conditions are unknown (not described), use null / "unknown" — do NOT guess.

JSON:"""
