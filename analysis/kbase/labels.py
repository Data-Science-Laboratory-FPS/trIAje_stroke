# ===============================================================
# Shared human-readable display labels for Table 1 generation and
# explainability outputs (feature importance, SHAP plots).
# ===============================================================

# Labels shared across all demand types.
COMMON_LABELS = {
    "age": "Age (years)",
    "age_15_24": "Youth: 15-24 years",
    "age_25_44": "Young Adults: 25-44 years",
    "age_45_59": "Middle-aged Adults: 45-59 years",
    "age_60_74": "Elderly: 60-74 years",
    "age_75_plus": "Seniors: >74 years",
    "year": "Year of the call",
    "sex": "Sex (female)",
    "triage": "Structured triage performed",
    "p1_assigned": "Assigned P1 priority",
    "p1_real_emerg": "Real emergency P1 priority",
    "p1_real_bps": "BPS P1 priority",
    "p1_real_emerg_bps": "Emergency or BPS P1 priority",
    "has_icd_emerg": "Emergency ICD diagnosis available",
    "has_history": "Past medical history available",
    "has_history_old": "Legacy past medical history available",
    "has_problems_old": "Legacy active problems available",
    "has_com": "Legacy history or active problem available",
    "has_med": "Medication history available",
    "alert_receiver_112": "112 emergency line",
    "alert_receiver_user": "User or patient",
    "alert_receiver_pol_fg": "Police or Civil Guard",
    "alert_receiver_hs": "Health services",
    "alert_receiver_tele": "Tele-assistance",
    "alert_receiver_others": "Other alert receiver",
    "location_patient_home": "Patient at home",
    "location_patient_public_road": "Patient on public road",
    "location_patient_other": "Other patient location",
    "time_of_day_early_morning": "Early morning (00-05 h)",
    "time_of_day_morning": "Morning (06-11 h)",
    "time_of_day_afternoon": "Afternoon (12-17 h)",
    "time_of_day_night": "Night (18-23 h)",
}

# Triage protocol question labels ("1a", "1b", ...), specific to each demand type.
TRIAGE_LABELS_BY_DEMAND = {
    # 54: Stroke / ACV (A54 GCTT, questions P1-P6)
    54: {
        "1a": "Q1A Eye opening - Yes",
        "1b": "Q1B Eye opening - No",
        "2a": "Q2A Respiration - Normal",
        "2b": "Q2B Respiration - Difficulty",
        "3a": "Q3A Complaint - Facial droop",
        "3b": "Q3B Complaint - Limb paralysis/weakness",
        "3c": "Q3C Complaint - Paresthesia",
        "3d": "Q3D Complaint - Speech difficulty",
        "3e": "Q3E Complaint - Seizures",
        "3f": "Q3F Complaint - Sudden vision loss",
        "4a": "Q4A First episode - Yes / prior without sequelae",
        "4b": "Q4B First episode - Prior with sequelae",
        "6a": "Q6A Functional status - Independent",
        "6b": "Q6B Functional status - Bedridden/dependent",
    },
    # 16: Dyspnea / Respiratory distress — to be filled in when a 02_0X_tables notebook exists
    16: {},
    # 23: Non-traumatic chest pain — to be filled in when a 02_0X_tables notebook exists
    23: {},
    # 36 & 58: Cardiac arrest / Unconsciousness — to be filled in when a 02_0X_tables notebook exists
    "cardiac_arrest": {},
}

# Manual fixes for labels that were misspelled, concatenated or in Spanish (validated 2026-10-08). Toxicos and Otherquiru pending clinical confirmation.
LABEL_OVERRIDES = {
    "hist_1_7_toxicos": "Substance Use",
    "hist_2_3_polimedicado": "Polypharmacy",
    "hist_3_7_sincope": "Syncope",
    "hist_4_4_etev": "Venous Thromboembolism",
    "hist_7_2_other_inmunodepressive": "Other Immunodepressive",
    "hist_3_4_peripheral_vasc": "Peripheral Vascular",
    "hist_3_5_chronicaortic": "Chronic Aortic",
    "hist_3_6_othercardiovascular": "Other Cardiovascular",
    "hist_4_5_respiratoryfailure": "Respiratory Failure",
    "hist_5_4_neurochronic": "Chronic Neurological",
    "hist_6_6_renalchronic": "Chronic Renal",
    "hist_7_1_otherquiru": "Other Surgical",
    "hist_7_4_otherinfec": "Other Infectious",
    "hist_7_5_othertrauma": "Other Trauma",
    "hist_8_mentalhealth": "Mental Health",
    "hist_10_11_pain_chronic": "Chronic Pain",
    "hist_4_1_copd": "COPD",
    "atc_group_nsaids": "NSAIDs",
    "atc_group_ace_inhibitors_and_arbs": "ACE Inhibitors And ARBs",
    "atc_group_drugs_for_peptic_ulcer_and_gerd": "Drugs For Peptic Ulcer And GERD",
    "atc_group_psychostimulants_adhd_and_nootropics": "Psychostimulants ADHD And Nootropics",
    "atc_group_oral_antidiabetic_drugs_excl_insulins": "Oral Antidiabetic Drugs Excl. Insulins",
    "atc_group_other_urologicals_incl_antispasmodics": "Other Urologicals Incl. Antispasmodics",
    "lr_fragility_social_telecare": "Frailty Social Telecare",
    "lr_fragility_baseline_dependency": "Frailty Baseline Dependency",
}


def resolve_demand_key(demand_code):
    """Maps a demand_code (int, list/tuple of ints, or None) to a TRIAGE_LABELS_BY_DEMAND key."""
    if isinstance(demand_code, (list, tuple)):
        return "cardiac_arrest" if set(demand_code) == {36, 58} else None
    return demand_code


def label_from_suffix(column: str, prefix: str, label_prefix: str = "") -> str:
    """Builds a display label by stripping a prefix and title-casing the remainder."""
    label = column.removeprefix(prefix).replace("_", " ").title()
    return f"{label_prefix}{label}"


def label_from_coded_suffix(column: str, prefix: str, label_prefix: str = "") -> str:
    """Like label_from_suffix, but drops purely numeric segments (e.g. hist_1_diabetes -> Diabetes)."""
    parts = column.removeprefix(prefix).split("_")
    label_parts = [part for part in parts if not part.isdigit()]
    label = " ".join(label_parts).title()
    return f"{label_prefix}{label}"


# (column prefix, label-building function) rules applied to grouped one-hot columns.
_SUFFIX_RULES = [
    ("day_week_", label_from_suffix),
    ("month_", label_from_suffix),
    ("province_", label_from_suffix),
    ("lr_", label_from_suffix),
    ("atc_group_", label_from_suffix),
    ("hist_", label_from_coded_suffix),
    ("com_", label_from_coded_suffix),
]


def get_labels_map(demand_code=None, columns=None) -> dict:
    """
    Builds a column-name -> display-label map for a given demand type.

    Combines COMMON_LABELS, the demand-specific triage question labels
    (TRIAGE_LABELS_BY_DEMAND), and auto-generated labels for grouped
    one-hot columns (day_week_*, month_*, province_*, lr_*, atc_group_*, hist_*, com_*)
    found in `columns`.
    """
    labels = dict(COMMON_LABELS)

    demand_key = resolve_demand_key(demand_code)
    labels.update(TRIAGE_LABELS_BY_DEMAND.get(demand_key, {}))

    if columns is not None:
        for prefix, label_fn in _SUFFIX_RULES:
            for col in columns:
                if col.startswith(prefix) and col not in labels:
                    labels[col] = label_fn(col, prefix)

    available_columns = None if columns is None else set(columns)
    overrides = dict(LABEL_OVERRIDES)
    overrides.update({
        key.replace("hist_", "com_", 1): label
        for key, label in LABEL_OVERRIDES.items()
        if key.startswith("hist_")
    })
    labels.update({
        key: label
        for key, label in overrides.items()
        if available_columns is None or key in available_columns
    })

    return labels


def get_display_label(column: str, labels_map: dict = None) -> str:
    """Returns the human-readable label for a column, falling back to a title-cased name."""
    if labels_map and column in labels_map:
        return labels_map[column]
    return column.replace("_", " ").title()
