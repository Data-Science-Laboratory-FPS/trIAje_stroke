import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from kbase.config import settings

wd = os.getcwd()
print(f"El directorio de trabajo actual es: {wd}")

os.chdir(settings.source_tables_path)

wd = os.getcwd()
print(f"El directorio de trabajo actual es: {wd}")

# Remove limitations in console prints
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Vectors and Dictionaries

var_order = [
    'DEMANDAPK',
    'FECHADEMANDA',
    'PATIENTID',
    'IDTARJETASANITARIA',
    'DNIAFECTADO',
    'DNI',
    'NOMBRE_APELLIDOS',
    'EDAD',
    'TIPOEDAD',
    'EDAD_llamada',
    'TIPOEDAD_llamada',
    'SEXO',
    'FNACIMIENTO',
    'IDDEMANDA',
    'IDRECURSO',
    'IDASISTENCIA',
    'IDALERTANTE',
    'COMENTARIO',
    'IDTIPODEMANDA1',
    'IDTIPODEMANDA2',
    'IDTIPODEMANDA3',
    'IDMOTIVOEXCLUSION',
    'IDTIPORECURSO',
    'IDTIPORECURSOAGR',
    'IDCODRESOLUCIONRECURSO',
    'LTR',
    'CADREC',
    'IDEMPRESA',
    'MOTIVOLITERAL',
    'JUICIOCLINICO1',
    'JUICIOCLINICO2',
    'JUICIOCLINICO3',
    'IDCODIGORESOLUCION',
    'IDCODRESOLDEMANDA',
    'IDCODRESOLASISTENCIA',
    'IDHCEPES',
    'IDRE',
    'IDTARJETACORAZON',
    'IDTARJETAAIRE',
    'ESHIPERFRECUENTADOR',
    'ESVULNERABLE',
    'PRIORIDAD',
    'IDHCDM',
    'IDESCENARIO',
    'HORAENTRADACONTACTO',
    'HORACONTESTACONTACTO',
    'HORAACTIVACION',
    'HORASALIDA',
    'HORALLEGADA',
    'HORACARGA',
    'HORAOPERATIVA',
    'HORADESTINO',
    'HORADISPONIBLE',
    'IDPROVINCIA',
    'DIRIDLOCALIDAD',
    'DIRIDLOCALIDADDEMANDA',
    'DIRIDPROVINCIAPACIENTE',
    'DIRIDLOCALIDADPACIENTE',
    'AFEC_LATITUD',
    'AFEC_LONGITUD',
    'ALER_COORDX',
    'ALER_LONGITUD',
    'ALER_OORDY',
    'DIRDESTINOIDPROVINCIARECURSO',
    'DIRDESTINOIDLOCALIDADRECURSO',
    'DIRDESTINOCALLERECURSO',
    'NODO_TIPIFICACION_ID_PK',
    'FK_STRIA_PK_IMPACTO'
]

rename_cols = {
    "DEMANDAPK": "demandpk",
    "FECHADEMANDA": "demand_date",
    "PATIENTID": "patientid",
    "IDTARJETASANITARIA": "healthcard_id",
    "DNI": "national_id",
    "NOMBRE_APELLIDOS": "full_name",
    "EDAD": "age",
    "TIPOEDAD": "age_type",
    "EDAD_llamada": "call_age",
    "TIPOEDAD_llamada": "call_age_type",
    "SEXO": "sex",
    "FNACIMIENTO": "birth_date",
    "IDDEMANDA": "demand_id",
    "IDRECURSO": "resource_id",
    "IDASISTENCIA": "assistance_id",
    "IDALERTANTE": "alert_receiver",
    "COMENTARIO": "comment",
    "IDTIPODEMANDA1": "demand_type_1",
    "IDTIPODEMANDA2": "demand_type_2",
    "IDTIPODEMANDA3": "demand_type_3",
    "IDMOTIVOEXCLUSION": "exclusion_reason_id",
    "IDTIPORECURSO": "resource_type",
    "IDTIPORECURSOAGR": "resource_group_type_id",
    "IDCODRESOLUCIONRECURSO": "resource_resolution_code_id",
    "LTR": "ltr",
    "CADREC": "cadrec",
    "IDEMPRESA": "company_id",
    "MOTIVOLITERAL": "literal_reason",
    "JUICIOCLINICO1": "icd1",
    "JUICIOCLINICO2": "icd2",
    "JUICIOCLINICO3": "icd3",
    "IDCODIGORESOLUCION": "resolution_code_id",
    "IDCODRESOLDEMANDA": "demand_resolution_code_id",
    "IDCODRESOLASISTENCIA": "assistance_resolution_code_id",
    "IDHCEPES": "hcepes_id",
    "IDRE": "idre",
    "IDTARJETACORAZON": "healthcard",
    "IDTARJETAAIRE": "aircard",
    "ESHIPERFRECUENTADOR": "is_frequent_user",
    "ESVULNERABLE": "is_vulnerable",
    "PRIORIDAD": "priority_assigned",
    "IDHCDM": "hcdm_id",
    "IDESCENARIO": "location_patient",
    "HORAENTRADACONTACTO": "contact_entry_time",
    "HORACONTESTACONTACTO": "contact_answer_time",
    "HORAACTIVACION": "activation_time",
    "HORASALIDA": "departure_time",
    "HORALLEGADA": "arrival_time",
    "HORACARGA": "load_time",
    "HORAOPERATIVA": "operative_time",
    "HORADESTINO": "destination_time",
    "HORADISPONIBLE": "available_time",
    "IDPROVINCIA": "province",
    "DIRIDLOCALIDAD": "address_locality_id",
    "DIRIDLOCALIDADDEMANDA": "demand_locality_id",
    "DIRIDPROVINCIAPACIENTE": "patient_province_id",
    "DIRIDLOCALIDADPACIENTE": "patient_locality_id",
    "AFEC_LATITUD": "incident_latitude",
    "AFEC_LONGITUD": "incident_longitude",
    "ALER_COORDX": "alert_coord_x",
    "ALER_LONGITUD": "alert_longitude",
    "ALER_OORDY": "alert_coord_y",
    "DIRDESTINOIDPROVINCIARECURSO": "resource_dest_province_id",
    "DIRDESTINOIDLOCALIDADRECURSO": "resource_dest_locality_id",
    "DIRDESTINOCALLERECURSO": "resource_dest_street",
    "NODO_TIPIFICACION_ID_PK": "node_typification_pk_id",
    "FK_STRIA_PK_IMPACTO": "impact_fk_stria_pk"
}

dtype_cols = {
    "demandpk": "Int32",
    "demand_date": "datetime64[ns]",
    "patientid": "object",
    "healthcard_id": "object",
    "national_id": "object",
    "full_name": "object",
    "age": "Int16",
    "age_0_14": "Int8",
    "age_15_24": "Int8",
    "age_25_44": "Int8",
    "age_45_59": "Int8",
    "age_60_74": "Int8",
    "age_75_plus": "Int8",
    "age_type": "category",
    "call_age": "float64",
    "call_age_type": "category",
    "sex": "category",
    "birth_date": "datetime64[ns]",
    "demand_id": "Int32",
    "resource_id": "object",
    "assistance_id": "object",
    "alert_receiver": "category",
    "alert_receiver_112": "int8",
    "alert_receiver_user": "int8",
    "alert_receiver_pol_fg": "int8",
    "alert_receiver_hs": "int8",
    "alert_receiver_tele": "int8",
    "alert_receiver_others": "int8",
    "comment": "object",
    "demand_type_1": "Int8",
    "demand_type_2": "float64",
    "demand_type_3": "object",
    "exclusion_reason_id": "category",
    "resource_type": "object",
    "resource_group_type_id": "object",
    "resource_resolution_code_id": "object",
    "ltr": "object",
    "cadrec": "object",
    "company_id": "object",
    "literal_reason": "object",
    "icd1": "object",
    "icd2": "object",
    "icd3": "object",
    "icd_emerg_combined": "object",
    "has_icd_emerg": "int8",
    "resolution_code_id": "object",
    "demand_resolution_code_id": "category",
    "assistance_resolution_code_id": "category",
    "hcepes_id": "object",
    "idre": "object",
    "healthcard": "category",
    "aircard": "category",
    "is_frequent_user": "category",
    "is_vulnerable": "category",
    "priority_assigned": "Int8",
    "p1_assigned": "Int8",
    "p1_real_emerg": "Int8",
    "p1_real_bps": "Int8",
    "hcdm_id": "object",
    "location_patient": "category",
    "location_patient_home": "int8",
    "location_patient_public_road": "int8",
    "location_patient_other": "int8",
    "day_week": "category",
    "day_week_monday": "int8",
    "day_week_tuesday": "int8",
    "day_week_wednesday": "int8",
    "day_week_thursday": "int8",
    "day_week_friday": "int8",
    "day_week_saturday": "int8",
    "day_week_sunday": "int8",
    "time_of_day": "category",
    "time_of_day_early_morning": "int8",
    "time_of_day_morning": "int8",
    "time_of_day_afternoon": "int8",
    "time_of_day_night": "int8",
    "month": "category",
    "month_january": "int8",
    "month_february": "int8",
    "month_march": "int8",
    "month_april": "int8",
    "month_may": "int8",
    "month_june": "int8",
    "month_july": "int8",
    "month_august": "int8",
    "month_september": "int8",
    "month_october": "int8",
    "month_november": "int8",
    "month_december": "int8",
    "season": "category",
    "season_spring": "int8",
    "season_summer": "int8",
    "season_autumn": "int8",
    "season_winter": "int8",
    "year": "category",
    "contact_entry_time": "datetime64[ns]",
    "contact_answer_time": "datetime64[ns]",
    "activation_time": "datetime64[ns]",
    "departure_time": "datetime64[ns]",
    "arrival_time": "datetime64[ns]",
    "load_time": "datetime64[ns]",
    "operative_time": "datetime64[ns]",
    "destination_time": "datetime64[ns]",
    "available_time": "datetime64[ns]",
    "province": "category",
    "province_almeria": "int8",
    "province_cadiz": "int8",
    "province_cordoba": "int8",
    "province_granada": "int8",
    "province_huelva": "int8",
    "province_jaen": "int8",
    "province_malaga": "int8",
    "province_sevilla": "int8",
    "address_locality_id": "float64",
    "demand_locality_id": "float64",
    "patient_province_id": "float64",
    "patient_locality_id": "float64",
    "incident_latitude": "float64",
    "incident_longitude": "float64",
    "alert_coord_x": "float64",
    "alert_longitude": "float64",
    "alert_coord_y": "float64",
    "resource_dest_province_id": "category",
    "resource_dest_locality_id": "float64",
    "resource_dest_street": "object",
    "node_typification_pk_id": "category",
    "impact_fk_stria_pk": "category", 
    "triage": "category",
    "q1": "category",
    "q2": "category",
    "q3": "category",
    "q4": "category",
    "q5": "category",
    "q6": "category",
    "q7": "category", 
    "q1_a": "int8",
    "q1_b": "int8",
    "q1_c": "int8",
    "q1_d": "int8",
    "q2_a": "int8",
    "q2_b": "int8",
    "q2_c": "int8",
    "q2_d": "int8",
    "q2_e": "int8",
    "q2_f": "int8",
    "q2_g": "int8",
    "q2_h": "int8",
    "q3_a": "int8",
    "q3_b": "int8",
    "q3_c": "int8",
    "q3_d": "int8",
    "q3_e": "int8",
    "q3_f": "int8",
    "q4_a": "int8",
    "q4_b": "int8",
    "q4_c": "int8",
    "q4_d": "int8",
    "q4_e": "int8",
    "q4_f": "int8",
    "q4_g": "int8",
    "q4_h": "int8",
    "q4_i": "int8",
    "q4_j": "int8",
    "q4_k": "int8",
    "q4_l": "int8",
    "q4_m": "int8",
    "q5_a": "int8",
    "q5_b": "int8",
    "q5_c": "int8",
    "q5_d": "int8",
    "q5_e": "int8",
    "q5_f": "int8",
    "q6_a": "int8",
    "q6_b": "int8",
    "q6_c": "int8",
    "q6_d": "int8",
    "q7_a": "int8",
    "q7_b": "int8"
}

# Functions
## Function to extracts information on nº of id rows from a df
def summarize_numbers(
    df: pd.DataFrame,
    key_col: str,
    id_cols: list
):
    """
    Summarize key identifier variables and their completeness.
    
    id_cols should be a list of exactly 3 columns:
        [id1_col, id2_col, id3_col]
    """
    # --- Unpack ID columns ---
    if len(id_cols) != 3:
        raise ValueError("id_cols must contain exactly 3 column names.")
    
    id1_col, id2_col, id3_col = id_cols

    # --- Basic counts ---
    n_rows = len(df)
    n_key_unique = df[key_col].nunique()

    # Optional PATIENTID column
    patient_id_col = None
    if "PATIENTID" in df.columns:
        patient_id_col = "PATIENTID"
    elif "patientid" in df.columns:
        patient_id_col = "patientid"
    if patient_id_col:
        n_unique_patient = df[patient_id_col].nunique()
        patient_counts = df[patient_id_col].value_counts()
        n_patient_once = (patient_counts == 1).sum()
        n_patient_multiple = (patient_counts > 1).sum()
        # Rows corresponding to patients appearing once or more than once
        n_rows_patient_once = (patient_counts == 1).sum()       # same as n_patient_once (1 row each)
        n_rows_patient_multiple = df[patient_id_col].isin(
            patient_counts[patient_counts > 1].index
        ).sum()
    else:
        n_unique_patient = None

    # Counts for ID variables
    n_id1_notna = df[id1_col].notna().sum()
    n_id1_unique = df[id1_col].nunique()
    n_id1_nan = df[id1_col].isna().sum()

    n_id2_notna = df[id2_col].notna().sum()
    n_id2_unique = df[id2_col].nunique()
    n_id2_nan = df[id2_col].isna().sum()

    n_id3_notna = df[id3_col].notna().sum()
    n_id3_unique = df[id3_col].nunique()
    n_id3_nan = df[id3_col].isna().sum()

    # Rows where BOTH id1 and id2 are missing
    n_id1_and_id2_nan = (
        df[id1_col].isna() &
        df[id2_col].isna()
    ).sum()

    # Rows where ALL THREE IDs are missing
    n_all_three_nan = (
        df[id1_col].isna() &
        df[id2_col].isna() &
        df[id3_col].isna()
    ).sum()

    # Unique pairs with key_col
    n_unique_pairs_id1 = df[[key_col, id1_col]].dropna(subset=[id1_col]).nunique().min()
    n_unique_pairs_id2 = df[[key_col, id2_col]].dropna(subset=[id2_col]).nunique().min()
    n_unique_pairs_id3 = df[[key_col, id3_col]].dropna(subset=[id3_col]).nunique().min()

    # Rows with id1 but not id2, and vice versa
    n_id1_not_id2 = (df[id1_col].notna() & df[id2_col].isna()).sum()
    n_id2_not_id1 = (df[id2_col].notna() & df[id1_col].isna()).sum()

    # --- Helper ---
    def pct(n): return n / n_rows * 100

    # --- PRINT BLOCK ---
    print("\n--- Summary of key and patient identifiers ---")
    print(f"Total rows: {n_rows:,} (100.00%)")
    print(f"Unique {key_col}: {n_key_unique:,} ({pct(n_key_unique):.2f}%)")
    if n_unique_patient is not None:
        print(f"Unique patient IDs ({patient_id_col}): {n_unique_patient:,} ({pct(n_unique_patient):.2f}%)")
        print(f"  patients appearing exactly once:      {n_patient_once:,} ({pct(n_patient_once):.2f}%) — rows: {n_rows_patient_once:,} ({pct(n_rows_patient_once):.2f}%)")
        print(f"  patients appearing more than once:    {n_patient_multiple:,} ({pct(n_patient_multiple):.2f}%) — rows: {n_rows_patient_multiple:,} ({pct(n_rows_patient_multiple):.2f}%)")

    print(f"\n{id1_col}:")
    print(f"  notna:  {n_id1_notna:,} ({pct(n_id1_notna):.2f}%)")
    print(f"  unique: {n_id1_unique:,} ({pct(n_id1_unique):.2f}%)")
    print(f"  NaN:    {n_id1_nan:,} ({pct(n_id1_nan):.2f}%)")

    print(f"\n{id2_col}:")
    print(f"  notna:  {n_id2_notna:,} ({pct(n_id2_notna):.2f}%)")
    print(f"  unique: {n_id2_unique:,} ({pct(n_id2_unique):.2f}%)")
    print(f"  NaN:    {n_id2_nan:,} ({pct(n_id2_nan):.2f}%)")

    print(f"\n{id3_col}:")
    print(f"  notna:  {n_id3_notna:,} ({pct(n_id3_notna):.2f}%)")
    print(f"  unique: {n_id3_unique:,} ({pct(n_id3_unique):.2f}%)")
    print(f"  NaN:    {n_id3_nan:,} ({pct(n_id3_nan):.2f}%)")

    print(f"\nRows with {id1_col} AND {id2_col} missing: {n_id1_and_id2_nan:,} ({pct(n_id1_and_id2_nan):.2f}%)")
    print(f"Rows with ALL THREE IDs missing: {n_all_three_nan:,} ({pct(n_all_three_nan):.2f}%)")

    print("\n--- Unique pairs with key_col ---")
    print(f"({key_col}, {id1_col}) → {n_unique_pairs_id1:,} unique pairs ({pct(n_unique_pairs_id1):.2f}%)")
    print(f"({key_col}, {id2_col}) → {n_unique_pairs_id2:,} unique pairs ({pct(n_unique_pairs_id2):.2f}%)")
    print(f"({key_col}, {id3_col}) → {n_unique_pairs_id3:,} unique pairs ({pct(n_unique_pairs_id3):.2f}%)")

    print(f"\n--- {id1_col} / {id2_col} cross-presence ---")
    print(f"Rows with {id1_col} but NOT {id2_col}: {n_id1_not_id2:,} ({pct(n_id1_not_id2):.2f}%)")
    print(f"Rows with {id2_col} but NOT {id1_col}: {n_id2_not_id1:,} ({pct(n_id2_not_id1):.2f}%)")

## Reorders the columns of a DataFrame based on a specified column order.
def reorder_dataframe(df: pd.DataFrame, column_order: list = var_order) -> pd.DataFrame:
    """
    Reorder the columns of a DataFrame based on a specified ordered list.
    Only columns present in `df` and in `column_order` are kept.
    Prints which columns were removed during the process.
    """

    # Columns that are both in the desired order and in the DataFrame
    existing_columns = [col for col in column_order if col in df.columns]

    # Columns that were removed (present in df but not in the reindexed version)
    removed_columns = [col for col in df.columns if col not in existing_columns]

    # Print removed columns
    if removed_columns:
        print("Columns removed:", removed_columns)
    else:
        print("No columns were removed.")

    # Return DataFrame with only the ordered existing columns
    return df.reindex(columns=existing_columns)

## Renames DataFrame columns based on a predefined translation dictionary.
def rename_columns(df: pd.DataFrame,
                   rename_dict: dict = rename_cols) -> pd.DataFrame:

    existing = {k: v for k, v in rename_dict.items() if k in df.columns}
    return df.rename(columns=existing)

## Reduce the size of a dataframe depending on a flag.
def df_pipeline_test(df: pd.DataFrame) -> pd.DataFrame:
    if settings.test_pipeline == "all":
        return df
    
    elif settings.test_pipeline == "test":
        return df.iloc[:10000].copy()
    
    else:
        raise ValueError('settings.test_pipeline must be either "all" or "test".')

# Function for converting data types and reordering columns
def transform_column_dtypes(df: pd.DataFrame, dtype_dict=dtype_cols):
    """
    Cast dataframe columns to specified types, dynamically including 
    prefix-based rules (lr_, hist_) and reordering.
    """
    # 1. Create a local copy of the dictionary to avoid modifying the original global one
    working_dtype_dict = dtype_dict.copy()

    # 2. Dynamically add columns starting with 'lr_' or 'hist_' as 'int8'
    for col in df.columns:
        if col.startswith(('lr_', 'has_hist', 'hist_', 'has_med', 'atc_', 'has_com', 'com_')):
            working_dtype_dict[col] = "int8"

    # 3. Keep only dtype rules that apply to existing columns
    applicable = {col: dtype for col, dtype in working_dtype_dict.items() if col in df.columns}

    # 4. Apply dtype conversions
    # Note: Using astype on the dictionary is efficient
    df = df.astype(applicable, errors="ignore")

    # 5. Reorder columns
    # We follow the dictionary order first, then any extra columns
    ordered_cols = [col for col in working_dtype_dict.keys() if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]

    return df[ordered_cols + remaining_cols]

# Function to filter dataset rows where a target ICD code matches any ICD column
def filter_rows_by_icd(df, icd_value, icd_columns):
    """
    Filter dataset rows where a target ICD code matches any ICD column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing ICD columns.
    icd_value : str
        ICD code to search for (exact match).
    icd_columns : list
        List of ICD column names to check (e.g., ["icd1", "icd2", "icd3"]).

    Returns
    -------
    pandas.DataFrame
        Subset of df where icd_value is found in any ICD column.
    """
    
    # Ensure ICDs are compared as strings
    df_icd = df.copy()
    df_icd[icd_columns] = df_icd[icd_columns].astype("string")

    # Build boolean mask for any column matching the ICD value
    mask = df_icd[icd_columns].apply(lambda col: col == icd_value).any(axis=1)

    # Return filtered rows
    return df_icd[mask]

import os
import pyarrow.parquet as pq

# Function to merge main df with embedding table (newly or already created)
def merge_text_embeddings(run_mode, df_embeddings, embedding_path, df_main):
    """
    Handles loading or utilizing text embeddings and merges them with the main DataFrame.
    
    Args:
        run_mode (int): If 0, loads embeddings from disk. Otherwise, uses the provided DataFrame.
        df_embeddings (pd.DataFrame): The newly created embedding DataFrame (if any).
        embedding_path (str): Filename/path for the saved embedding table.
        df_main (pd.DataFrame): The primary DataFrame (e.g., triage data).
        source_path (str): The directory path where embedding files are stored.
        
    Returns:
        pd.DataFrame: The main DataFrame merged with embedding features.
    """
    
    # --- Handling Embedding Table (New or Saved) ---
    if run_mode == 0:
        # Scenario: Embeddings already exist, just load them
        print("--- Loading previously saved embedding table ---")
        full_path = os.path.join(settings.source_tables_path, embedding_path)
        df_embeddings = pq.read_table(full_path).to_pandas()
    else:
        # Scenario: df_embeddings was just created by the embedding function
        print("--- Using newly created embeddings ---")

    # --- Unified Merge Logic ---
    # We only perform the join if df_embeddings is available (either loaded or created)
    if df_embeddings is not None:
        print(f"--- Merging embeddings ({len(df_embeddings)} rows) "
              f"with main table ({len(df_main)} rows) ---")
        
        # We drop 'literal_reason' from the embedding table to avoid duplication
        # We use 'on' because both tables must share the 'demandpk' column
        df_main = df_main.merge(
            df_embeddings.drop(columns=['literal_reason'], errors='ignore'), 
            on='demandpk', 
            how='left'
        )
        
        # Optional: ensure memory efficiency by deleting the temporary dataframe
        del df_embeddings
        print("--- Merge completed successfully ---")
    else:
        print("Error: No embedding data found to merge.")

    print(f"\n" + "="*40)
    print(f"Number of features for modeling: {len(df_main.columns)}")
    print("="*40 + "\n")
        
    return df_main

# Maps Spanish-abbreviated lr_* column names (as produced by the NLP one-hot pipeline)
# to their English equivalents used throughout the modeling pipeline.
lr_translation_dict = {
    "lr_sospecha_acv":    "lr_suspected_stroke",
    "lr_foc_habla":       "lr_focal_speech_deficit",
    "lr_foc_facial":      "lr_focal_facial_weakness",
    "lr_foc_motor_ms":    "lr_focal_arm_weakness",
    "lr_foc_lateral":     "lr_focal_laterality_mentioned",
    "lr_foc_motor_mi":    "lr_focal_leg_weakness",
    "lr_foc_sensiti":     "lr_focal_sensory_deficit",
    "lr_foc_visual":      "lr_focal_visual_deficit",
    "lr_confu_aguda":     "lr_acute_confusion",
    "lr_alt_consci":      "lr_altered_consciousness",
    "lr_tmp_hiperag":     "lr_onset_hyperacute",
    "lr_tmp_despert":     "lr_onset_on_waking",
    "lr_tmp_evolucionado":"lr_onset_subacute",
    "lr_cefalea":         "lr_headache",
    "lr_convul":          "lr_seizure_or_convulsion",
    "lr_sint_mareo":      "lr_symptom_dizziness_syncope",
    "lr_sint_digest":     "lr_symptom_digestive",
    "lr_sint_infecc":     "lr_symptom_infectious",
    "lr_sint_temblor":    "lr_symptom_tremor_rigidity",
    "lr_sint_disfagia":   "lr_symptom_dysphagia",
    "lr_frag_social":     "lr_fragility_social_telecare",
    "lr_frag_basal":      "lr_fragility_baseline_dependency",
    "lr_ant_acv_prev":    "lr_history_prior_stroke",
    "lr_riesgo_cv":       "lr_history_cardiovascular_risk",
    "lr_src_profes":      "lr_caller_is_professional",
    "lr_missing_dat":     "lr_no_clinical_data",
}

# Merges related ATC sub-groups into a single column via OR logic (max of dummies).
# Source columns are dropped after merging.
ATC_GROUPING_CONFIG = {
    'atc_group_IECA_ARAII': [
        'atc_group_ANTAGONISTAS_ANGIOTENSINA_II',
        'atc_group_INHIBIDORES_ENZIMA_CONVERTASA_ANGIOTENSINA_IECA',
        'atc_group_ANTAGONISTAS_ANGIOTENSINA_II_EN_ASOCIACION',
    ],
    'atc_group_DIURETICOS': [
        'atc_group_DIURETICOS_DE_ALTO_TECHO',
        'atc_group_DIURETICOS_DE_BAJO_TECHO_TIAZIDAS',
        'atc_group_DIURETICOS_DE_BAJO_TECHO_EXCLUIDAS_TIAZIDAS',
    ],
    'atc_group_ANTIBACTERIANOS': [
        'atc_group_ANTIBACTERIANOS_DERIVADOS_DE_LA_QUINOLONA_SISTEM',
        'atc_group_OTROS_ANTIBACTERIANOS_ANTIINFECCIOSOS_SISTEMICOS',
        'atc_group_ANTIBACTERIANOS_BETALACTAMICOS_PENICILINAS',
    ],
    'atc_group_OBSTRUCCION_VIAS_RESPIRATORIAS_INHALADOS': [
        'atc_group_OTROS_PARA_OBSTRUCCION_DE_VIAS_RESPIRATORIAS_INHALADOS',
        'atc_group_ANTICOLINERGICOS_ANTIASMATICOS',
    ],
    'atc_group_MODIFICADORES_DE_LOS_LIPIDOS_SOLOS': [
        'atc_group_MODIFICADORES_DE_LOS_LIPIDOS_SOLOS',
        'atc_group_MODIFICADORES_DE_LOS_LIPIDOS_EN_ASOCIACION',
    ],
    'atc_group_ANSIOLITICOS': [
        'atc_group_ANSIOLITICOS',
        'atc_group_HIPNOTICOS_Y_SEDANTES',
        'atc_group_N05C_Hipn_ticos_y_sedantes_N05CM_Otros_hipn_ticos_y_sedantes',
    ],
    'atc_group_BETABLOQUEANTES_SELECTIVOS_SOLOS': [
        'atc_group_BETABLOQUEANTES_SELECTIVOS_SOLOS',
        'atc_group_BETABLOQUEANTES_SOLOS',
    ],
    'atc_group_MEDICAMENTOS_PARA_HIPERTROFIA_PROSTATICA_BENIGNA': [
        'atc_group_MEDICAMENTOS_PARA_HIPERTROFIA_PROSTATICA_BENIGNA',
        'atc_group_G04C_F_rmacos_usados_en_hipertrofia_prost_tica_benigna_otros_f_rmacos_urol_gicos',
    ],
    'atc_group_MEDICAMENTOS_QUE_AFECTAN_A_ESTRUCTURA_OSEA_Y_MINERALIZACION': [
        'atc_group_MEDICAMENTOS_QUE_AFECTAN_A_ESTRUCTURA_OSEA_Y_MINERALIZACION',
        'atc_group_CALCIO',
    ],
    'atc_group_HIERRO': [
        'atc_group_B03A_Preparados_con_hierro_B03AA_Hierro_bivalente_preparados_orales',
        'atc_group_B03A_Preparados_con_hierro_B03AB_Hierro_trivalente_preparados_orales',
    ],
    'atc_group_PROPULSIVOS': [
        'atc_group_PROPULSIVOS',
        'atc_group_A03F_Propulsivos_A03FA_Propulsivos_benzamidas_con_acci_n_procin_tica_antiem_tica',
    ],
    'atc_group_BLOQ_CANALES_CALCIO_SELECTIVOS_EFECTO_VASCULAR': [
        'atc_group_BLOQ_CANALES_CALCIO_SELECTIVOS_EFECTO_VASCULAR',
        'atc_group_BLOQ_CANALES_CALCIO_SELECTIVOS_EFECTO_CARDIACO',
    ],
    'atc_group_NITRATOS_ORGANICOS_CARDIOTERAPIA': [
        'atc_group_NITRATOS_ORGANICOS_CARDIOTERAPIA',
        'atc_group_VASODILATADORES_USADOS_EN_CARDIOTERAPIA',
        'atc_group_OTROS_PREPARADOS_CARDIACOS',
    ],
    'atc_group_DOPA_Y_DERIVADOS_ANTIPARKINSONIANOS': [
        'atc_group_DOPA_Y_DERIVADOS_ANTIPARKINSONIANOS',
        'atc_group_DOPAMINERGICOS_ANTIPARKINSONIANOS',
    ],
}

# Maps post-grouping column names (lowercase) to their final English names.
# Keys are the real dataset column names lowercased. Columns not listed here
# are dropped by the final filter in analyze_atc_columns.
ATC_TRANSLATION_DICT = {
    # Grouped columns
    'atc_group_ieca_araii':
        'atc_group_ace_inhibitors_and_arbs',
    'atc_group_diureticos':
        'atc_group_diuretics',
    'atc_group_antibacterianos':
        'atc_group_antibacterials',
    'atc_group_obstruccion_vias_respiratorias_inhalados':
        'atc_group_inhalants_for_obstructive_airway_diseases',
    'atc_group_modificadores_de_los_lipidos_solos':
        'atc_group_lipid_modifying_agents',
    'atc_group_ansioliticos':
        'atc_group_psycholeptics',
    'atc_group_betabloqueantes_selectivos_solos':
        'atc_group_beta_blocking_agents',
    'atc_group_medicamentos_para_hipertrofia_prostatica_benigna':
        'atc_group_benign_prostatic_hypertrophy_drugs',
    'atc_group_medicamentos_que_afectan_a_estructura_osea_y_mineralizacion':
        'atc_group_calcium_and_bone_structure_agents',
    'atc_group_hierro':
        'atc_group_iron_preparations',
    'atc_group_propulsivos':
        'atc_group_propulsives',
    'atc_group_bloq_canales_calcio_selectivos_efecto_vascular':
        'atc_group_calcium_channel_blockers',
    'atc_group_nitratos_organicos_cardioterapia':
        'atc_group_other_cardiac_therapy',
    'atc_group_dopa_y_derivados_antiparkinsonianos':
        'atc_group_anti_parkinson_drugs',

    # Individual columns
    'atc_group_medicamentos_para_ulcera_peptica_y_reflujo':
        'atc_group_drugs_for_peptic_ulcer_and_gerd',
    'atc_group_otros_analgesicos_y_antipireticos':
        'atc_group_other_analgesics_and_antipyretics',
    'atc_group_producto_sanitario_absorbentes':
        'atc_group_absorbent_sanitary_products',
    'atc_group_inhibidores_de_la_agregacion_plaquetaria':
        'atc_group_platelet_aggregation_inhibitors',
    'atc_group_antidiabeticos_orales_excl_insulinas':
        'atc_group_oral_antidiabetic_drugs_excl_insulins',
    'atc_group_antitromboticos':
        'atc_group_antithrombotic_agents',
    'atc_group_antidepresivos':
        'atc_group_antidepressants',
    'atc_group_analgesicos_opiaceos':
        'atc_group_opioid_analgesics',
    'atc_group_antipsicoticos':
        'atc_group_antipsychotics',
    'atc_group_vitaminas_a_y_d':
        'atc_group_vitamins_a_and_d',
    'atc_group_insulinas_y_analogos':
        'atc_group_insulins_and_analogues',
    'atc_group_adrenergicos_inhalados_antiasmaticos':
        'atc_group_inhaled_adrenergics',
    'atc_group_preparados_tirodeos':
        'atc_group_thyroid_preparations',
    'atc_group_antiepilepticos':
        'atc_group_antiepileptics',
    'atc_group_preparados_contra_la_gota':
        'atc_group_antigout_preparations',
    'atc_group_corticosteroides_de_uso_sistemico_solos':
        'atc_group_systemic_corticosteroids_plain',
    'atc_group_antihistaminicos_de_uso_sistemico':
        'atc_group_antihistamines_for_systemic_use',
    'atc_group_preparados_antiglaucoma_y_mioticos':
        'atc_group_antiglaucoma_preparations_and_miotics',
    'atc_group_medicamentos_contra_la_demencia':
        'atc_group_anti_dementia_drugs',
    'atc_group_antiinflamatorios_y_antireumaticos_no_esteroideos':
        'atc_group_nsaids',
    'atc_group_medicamentos_contra_el_vertigo':
        'atc_group_antivertigo_preparations',
    'atc_group_diureticos_ahorradores_de_potasio':
        'atc_group_potassium_sparing_diuretics',
    'atc_group_vasodilatadores_perifericos':
        'atc_group_peripheral_vasodilators',
    'atc_group_vitamina_b12_y_derivados':
        'atc_group_vitamin_b12_and_folic_acid',
    'atc_group_antiadrenergicos_de_accion_periferica_antihipert':
        'atc_group_peripheral_antiadrenergics',
    'atc_group_otros_preparados_urologicos_incl_antiespasmodicos':
        'atc_group_other_urologicals_incl_antispasmodics',
    'atc_group_laxantes':
        'atc_group_laxatives',
    'atc_group_nutrici_n_cl_nica_diet_ticos_sueros':
        'atc_group_clinical_nutrition_and_electrolytes',
    'atc_group_glicosidos_cardiacos':
        'atc_group_cardiac_glycosides',
    'atc_group_expectorantes_excluid_asociac_con_antitusigenos':
        'atc_group_expectorants',
    'atc_group_psicoestimulantes_medic_para_adhd_y_nootropicos':
        'atc_group_psychostimulants_adhd_and_nootropics',
    'atc_group_antiarritmicos':
        'atc_group_antiarrhythmics',
}


def analyze_atc_columns(
    df: pd.DataFrame,
    threshold: float = 1.0,
    show_counts: bool = False,
    show_histogram: bool = False,
    export: bool = False,
    export_path: str = None,
    drop_columns: bool = False,
    group_and_translate: bool = True,
) -> pd.DataFrame:
    """
    Analyzes ATC medication columns (those starting with 'atc_').

    Parameters
    ----------
    df : pd.DataFrame
    threshold : float
        Minimum percentage of non-zero rows to keep a column (e.g. 1.0 = 1%).
    show_counts : bool
        Print the column count summary after threshold filtering and, if
        group_and_translate=True, the final column list with n and completeness %
        after grouping and translation.
    show_histogram : bool
        Plot the completeness histogram before threshold filtering.
    export : bool
        Save the cleaned dataframe to parquet. Requires export_path.
    export_path : str, optional
        Full path for the output parquet file. Used only when export=True.
    drop_columns : bool
        If True, returns the cleaned dataframe. If False, returns None.
    group_and_translate : bool
        If True (default), merges related ATC sub-groups via OR logic (max of
        dummies), lowercases all column headers, applies ATC_TRANSLATION_DICT,
        and drops any remaining untranslated ATC columns.

    Returns
    -------
    pd.DataFrame or None
        Cleaned dataframe if drop_columns=True, otherwise None.
    """
    threshold_dec = threshold / 100
    atc_cols = [c for c in df.columns if c.startswith("atc_")]
    n_total = len(df)

    # 1. Calculate completeness on the original ATC columns
    completeness = (
        df[atc_cols]
        .astype(bool)
        .sum()
        .rename("n_nonzero")
        .to_frame()
    )
    completeness["pct"] = completeness["n_nonzero"] / n_total
    completeness = completeness.sort_values("pct", ascending=False)

    above     = completeness[completeness["pct"] >= threshold_dec]
    below     = completeness[completeness["pct"] < threshold_dec]
    all_zero  = completeness[completeness["n_nonzero"] == 0]

    if show_counts:
        print(f"Total ATC columns:                        {len(atc_cols)}")
        print(f"All-zero columns (drop):                  {len(all_zero)}")
        print(f"Below threshold ({threshold:.2f}%, drop):  {len(below)}")
        print(f"Above threshold ({threshold:.2f}%, keep):  {len(above)}")
        print()
        print(f"Columns above {threshold:.2f}% threshold:")
        print(
            above
            .rename(columns={"n_nonzero": "n", "pct": "completeness"})
            .assign(completeness=lambda x: x["completeness"].map("{:.2%}".format))
            .to_string()
        )

    if show_histogram:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(completeness["pct"] * 100, bins=50, edgecolor="white", color="steelblue")
        ax.axvline(threshold, color="crimson", linestyle="--",
                   label=f"Threshold {threshold:.2f}%")
        ax.set_xlabel("Completeness (% rows with non-zero value)")
        ax.set_ylabel("Number of ATC columns")
        ax.set_title("Distribution of completeness across ATC medication columns")
        ax.legend()
        plt.tight_layout()
        plt.show()

    # 2. Drop columns below threshold
    cols_to_drop = below.index.tolist()
    df_clean = df.drop(columns=cols_to_drop)

    if show_counts:
        print(f"\nATC columns eliminated (below threshold {threshold:.2f}%): {len(cols_to_drop)}")
        print(f"Remaining columns after dropping:                          {df_clean.shape[1]}")

    # 3. Grouping and translation
    if group_and_translate:
        # A. Merge sub-groups via OR logic (max of dummies).
        #    When target name matches a source name, compute first then drop others
        #    to avoid dropping the column before it is assigned.
        for target_group, source_columns in ATC_GROUPING_CONFIG.items():
            available_cols = [c for c in source_columns if c in df_clean.columns]
            if available_cols:
                df_clean[target_group] = df_clean[available_cols].max(axis=1)
                cols_to_remove = [c for c in available_cols if c != target_group]
                df_clean.drop(columns=cols_to_remove, inplace=True)

        # B. Standardize all column names to lowercase
        df_clean.columns = [col.lower() for col in df_clean.columns]

        # C. Rename to final English names
        df_clean.rename(columns=ATC_TRANSLATION_DICT, inplace=True)

        # D. Drop any remaining ATC columns not covered by the translation dictionary
        english_atc_targets = set(ATC_TRANSLATION_DICT.values())
        final_cols_to_keep = [
            c for c in df_clean.columns
            if not c.startswith('atc_') or c in english_atc_targets
        ]
        df_clean = df_clean[final_cols_to_keep]

        if show_counts:
            final_atc_cols = sorted(c for c in df_clean.columns if c.startswith('atc_'))
            final_completeness = (
                df_clean[final_atc_cols]
                .astype(bool)
                .sum()
                .rename("n")
                .to_frame()
            )
            final_completeness["completeness"] = (
                final_completeness["n"] / n_total
            ).map("{:.2%}".format)
            final_completeness = final_completeness.sort_values("n", ascending=False)

            print(f"\nATC columns after grouping and translation: {len(final_atc_cols)}")
            print(final_completeness.to_string())

    if export:
        if export_path is None:
            raise ValueError("export=True requires export_path to be provided.")
        df_clean.to_parquet(export_path, index=False)

    return df_clean if drop_columns else None

## Function to analyze missing values
def analyze_missing_values(df):
    """
    Analyze missingness in a DataFrame, including:
    - NaN percentages
    - Blank-string percentages
    - Combined summary
    - Matplotlib plots of missingness
    
    Only the combined summary is printed.
    """

    print("\n" + "="*70)
    print("MISSING VALUE ANALYSIS")
    print("="*70 + "\n")

    # ==============================================================
    # 1. NaN percentage table
    # ==============================================================
    nan_percent = df.isna().mean() * 100
    missing_table = nan_percent.reset_index()
    missing_table.columns = ["variable", "percent_nan"]
    missing_table = missing_table.sort_values(by="percent_nan", ascending=False)

    # ==============================================================
    # 2. Blank-string percentage table
    # ==============================================================
    blank_percent = (
        df.apply(lambda col: col.map(lambda x: isinstance(x, str) and x.strip() == ""))
        .mean() * 100
    )

    blank_table = blank_percent.reset_index()
    blank_table.columns = ["variable", "percent_blank"]
    blank_table = blank_table.sort_values(by="percent_blank", ascending=False)

    # ==============================================================
    # 3. Combined table (only this is printed)
    # ==============================================================
    combined_table = missing_table.merge(blank_table, on="variable", how="outer")
    combined_table["percent_total_missing"] = (
        combined_table["percent_nan"].fillna(0)
        + combined_table["percent_blank"].fillna(0)
    )
    combined_table = combined_table.sort_values(
        by="percent_total_missing", ascending=False
    )

    print("COMBINED SUMMARY: NaN + Blank")
    print("-"*70)
    print(combined_table.to_string(index=False))
    print("\n" + "="*70 + "\n")

    # ==============================================================
    # 4. Plot: NaN missingness
    # ==============================================================
    plt.figure(figsize=(12, 6))
    plt.bar(missing_table["variable"], missing_table["percent_nan"])
    plt.xticks(rotation=90)
    plt.ylabel("Percentage of NaN values")
    plt.title("Missingness (NaN) by Variable")
    plt.tight_layout()
    plt.show()

    # ==============================================================
    # 5. Plot: Blank-string missingness
    # ==============================================================
    plt.figure(figsize=(12, 6))
    plt.bar(blank_table["variable"], blank_table["percent_blank"])
    plt.xticks(rotation=90)
    plt.ylabel("Percentage of blank-string values")
    plt.title("Missingness (Blank Strings) by Variable")
    plt.tight_layout()
    plt.show()









