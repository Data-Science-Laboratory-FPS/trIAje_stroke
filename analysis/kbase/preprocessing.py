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
    "PRIORIDAD": "p1_predicted",
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
    "age_type": "category",
    "call_age": "float64",
    "call_age_type": "category",
    "sex": "category",
    "birth_date": "datetime64[ns]",
    "demand_id": "Int32",
    "resource_id": "object",
    "assistance_id": "object",
    "alert_receiver": "category",
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
    "resolution_code_id": "object",
    "demand_resolution_code_id": "category",
    "assistance_resolution_code_id": "category",
    "hcepes_id": "object",
    "idre": "object",
    "healthcard": "category",
    "aircard": "category",
    "is_frequent_user": "category",
    "is_vulnerable": "category",
    "p1_predicted": "category",
    "p1_real_emerg": "Int8",
    "p1_real_bps": "Int8",
    "hcdm_id": "object",
    "location_patient": "category",
    "day_week": "category",
    "time_of_day": "category",
    "month": "category",
    "season": "category",
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
    else:
        n_unique_patient = None

    # Counts for ID variables
    n_id1_unique = df[id1_col].nunique()
    n_id1_nan = df[id1_col].isna().sum()

    n_id2_unique = df[id2_col].nunique()
    n_id2_nan = df[id2_col].isna().sum()

    n_id3_unique = df[id3_col].nunique()
    n_id3_nan = df[id3_col].isna().sum()

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

    # --- PRINT BLOCK ---
    print("\n--- Summary of key and patient identifiers ---")
    print(f"Total rows: {n_rows:,}")
    print(f"Unique {key_col}: {n_key_unique:,}")

    if n_unique_patient is not None:
        print(f"Unique patient IDs ({patient_id_col}): {n_unique_patient:,}")

    # print("\n")
    print(f"\n{id1_col}: {n_id1_unique:,} unique — NaN: {n_id1_nan:,}")
    print(f"{id2_col}: {n_id2_unique:,} unique — NaN: {n_id2_nan:,}")
    print(f"{id3_col}: {n_id3_unique:,} unique — NaN: {n_id3_nan:,}")

    # print("\n")
    print(f"\nRows with ALL THREE IDs missing: {n_all_three_nan:,}")

    print("\n--- Unique pairs with key_col ---")
    print(f"({key_col}, {id1_col}) → {n_unique_pairs_id1:,} unique pairs")
    print(f"({key_col}, {id2_col}) → {n_unique_pairs_id2:,} unique pairs")
    print(f"({key_col}, {id3_col}) → {n_unique_pairs_id3:,} unique pairs")

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
    Cast dataframe columns to the specified data types,
    selecting only columns that exist in the dataframe.
    After conversion, reorder dataframe columns according to dtype_dict.
    """

    # 1. Keep only dtype rules that apply to existing columns
    applicable = {col: dtype for col, dtype in dtype_dict.items() if col in df.columns}

    # 2. Apply dtype conversions (ignore errors to avoid crashes)
    df = df.astype(applicable, errors="ignore")

    # 3. Reorder columns based on dtype_dict order
    ordered_cols = [col for col in dtype_dict.keys() if col in df.columns]
    remaining_cols = [col for col in df.columns if col not in ordered_cols]

    # 4. Return dataframe with ordered columns first, then the rest
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









