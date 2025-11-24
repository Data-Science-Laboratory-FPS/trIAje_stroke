import pandas as pd
import numpy as np
import os

wd = os.getcwd()
print(f"El directorio de trabajo actual es: {wd}")

source_tables_path = '/opt/datos_compartidos/trIAje/data'
os.chdir(source_tables_path)

wd = os.getcwd()
print(f"El directorio de trabajo actual es: {wd}")

pd.set_option('display.max_columns', None)
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
    "PRIORIDAD": "priority",
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
    "resolution_code_id": "object",
    "demand_resolution_code_id": "category",
    "assistance_resolution_code_id": "category",
    "hcepes_id": "object",
    "idre": "object",
    "healthcard": "category",
    "aircard": "category",
    "is_frequent_user": "category",
    "is_vulnerable": "category",
    "priority": "category",
    "hcdm_id": "object",
    "location_patient": "category",
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
    "impact_fk_stria_pk": "category"
}



# Functions
## Function to extracts information on nº of id rows from a df
def summarize_numbers(df: pd.DataFrame):

    # --- 1) Basic counts ---
    n_rows = len(df)
    n_demandapk = df["DEMANDAPK"].nunique()
    n_nuhsa = df["IDTARJETASANITARIA"].nunique()
    n_nuhsa_nan = df["IDTARJETASANITARIA"].isna().sum()
    n_dni = df["DNI"].nunique()
    n_dni_nan = df["DNI"].isna().sum()
    n_nombre = df["NOMBRE_APELLIDOS"].nunique()
    n_nombre_nan = df["NOMBRE_APELLIDOS"].isna().sum()

    # --- 2) Unique (DEMANDAPK, ID) pairs (exclude rows with null NUHSA) ---
    unique_pairs_NUHSA = (
        df[["DEMANDAPK", "IDTARJETASANITARIA"]]
        .dropna(subset=["IDTARJETASANITARIA"])
        .drop_duplicates()
    )
    n_unique_pairs_NUHSA = len(unique_pairs_NUHSA)
    
    unique_pairs_DNI = (
        df[["DEMANDAPK", "DNI"]]
        .dropna(subset=["DNI"])
        .drop_duplicates()
    )
    n_unique_pairs_DNI = len(unique_pairs_DNI)
    
    unique_pairs_nombre = (
        df[["DEMANDAPK", "NOMBRE_APELLIDOS"]]
        .dropna(subset=["NOMBRE_APELLIDOS"])
        .drop_duplicates()
    )
    n_unique_pairs_nombre = len(unique_pairs_nombre)

    # --- Final print in one block ---
    print("\n--- Summary of DEMANDA and PATIENTIDs ---")
    print(f"Nº of rows: {n_rows:,}")
    print(f"Nº of unique DEMANDAPK: {n_demandapk:,}")
    print(f"Nº of unique valid NUHSA: {n_nuhsa:,}")
    print(f"Nº of NaN in NUHSA: {n_nuhsa_nan:,}")
    print(f"Nº of unique valid DNI: {n_dni:,}")
    print(f"Nº of NaN in DNI: {n_dni_nan:,}")
    print(f"Nº of unique valid Names and Surnames: {n_nombre:,}")
    print(f"Nº of NaN in Names and Surnames: {n_nombre_nan:,}")
    print(f"Unique (DEMANDAPK, NUHSA) pairs (ID not null): {n_unique_pairs_NUHSA:,}")
    print(f"Unique (DEMANDAPK, DNI) pairs (ID not null): {n_unique_pairs_DNI:,}")
    print(f"Unique (DEMANDAPK, Name and Surname) pairs (ID not null): {n_unique_pairs_nombre:,}")

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
def df_pipeline_test(df: pd.DataFrame, flag: str) -> pd.DataFrame:
    if flag == "all":
        return df
    
    elif flag == "test":
        return df.iloc[:100000].copy()
    
    else:
        raise ValueError('Flag must be either "all" or "test".')

## Count how many rows have ALL specified columns as None/NaN.
def count_rows_all_none(df: pd.DataFrame, columns_to_check: list) -> int:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns_to_check : list
        List of column names to inspect.

    Returns
    -------
    int
        Number of rows where all specified columns are None/NaN.
    """

    # Mask of nulls for selected columns
    null_mask = df[columns_to_check].isnull()

    # Row-wise check: all specified columns are null
    all_null_rows = null_mask.all(axis=1)

    # Count how many rows match
    count = all_null_rows.sum()

    # Integrated print
    print(f"\nNumber of rows with None in {columns_to_check}: {count}")

    return count


# Function for converting data types to reduce memory for computation
def transform_column_dtypes(df: pd.DataFrame, dtype_dict=dtype_cols):
    """
    Cast dataframe columns to the specified data types,
    selecting only columns that exist in the dataframe.
    """
    applicable = {col: dtype for col, dtype in dtype_dict.items() if col in df.columns}
    return df.astype(applicable, errors="ignore")





