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
def transform_column_dtypes(df: pd.DataFrame):
    type_dict = {
        "DEMANDAPK": "int64",
        "FECHADEMANDA": "int64",
        "IDTARJETASANITARIA": "object",
        "DNI": "object",
        "NOMBRE_APELLIDOS": "object",
        "EDAD": "float64",
        "TIPOEDAD": "object",
        "EDAD_llamada": "float64",
        "TIPOEDAD_llamada": "float64",
        "SEXO": "object",
        "FNACIMIENTO": "object",
        "IDDEMANDA": "object",
        "IDRECURSO": "object",
        "IDASISTENCIA": "object",
        "IDALERTANTE": "object",
        "COMENTARIO": "object",
        "IDTIPODEMANDA1": "float64",
        "IDTIPODEMANDA2": "float64",
        "IDTIPODEMANDA3": "object",
        "IDMOTIVOEXCLUSION": "object",
        "IDTIPORECURSO": "object",
        "IDTIPORECURSOAGR": "object",
        "IDCODRESOLUCIONRECURSO": "object",
        "LTR": "object",
        "CADREC": "object",
        "IDEMPRESA": "object",
        "MOTIVOLITERAL": "object",
        "JUICIOCLINICO1": "object",
        "JUICIOCLINICO2": "object",
        "JUICIOCLINICO3": "object",
        "IDCODIGORESOLUCION": "object",
        "IDCODRESOLDEMANDA": "object",
        "IDCODRESOLASISTENCIA": "object",
        "IDHCEPES": "object",
        "IDRE": "object",
        "IDTARJETACORAZON": "object",
        "IDTARJETAAIRE": "object",
        "ESHIPERFRECUENTADOR": "Int64",
        "ESVULNERABLE": "float64",
        "PRIORIDAD": "object",
        "IDHCDM": "object",
        "IDESCENARIO": "object",
        "HORAENTRADACONTACTO": "object",
        "HORACONTESTACONTACTO": "object",
        "HORAACTIVACION": "object",
        "HORASALIDA": "object",
        "HORALLEGADA": "object",
        "HORACARGA": "object",
        "HORAOPERATIVA": "object",
        "HORADESTINO": "object",
        "HORADISPONIBLE": "object",
        "IDPROVINCIA": "int64",
        "DIRIDLOCALIDAD": "float64",
        "DIRIDLOCALIDADDEMANDA": "float64",
        "DIRIDPROVINCIAPACIENTE": "float64",
        "DIRIDLOCALIDADPACIENTE": "float64",
        "AFEC_LATITUD": "float64",
        "AFEC_LONGITUD": "float64",
        "ALER_COORDX": "float64",
        "ALER_LONGITUD": "float64",
        "ALER_OORDY": "float64",
        "DIRDESTINOIDPROVINCIARECURSO": "float64",
        "DIRDESTINOIDLOCALIDADRECURSO": "float64",
        "DIRDESTINOCALLERECURSO": "object",
        "NODO_TIPIFICACION_ID_PK": "int64",
        "FK_STRIA_PK_IMPACTO": "int64"
}

    return {key: val for key, val in type_dict.items() if key in df.columns}

    vent_df = vent_df.astype(kb.transform_column_dtypes(vent_df))



