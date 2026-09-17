from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    triaje_project_path: str
    source_tables_path: str
    edatoscaso_load_path: str
    master_table_path: str
    master_table_4demands_path: str
    table_embedding_sbert_path: str
    table_embedding_rigobert_path: str
    triaje_table_cleaned_path: str
    stroke_table_cleaned_path: str
    cardiacarrest_table_modeling: str
    chestpain_table_modeling: str
    dyspnea_table_modeling: str

    # Run embedding function of literal_reason column: Yes (1) or No (0)
    run_embedding_sbert: int = 0
    run_embedding_rigobert: int = 0
    # Run pipeline in a subset of rows: 'all' for all source rows, 'test' for a specific
    # subset of rows indicated in df_pipeline_test function in kbase/preprocessing.py
    test_pipeline: str = "all"
    
    class Config:
        env_file = ".env"

settings = Settings()

