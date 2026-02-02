from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    source_tables_path: str
    edatoscaso_load_path: str
    master_table_path: str
    master_table_4demands_path: str
    triaje_table_cleaned_path: str

    class Config:
        env_file = ".env"

settings = Settings()

