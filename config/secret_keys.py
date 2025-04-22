from pydantic_settings import BaseSettings, SettingsConfigDict

class ProjectConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)
    
    api_key:str
    
project_config = ProjectConfig()