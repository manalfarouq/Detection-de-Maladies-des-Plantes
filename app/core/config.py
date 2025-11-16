from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    USER_DB: str
    PASSWORD: str
    HOST: str
    PORT: int
    DATABASE: str

    @property
    def DATABASE_URL(self):
        return f"postgresql://{self.USER_DB}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}"

    # Syntaxe alternative pour Pydantic v2 si ConfigDict n'existe pas
    model_config = {"env_file": ".env"}

settings = Settings()