import os

from pydantic import config
from app.observability.logger import logger
from .config_reader import PropertyFileReader

class Settings:
    """
    Centralized configuration management.
    Loads settings from config.properties into accessible attributes.
    """
    _instance = None

    def __new__(cls):
        # Singleton pattern: ensures the configuration file is parsed only once
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        try:
            # Safely locate config.properties relative to this directory
            # Adjust the file path hierarchy ("../../") depending on your precise layout
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.abspath(os.path.join(base_dir, "../../../config.properties"))
            
            logger.info(f"Loading application properties from: {config_path}")
            config = PropertyFileReader(config_path)

            # Assign properties to instance attributes so they persist
            self.base_model = config.get("BASE_GENERATION_MODEL", default="gpt-4o-mini")
            # NEW: Load Redis settings securely
            self.redis_host = config.get("redis.host", default="localhost")
            self.redis_port = config.get_int("redis.port", default=6379)
            self.redis_db = config.get_int("redis.db", default=0)
            self.redis_default_ttl = config.get_int("redis.default_ttl", default=3600)
            self.sliding_response_ttl = config.get_int("redis.sliding_response_ttl", default=1800)
            # Use an empty string if password isn't set
            self.redis_password = config.get("redis.password", default="")
            if self.redis_password in ("\"\"", "''"):
                self.redis_password = None

            self.db_host = config.get("db.host", default="localhost")
            self.db_port = config.get_int("db.port", default=5432)
            self.db_user = config.get("db.user", default="myuser")
            self.db_password = config.get("db.password", default="mypassword")
            self.db_name = config.get("db.name", default="mydatabase")

            self.neo4j_uri = config.get("neo4j.uri", default="bolt://localhost:7687")
            self.neo4j_user = config.get("neo4j.user", default="neo4j")
            self.neo4j_password = config.get("neo4j.password", default="mypassword123")

            self.storage_local_dir = config.get("storage.local_dir", default="storage/uploads")
            self.storage_max_file_size_mb = config.get_int("storage.max_file_size_mb", default=50)

            self.text_embedding_model = config.get("text_embedding_model", default="text-embedding-3-small")
            self.chunk_size = config.get_int("chunk_size", default=1000)
            self.chunk_overlap = config.get_int("chunk_overlap", default=200)

            logger.info(f"Configuration initialized successfully. Base Model: {self.base_model}")

        except Exception as err:
            logger.critical("Configuration initialization failed", error=str(err))
            # Raise exception to prevent the application from starting in an invalid state
            raise

# Instantiate a single instance to be shared across the entire app
settings = Settings()