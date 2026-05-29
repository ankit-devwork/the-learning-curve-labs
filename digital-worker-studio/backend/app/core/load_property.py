import os

from app.observability.logger import logger
from .config_reader import PropertyFileReader

class Settings:
    """
    Centralized configuration management.
    Loads settings from config.properties into accessible attributes with structural environmental overrides.
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
            base_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.properties")
            
            if not os.path.exists(config_path):
                config_path = os.path.join(os.getcwd(), "config.properties")
            logger.info(f"Loading application properties from: {config_path}")
            config = PropertyFileReader(config_path)

            # 🚀 1. BASE MODEL & EMBEDDING CONFIGURATIONS
            raw_model = os.getenv("BASE_MODEL", config.get("BASE_GENERATION_MODEL", default="gpt-4o-mini"))
            if "/" not in raw_model:
                if "llama" in raw_model.lower() or "mixtral" in raw_model.lower():
                    self.base_model = f"groq/{raw_model}"
                elif "gpt" in raw_model.lower():
                    self.base_model = f"openai/{raw_model}"
                elif "claude" in raw_model.lower():
                    self.base_model = f"anthropic/{raw_model}"
                else:
                    self.base_model = f"groq/{raw_model}"
            else:
                self.base_model = raw_model

            self.text_embedding_model = os.getenv("TEXT_EMBEDDING_MODEL", config.get("text_embedding_model", default="text-embedding-3-small"))
            self.chunk_size = int(os.getenv("CHUNK_SIZE", config.get_int("chunk_size", default=1000)))
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", config.get_int("chunk_overlap", default=200)))

            # 🚀 2. REDIS ENVIRONMENT CONFIGURATIONS
            self.redis_host = os.getenv("REDIS_HOST", config.get("redis.host", default="localhost"))
            self.redis_port = int(os.getenv("REDIS_PORT", config.get_int("redis.port", default=6379)))
            self.redis_db = int(os.getenv("REDIS_DB", config.get_int("redis.db", default=0)))
            self.redis_default_ttl = int(os.getenv("REDIS_DEFAULT_TTL", config.get_int("redis.default_ttl", default=3600)))
            self.sliding_response_ttl = int(os.getenv("REDIS_SLIDING_RESPONSE_TTL", config.get_int("redis.sliding_response_ttl", default=1800)))
            
            self.redis_password = os.getenv("REDIS_PASSWORD", config.get("redis.password", default=""))
            if str(self.redis_password).strip() in ("\"\"", "''", ""):
                self.redis_password = None

            # 🚀 3. POSTGRES DATABASE CONFIGURATIONS
            self.db_host = os.getenv("DB_HOST", config.get("db.host", default="localhost"))
            self.db_port = str(os.getenv("DB_PORT", config.get("db.port", default="5432"))).strip()
            self.db_user = os.getenv("DB_USER", config.get("db.user", default="postgres"))
            self.db_password = os.getenv("DB_PASSWORD", config.get("db.password", default="mypassword"))
            self.db_name = os.getenv("DB_NAME", config.get("db.name", default="digital_worker_db"))

            # 🚀 4. NEO4J ENVIRONMENT CONFIGURATIONS
            self.neo4j_uri = os.getenv("NEO4J_URI", config.get("neo4j.uri", default="bolt://localhost:7687"))
            self.neo4j_user = os.getenv("NEO4J_USER", config.get("neo4j.user", default="neo4j"))
            self.neo4j_password = os.getenv("NEO4J_PASSWORD", config.get("neo4j.password", default="mypassword123"))

            # 🚀 5. STORAGE CONFIGURATIONS
            self.storage_local_dir = os.getenv("STORAGE_LOCAL_DIR", config.get("storage.local_dir", default="storage/uploads"))
            self.storage_max_file_size_mb = int(os.getenv("STORAGE_MAX_FILE_SIZE_MB", config.get_int("storage.max_file_size_mb", default=50)))

            logger.info(f"Configuration initialized successfully. Sanitized Base Model Router: {self.base_model}")

        except Exception as err:
            logger.critical("Configuration initialization failed", error=str(err))
            raise

# Instantiate a single instance to be shared across the entire app
settings = Settings()