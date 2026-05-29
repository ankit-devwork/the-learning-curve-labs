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

            # 🚀 BASE MODEL OVERRIDE & SANITIZATION LAYER
            # 1. Fetch raw value from environment or properties file fallback
            raw_model = os.getenv("BASE_MODEL", config.get("BASE_GENERATION_MODEL", default="gpt-4o-mini"))
            
            # 2. Force structural verification to ensure LiteLLM provider prefix is intact
            if "/" not in raw_model:
                # If it's a known llama or groq model name missing its prefix, prepend 'groq'
                if "llama" in raw_model.lower() or "mixtral" in raw_model.lower():
                    self.base_model = f"groq/{raw_model}"
                elif "gpt" in raw_model.lower():
                    self.base_model = f"openai/{raw_model}"
                elif "claude" in raw_model.lower():
                    self.base_model = f"anthropic/{raw_model}"
                else:
                    # Generic safe fallback to Groq since it's the primary system runtime engine
                    self.base_model = f"groq/{raw_model}"
            else:
                self.base_model = raw_model
            
            # 🚀 REDIS ENVIRONMENT OVERRIDES
            # Prioritizes REDIS_HOST/REDIS_PORT environment variables passed to Docker
            self.redis_host = os.getenv("REDIS_HOST", config.get("redis.host", default="localhost"))
            self.redis_port = int(os.getenv("REDIS_PORT", config.get_int("redis.port", default=6379)))
            self.redis_db = int(os.getenv("REDIS_DB", config.get_int("redis.db", default=0)))
            self.redis_default_ttl = config.get_int("redis.default_ttl", default=3600)
            self.sliding_response_ttl = config.get_int("redis.sliding_response_ttl", default=1800)
            
            self.redis_password = os.getenv("REDIS_PASSWORD", config.get("redis.password", default=""))
            if self.redis_password in ("\"\"", "''", ""):
                self.redis_password = None

            # 🚀 POSTGRES DB ENVIRONMENT OVERRIDES
            self.db_host = os.getenv("DB_HOST", config.get("db.host", default="localhost"))
            self.db_port = int(os.getenv("DB_PORT", config.get_int("db.port", default=5432)))
            self.db_user = os.getenv("DB_USER", config.get("db.user", default="myuser"))
            self.db_password = os.getenv("DB_PASSWORD", config.get("db.password", default="mypassword"))
            self.db_name = os.getenv("DB_NAME", config.get("db.name", default="mydatabase"))

            # 🚀 NEO4J ENVIRONMENT OVERRIDES
            # Prioritizes NEO4J_URI environment variable passed to Docker
            self.neo4j_uri = os.getenv("NEO4J_URI", config.get("neo4j.uri", default="bolt://localhost:7687"))
            self.neo4j_user = os.getenv("NEO4J_USER", config.get("neo4j.user", default="neo4j"))
            self.neo4j_password = os.getenv("NEO4J_PASSWORD", config.get("neo4j.password", default="mypassword123"))

            # Storage & Embedding configurations
            self.storage_local_dir = config.get("storage.local_dir", default="storage/uploads")
            self.storage_max_file_size_mb = config.get_int("storage.max_file_size_mb", default=50)
            self.text_embedding_model = config.get("text_embedding_model", default="text-embedding-3-small")
            self.chunk_size = config.get_int("chunk_size", default=1000)
            self.chunk_overlap = config.get_int("chunk_overlap", default=200)

            logger.info(f"Configuration initialized successfully. Sanitized Base Model Router: {self.base_model}")

        except Exception as err:
            logger.critical("Configuration initialization failed", error=str(err))
            raise

# Instantiate a single instance to be shared across the entire app
settings = Settings()