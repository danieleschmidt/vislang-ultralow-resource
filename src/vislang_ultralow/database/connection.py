"""Database connection management."""

import os
import logging
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, Engine, event
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
import redis
from redis import Redis

logger = logging.getLogger(__name__)

# SQLAlchemy base
Base = declarative_base()


class DatabaseManager:
    """Database connection manager with connection pooling."""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600
    ):
        """Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection URL
            redis_url: Redis connection URL  
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_timeout: Pool timeout in seconds
            pool_recycle: Pool recycle time in seconds
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "postgresql://postgres:password@localhost:5432/vislang"
        )
        self.redis_url = redis_url or os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0"
        )
        
        # Create PostgreSQL engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
        )
        
        # Add connection event listeners
        event.listen(self.engine, 'connect', self._on_connect)
        event.listen(self.engine, 'checkout', self._on_checkout)
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Test Redis connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
        
        logger.info(f"Database manager initialized with URL: {self._mask_url(self.database_url)}")
    
    @staticmethod
    def _mask_url(url: str) -> str:
        """Mask sensitive information in URL."""
        if "@" in url:
            parts = url.split("@")
            masked_auth = parts[0].split("://")[0] + "://***:***"
            return masked_auth + "@" + parts[1]
        return url
    
    @staticmethod
    def _on_connect(dbapi_connection, connection_record):
        """Handle new database connections."""
        logger.debug("New database connection established")
    
    @staticmethod  
    def _on_checkout(dbapi_connection, connection_record, connection_proxy):
        """Handle connection checkout from pool."""
        logger.debug("Database connection checked out from pool")
    
    def create_tables(self) -> None:
        """Create all database tables."""
        try:
            # Import models to register them
            from . import models
            
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self) -> None:
        """Drop all database tables."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_redis(self) -> Optional[Redis]:
        """Get Redis client."""
        return self.redis_client
    
    def health_check(self) -> dict:
        """Perform health check on database connections."""
        health = {
            "database": False,
            "redis": False,
            "pool_size": 0,
            "pool_checked_out": 0
        }
        
        # Check PostgreSQL
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            health["database"] = True
            health["pool_size"] = self.engine.pool.size()
            health["pool_checked_out"] = self.engine.pool.checkedout()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
        
        # Check Redis
        if self.redis_client:
            try:
                self.redis_client.ping()
                health["redis"] = True
            except Exception as e:
                logger.error(f"Redis health check failed: {e}")
        
        return health
    
    def close(self) -> None:
        """Close all connections."""
        if hasattr(self, 'engine'):
            self.engine.dispose()
        if self.redis_client:
            self.redis_client.close()
        logger.info("Database connections closed")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get or create global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_session() -> Generator[Session, None, None]:
    """Get database session - convenience function."""
    db_manager = get_database_manager()
    with db_manager.get_session() as session:
        yield session


def get_redis() -> Optional[Redis]:
    """Get Redis client - convenience function."""
    db_manager = get_database_manager()
    return db_manager.get_redis()