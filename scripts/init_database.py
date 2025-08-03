#!/usr/bin/env python3
"""Database initialization script for VisLang-UltraLow-Resource."""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vislang_ultralow.database import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize database with schema and test data."""
    try:
        # Initialize database manager
        db_manager = DatabaseManager()
        
        # Create tables
        logger.info("Creating database tables...")
        db_manager.create_tables()
        
        # Run health check
        health = db_manager.health_check()
        logger.info(f"Database health check: {health}")
        
        if health['database']:
            logger.info("✅ Database initialization completed successfully!")
        else:
            logger.error("❌ Database health check failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())