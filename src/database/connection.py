"""
Database connection management and utilities.

This module provides functions for creating database connections,
managing sessions, and initializing the database schema.
"""

import os
from typing import Generator
from sqlalchemy import create_engine as sa_create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from ..models.database import Base, create_all_tables


def get_database_url() -> str:
    """
    Get database URL from environment variables or use default.
    
    Returns:
        str: Database URL for SQLAlchemy
    """
    # Check for environment variable first
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url
    
    # Default to SQLite database in data directory
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    db_path = os.path.join(data_dir, "experiments.db")
    return f"sqlite:///{db_path}"


def create_engine(database_url: str = None, **kwargs) -> Engine:
    """
    Create SQLAlchemy engine with appropriate configuration.
    
    Args:
        database_url: Database URL. If None, uses get_database_url()
        **kwargs: Additional arguments passed to create_engine
        
    Returns:
        Engine: Configured SQLAlchemy engine
    """
    if database_url is None:
        database_url = get_database_url()
    
    # Default engine configuration
    engine_kwargs = {
        "echo": os.getenv("DATABASE_ECHO", "false").lower() == "true",
        "pool_pre_ping": True,
    }
    
    # SQLite-specific configuration
    if database_url.startswith("sqlite"):
        engine_kwargs.update({
            "connect_args": {"check_same_thread": False},
            "pool_recycle": -1,
        })
    
    # Override with provided kwargs
    engine_kwargs.update(kwargs)
    
    return sa_create_engine(database_url, **engine_kwargs)


def get_session_factory(engine: Engine = None) -> sessionmaker:
    """
    Create a session factory for database operations.
    
    Args:
        engine: SQLAlchemy engine. If None, creates a new one.
        
    Returns:
        sessionmaker: Session factory
    """
    if engine is None:
        engine = create_engine()
    
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False
    )


@contextmanager
def get_session(session_factory: sessionmaker = None) -> Generator[Session, None, None]:
    """
    Context manager for database sessions with automatic cleanup.
    
    Args:
        session_factory: Session factory. If None, creates a new one.
        
    Yields:
        Session: Database session
        
    Example:
        with get_session() as session:
            experiment = session.query(Experiment).first()
    """
    if session_factory is None:
        session_factory = get_session_factory()
    
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database(engine: Engine = None, create_tables: bool = True) -> Engine:
    """
    Initialize the database with tables and initial data.
    
    Args:
        engine: SQLAlchemy engine. If None, creates a new one.
        create_tables: Whether to create database tables
        
    Returns:
        Engine: The database engine used
        
    Note:
        In production, use Alembic migrations instead of create_tables=True
    """
    if engine is None:
        engine = create_engine()
    
    if create_tables:
        create_all_tables(engine)
    
    return engine


# Global session factory for application use
_session_factory = None


def get_global_session_factory() -> sessionmaker:
    """
    Get or create the global session factory.
    
    Returns:
        sessionmaker: Global session factory
    """
    global _session_factory
    if _session_factory is None:
        _session_factory = get_session_factory()
    return _session_factory


def set_global_session_factory(session_factory: sessionmaker) -> None:
    """
    Set the global session factory.
    
    Args:
        session_factory: Session factory to use globally
    """
    global _session_factory
    _session_factory = session_factory