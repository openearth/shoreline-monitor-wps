"""
Shared SQLAlchemy engine for the WPS Flask app.

Reads the [database] section from configuration.txt (same credentials used by
shoreline_getprofile._initialize_config and shoreline_getstats._initialize_config).
The engine is created once and reused across requests.
"""

import configparser
import logging
import os

from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger("PYWPS")

_engine = None


def get_engine():
    """Return the shared SQLAlchemy engine, creating it on first call.

    Reads PG_USER, PG_PASS, PG_HOST, PG_DB from configuration.txt [database].

    Returns:
        sqlalchemy.engine.Engine

    Raises:
        SQLAlchemyError: if the connection cannot be established.
    """
    global _engine
    if _engine is not None:
        return _engine

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_file = os.path.join(project_root, "configuration.txt")

    config = configparser.ConfigParser()
    config.read(config_file, encoding="utf-8")

    pg_user = config.get("database", "PG_USER")
    pg_pass = config.get("database", "PG_PASS")
    pg_host = config.get("database", "PG_HOST")
    pg_db = config.get("database", "PG_DB")
    pg_port = 5432

    try:
        _engine = create_engine(
            f"postgresql+psycopg2://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}"
        )
        with _engine.connect():
            pass
        logger.info("db.py: shared engine connected to %s/%s", pg_host, pg_db)
    except SQLAlchemyError as err:
        logger.error("db.py: database connection failed: %s", err)
        raise

    return _engine
