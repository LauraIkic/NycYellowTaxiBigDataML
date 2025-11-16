"""
Utility for dropping tables from the NYC Yellow Taxi data warehouse.

This module defines a small helper class that can be used to purge
dimension and fact tables from the PostgreSQL database.  It is
intended for local development when you need to start the ETL pipeline
from a clean slate.  Use this with caution: dropping tables will
permanently remove any data stored in them.

Example usage::

    from db_cleaner import DBCleaner

    # Connection details for your local database
    cleaner = DBCleaner(
        user="postgres",
        password="airbnb123d",
        host="localhost",
        port="5432",
        dbname="ny_taxi_dwh",
    )

    # List the tables to drop.  Adjust as needed.
    tables_to_drop = [
        "dim_datetime",
        "dim_fare",
        "dim_weather",
        "fact_trips",
    ]

    cleaner.drop_tables(tables_to_drop)

This will connect to the ``ny_taxi_dwh`` database and drop each
specified table if it exists, printing a message for each one.

Note
----
Dropping tables irreversibly deletes data.  Do not run this against a
production database.  It is meant for a local development database.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sqlalchemy import text

from db_connection import DBConnection


@dataclass
class DBCleaner:
    """Helper for dropping tables from the taxi data warehouse.

    Parameters
    ----------
    user : str
        Database user name.
    password : str
        Password for the database user.
    host : str
        Host name where the database server is running.
    port : str
        Port number on which the PostgreSQL server is listening.
    dbname : str
        Name of the database to connect to.
    """

    user: str
    password: str
    host: str
    port: str
    dbname: str

    def __post_init__(self) -> None:
        # Reuse the existing DBConnection class to obtain an SQLAlchemy engine.
        self._db_conn = DBConnection(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            dbname=self.dbname,
        )
        # Ensure the database exists and obtain an engine to it.
        self._db_conn.create_database_if_not_exists()
        self.engine = self._db_conn.connect()

    def drop_tables(self, tables: Iterable[str]) -> None:
        """Drop the specified tables from the database.

        Parameters
        ----------
        tables : iterable of str
            Names of the tables to drop.  Any tables that do not exist
            will be ignored.  The ``CASCADE`` option is used to drop
            dependent objects (such as foreign key constraints) safely.
        """
        with self.engine.begin() as conn:
            for table_name in tables:
                # Safely format the table name as an identifier.  SQLAlchemy
                # will escape it as needed when using ``text``.
                stmt = text(f"DROP TABLE IF EXISTS \"{table_name}\" CASCADE")
                conn.execute(stmt)
                print(f"[DBCleaner] Dropped table if exists: {table_name}")