import sqlite3
import threading
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class MetadataDatabase:
    """A thread-safe SQLite-based metadata store for FAISS."""

    def __init__(self, db_path: str):
        """
        Initializes the MetadataDatabase.

        Args:
            db_path: The path to the SQLite database file.
        """
        self.db_path = db_path
        self.local = threading.local()
        self.create_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Gets a thread-local database connection."""
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        return self.local.connection

    def create_table(self):
        """Creates the metadata table if it doesn't exist."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        faiss_id INTEGER PRIMARY KEY,
                        original_id TEXT NOT NULL,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.info("Create table successful.")
        except sqlite3.Error as e:
            logger.error(f"Error creating table: {e}")
            raise

    def add_mapping(self, faiss_id: int, original_id: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Adds a mapping between a FAISS ID and an original ID.

        Args:
            faiss_id: The FAISS index ID.
            original_id: The original ID of the document/function.
            metadata: Optional dictionary of metadata to store as a JSON string.
        """
        conn = self._get_connection()
        metadata_json = json.dumps(metadata) if metadata else None
        try:
            with conn:
                conn.execute(
                    "INSERT INTO metadata (faiss_id, original_id, metadata) VALUES (?, ?, ?)",
                    (faiss_id, original_id, metadata_json)
                )
        except sqlite3.IntegrityError:
            logger.warning(f"faiss_id {faiss_id} already exists. Ignoring.")
        except sqlite3.Error as e:
            logger.error(f"Error adding mapping: {e}")
            raise

    def get_original_id(self, faiss_id: int) -> Optional[str]:
        """
        Retrieves the original ID for a given FAISS ID.

        Args:
            faiss_id: The FAISS index ID.

        Returns:
            The original ID, or None if not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT original_id FROM metadata WHERE faiss_id = ?", (faiss_id,))
            result = cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting original_id: {e}")
            raise

    def get_faiss_id(self, original_id: str) -> Optional[int]:
        """
        Retrieves the FAISS ID for a given original ID.

        Args:
            original_id: The original ID.

        Returns:
            The FAISS ID, or None if not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT faiss_id FROM metadata WHERE original_id = ?", (original_id,))
            result = cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting faiss_id: {e}")
            raise

    def close(self):
        """Closes the thread-local database connection."""
        if hasattr(self.local, "connection"):
            self.local.connection.close()
            del self.local.connection
