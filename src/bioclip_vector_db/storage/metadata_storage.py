import sqlite3
import threading
import json
import logging
from typing import Optional, Dict, Any, List
import os

logger = logging.getLogger(__name__)


class MetadataDatabase:
    """
    A thread-safe SQLite-based metadata store for FAISS.

    This database stores the mapping between a FAISS vector ID and its original
    Image ID. Since the vectors may be partitioned into multiple indexes, the partition
    ID is also stored to uniquely identify the location of the vector.
    """

    def __init__(self, db_path: str, reset: bool = False):
        """
        Initializes the MetadataDatabase.

        Args:
            db_path: The path to the SQLite database file.
        """
        self.db_path = os.path.join(db_path, "metadata.db")
        self.local = threading.local()
        if reset:
            self._reset()
        self.create_table()

    def _get_connection(self) -> sqlite3.Connection:
        """Gets a thread-local database connection."""
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(
                self.db_path, check_same_thread=False
            )
        return self.local.connection

    def create_table(self):
        """Creates the metadata table if it doesn't exist."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS id_mapping (
                        partition_id INTEGER NOT NULL,
                        faiss_id INTEGER NOT NULL,
                        original_id TEXT NOT NULL,
                        metadata BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (partition_id, faiss_id)
                    )
                """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_original_id ON id_mapping (original_id)
                    """
                )
                logger.info("SQLITE: Create table successful.")
        except sqlite3.Error as e:
            logger.error(f"Error creating table: {e}")
            raise

        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM id_mapping")
        result = cursor.fetchone()
        logger.info(f"Total number of records: {result[0]}")

    def add_mapping(
        self,
        partition_id: int,
        faiss_id: int,
        original_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Adds a mapping between a FAISS ID and an original ID for a given partition.

        Args:
            partition_id: The ID of the partition file.
            faiss_id: The FAISS index ID within the partition.
            original_id: The original ID of the document/function.
            metadata: Optional dictionary of metadata to store as a JSON string.
        """
        conn = self._get_connection()
        metadata_blob = json.dumps(metadata).encode("utf-8") if metadata else None
        try:
            with conn:
                conn.execute(
                    "INSERT INTO id_mapping (partition_id, faiss_id, original_id, metadata) VALUES (?, ?, ?, ?)",
                    (int(partition_id), int(faiss_id), original_id, metadata_blob),
                )
                logger.debug(
                    f"Added mapping: partition_id={partition_id}, faiss_id={faiss_id}, original_id={original_id}"
                )
        except sqlite3.IntegrityError:
            logger.warning(
                f"faiss_id {faiss_id} in partition {partition_id} already exists. Ignoring."
            )
        except sqlite3.Error as e:
            logger.error(f"Error adding mapping: {e}")
            raise

    def get_original_id(self, partition_id: int, faiss_id: int) -> Optional[str]:
        """
        Retrieves the original ID for a given FAISS ID in a specific partition.

        Args:
            partition_id: The ID of the partition file.
            faiss_id: The FAISS index ID.

        Returns:
            The original ID, or None if not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT original_id FROM id_mapping WHERE partition_id = ? AND faiss_id = ?",
                (int(partition_id), int(faiss_id)),
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            logger.error(f"Error getting original_id: {e}")
            raise

    def get_metadata(self, partition_id: int, faiss_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves the metadata for a given FAISS ID in a specific partition.

        Args:
            partition_id: The ID of the partition file.
            faiss_id: The FAISS index ID.

        Returns:
            The metadata dictionary, or None if not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT metadata FROM id_mapping WHERE partition_id = ? AND faiss_id = ?",
                (int(partition_id), int(faiss_id)),
            )
            result = cursor.fetchone()
            if result and result[0]:
                return json.loads(result[0].decode('utf-8'))
            return None
        except sqlite3.Error as e:
            logger.error(f"Error getting metadata: {e}")
            raise

    def get_metadata(self, original_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the metadata for a given original ID.

        Args:
            original_id: The original ID
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT metadata FROM id_mapping WHERE original_id = ?",
                (original_id,),
            )
            result = cursor.fetchone()
            if result and result[0]:
                return json.loads(result[0].decode("utf-8"))
            return None
        except sqlite3.Error as e:
            logger.error(f"Error getting metadata: {e}")
            raise

    def batch_get_original_id(self, partition_id: int, faiss_ids: List[int]) -> Dict[int, str]:
        # Implement me.
        pass

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
            cursor.execute(
                "SELECT faiss_id FROM id_mapping WHERE original_id = ?", (original_id,)
            )
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

    def _reset(self):
        """Deletes the metadata table."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DROP TABLE IF EXISTS id_mapping")
                logger.info("SQLITE: Reset table successful.")
        except sqlite3.Error as e:
            logger.error(f"Error resetting table: {e}")
            raise
