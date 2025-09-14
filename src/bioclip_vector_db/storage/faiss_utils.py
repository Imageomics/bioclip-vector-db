import logging
import faiss
import numpy as np
import os

from collections import defaultdict
from .metadata_storage import MetadataDatabase

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger()


class IndexPartitionWriter:
    """
    A class to partition embeddings based on a trained Faiss quantizer and
    write them to temporary batch files on disk.

    Also holds a pointer to a sqllite database to store additional metadata.
    """

    def __init__(
        self,
        centroid_index: faiss.Index,
        batch_size: int,
        collection_dir: str,
        cleanup_temp_files=False,
    ):
        """Initializes the IndexPartitionWriter.

        Args:
            centroid_index: The trained Faiss index to use for partitioning.
            batch_size: The number of embeddings to buffer before writing to disk.
            collection_dir: The directory to write the partitioned indexes to.
            cleanup_temp_files: Whether to remove temporary numpy files after creating the indexes.
        """
        self._centroid_index = centroid_index
        self._partition_to_embedding_map = defaultdict(list)
        self._batch_size = batch_size
        self._collection_dir = collection_dir

        self._centroid_index_file = "leader.index"
        self._local_index_file = "local_{idx}.index"
        self._cleanup_temp_files = cleanup_temp_files
        self._metadata_db = MetadataDatabase(collection_dir)
        self._partition_faiss_ids = defaultdict(int)

        # Ensure the output directory exists
        os.makedirs(self._collection_dir, exist_ok=True)

    @staticmethod
    def _make_temp_numpy_file(collection_dir: str, partition_id: int):
        """Creates a temporary numpy file path.

        Args:
            collection_dir: The directory where the file will be located.
            partition_id: The partition ID to include in the filename.

        Returns:
            The full path to the temporary numpy file.
        """
        return os.path.join(collection_dir, f"partition_{partition_id}.npy")

    def _write_partition_to_file(self, partition_id: int):
        """Helper method to write a partition's buffer to disk.

        Args:
            partition_id: The ID of the partition to write.
        """
        embeddings_to_write = np.vstack(self._partition_to_embedding_map[partition_id])
        file_path = IndexPartitionWriter._make_temp_numpy_file(
            self._collection_dir, partition_id
        )

        # If the file already exists, load its content and append the new embeddings.
        # Writing in append mode won't work since np.save adds additional headers.
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, "rb") as f:
                existing_embeddings = np.load(f)
            embeddings_to_write = np.vstack([existing_embeddings, embeddings_to_write])

        with open(file_path, "wb") as f:
            np.save(f, embeddings_to_write)

        logger.info(
            f"Flushed {len(self._partition_to_embedding_map[partition_id])} embeddings to {file_path}. Total in file: {len(embeddings_to_write)}."
        )

        # Clear the buffer.
        self._partition_to_embedding_map[partition_id].clear()

    def _maybe_flush_buffers(self):
        """Checks all partition buffers and writes them to disk if they exceed batch size."""
        # Iterate over a copy of keys for safe modification
        for partition_id in list(self._partition_to_embedding_map.keys()):
            if len(self._partition_to_embedding_map[partition_id]) >= self._batch_size:
                self._write_partition_to_file(partition_id)

    def add_embedding(self, original_id: str, embedding: np.ndarray):
        """Adds a single embedding vector to the appropriate partition buffer.

        Args:
            embedding: The embedding vector to add.
            original_id: The original ID of the embedding.
        """
        # Ensure input is a 2D array for Faiss
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        _, partition_ids = self._centroid_index.quantizer.search(embedding, 1)
        partition_id = partition_ids[0][0]

        faiss_id = self._partition_faiss_ids[partition_id]
        self._metadata_db.add_mapping(partition_id, faiss_id, original_id)
        self._partition_faiss_ids[partition_id] += 1

        self._partition_to_embedding_map[partition_id].append(embedding)
        self._maybe_flush_buffers()

    def _flush(self):
        """Writes all remaining embeddings from all buffers to disk."""
        logger.info("Final flush: writing all remaining data to disk.")
        for partition_id in list(self._partition_to_embedding_map.keys()):
            # Check if there's anything left to write
            if self._partition_to_embedding_map[partition_id]:
                self._write_partition_to_file(partition_id)

    def _add_to_index_partitions(self):
        """Creates local Faiss indexes from the temporary numpy files."""
        for partition_id in list(self._partition_to_embedding_map.keys()):
            logger.info(
                f"Preparing to create the local index for partition: {partition_id}"
            )
            temp_file = IndexPartitionWriter._make_temp_numpy_file(
                self._collection_dir, partition_id
            )
            with open(temp_file, "rb") as f:
                embeddings = np.load(f)
            local_idx = faiss.IndexFlatIP(embeddings.shape[1])
            local_idx.add(embeddings)
            faiss.write_index(
                local_idx,
                f"{self._collection_dir}/{self._local_index_file.format(idx=partition_id)}",
            )
            logger.info(
                f"Write complete for index file: {self._local_index_file.format(idx=partition_id)}"
            )

            if self._cleanup_temp_files:
                os.remove(temp_file)

    def close(self):
        """Finalizer that flushes the temp buffers and creates local indexes."""
        self._flush()
        self._add_to_index_partitions()

        # creates the centroid index once all the partitions are written.
        faiss.write_index(
            self._centroid_index, f"{self._collection_dir}/{self._centroid_index_file}"
        )