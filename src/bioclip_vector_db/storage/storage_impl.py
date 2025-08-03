import chromadb
import logging
import faiss
import numpy as np
import os

from .storage_interface import StorageInterface
from .faiss_utils import IndexPartitionWriter
from typing import List, Dict
from collections import defaultdict


_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger()


class Chroma(StorageInterface):
    def init(self, name: str, **kwargs):
        if "metadata" not in kwargs:
            raise ValueError("Chromadb cannot be initialized without metadata.")

        if "collection_dir" not in kwargs:
            raise ValueError("Chromadb cannot be initialized without collection_dir.")

        self._metadata = kwargs["metadata"]
        self._collection_dir = kwargs["collection_dir"]

        logger.info("Initializing ChromaDb client.")
        self._chroma_client = chromadb.PersistentClient(path=self._collection_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name=name, metadata=self._metadata
        )
        logger.info(f"Created/Initialized collection: {self._collection.name}")

        return self

    def add_embedding(self, id: str, embedding: List[float], metadata: Dict[str, str]):
        self._collection.add(embeddings=[embedding], ids=[id], metadatas=[metadata])
        self._count += 1

    def batch_add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, str]],
    ):
        assert (
            len(ids) == len(embeddings) == len(metadatas)
        ), "Lengths of ids, embeddings, and metadatas must be the same."
        self._collection.add(embeddings=embeddings, ids=ids, metadatas=metadatas)
        self._count += len(ids)

    def query(self, id: str):
        return self._collection.get(id)["documents"]

    def reset(self, force=False):
        if force:
            self._chroma_client.delete_collection(self._collection.name)

        return self.init(
            name=self._collection.name,
            metadata=self._metadata,
            collection_dir=self._collection_dir,
        )


class FaissIvf(StorageInterface):
    """Faiss index with inverted file index. Requires training to use."""

    def init(self, name: str, **kwargs):
        if "collection_dir" not in kwargs:
            raise ValueError("Faiss cannot be initialized without collection_dir.")
        if "dimensions" not in kwargs:
            raise ValueError("Faiss cannot be initialized without dimensions.")
        if "nlist" not in kwargs:
            self._nlist = 2**15  # ~ 10x sqrt(N); N is 10M
        else:
            self._nlist = kwargs["nlist"]

        self._train_set_size = 50 * self._nlist

        self._collection_dir = kwargs["collection_dir"]
        self._dimensions = kwargs["dimensions"]
        self._factory_string = f"IVF{self._nlist},SQfp16"

        self._index = faiss.index_factory(self._dimensions, self._factory_string)
        logger.info(
            f"Initializing Faiss client with the factory string: {self._factory_string}."
        )
        self._centroid_index = "leader.index"
        self._local_index = "local_{idx}.index"

        self._writer = IndexPartitionWriter(
            self._index, 
            1000, # batch_size
            self._collection_dir
        )
        
        logger.info(f"Number of clusters: {self._nlist}")
        logger.info(f"Training set size: {self._train_set_size}")

        self._train_ids = []
        self._train_embeddings = []
        self._train_metadatas = []

        # todo: sreejith; has to be written to some other db store.
        self._metadata_store = {}
        return self

    def _make_temp_local_index_map(self):
        self._local_index_map = {
            i: self._local_index.format(idx=i) for i in range(self._nlist)
        }

    def _add_embedding_to_index(
        self, id: str, embedding: List[float], metadata: Dict[str, str]
    ):
        embedding_np = np.array([embedding]).astype("float32")
        self._metadata_store[self._index.ntotal] = {"id": id, "metadata": metadata}
        self._writer.add_embedding(embedding_np)

    def add_embedding(self, id: str, embedding: List[float], metadata: Dict[str, str]):
        if len(self._train_ids) < self._train_set_size:
            self._train_ids.append(id)
            self._train_embeddings.append(embedding)
            self._train_metadatas.append(metadata)
        elif not self._index.is_trained:
            self._train_index()
        else:
            self._add_embedding_to_index(id, embedding, metadata)

    def batch_add_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, str]],
    ):
        if len(self._train_ids) < self._train_set_size:
            self._train_ids.extend(ids)
            self._train_embeddings.extend(embeddings)
            self._train_metadatas.extend(metadatas)
        elif not self._index.is_trained:
            self._train_index()
        else:
            for id, embedding, metadata in zip(ids, embeddings, metadatas):
                self._add_embedding_to_index(id, embedding, metadata)

    def query(self, id: str):
        pass

    def reset(self, force=False):
        pass

    def _train_index(self):
        train_stack = np.vstack(self._train_embeddings)
        logging.info(f"Training index with shape: {train_stack.shape}")
        self._index.train(train_stack)
        logging.info("Training complete.")

        # once trained, add all the training data back into the db.
        for id, embedding, metadata in zip(
            self._train_ids, self._train_embeddings, self._train_metadatas
        ):
            self._add_embedding_to_index(id, embedding, metadata)

    def flush(self):
        self._writer.close()
        faiss.write_index(self._index, f"{self._collection_dir}/{self._centroid_index}")