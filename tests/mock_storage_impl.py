import src.bioclip_vector_db.storage.storage_interface as storage_interface

class MockStorageInterface(storage_interface.StorageInterface):
    def init(self, name: str, **kwargs):
        pass

    def add_embedding(self, id: str, embedding: list, metadata: dict):
        pass

    def batch_add_embeddings(self, ids: list, embeddings: list, metadatas: list):
        pass

    def query(self, id: str):
        pass

    def reset(self, force=False):
        pass