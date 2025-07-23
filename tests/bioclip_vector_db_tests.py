import unittest
from unittest.mock import patch, MagicMock
from src.bioclip_vector_db.vector_db import BioclipVectorDatabase
import src.bioclip_vector_db.storage.storage_factory as storage_factory
from . import mock_storage_impl

class TestBioclipVectorDatabase(unittest.TestCase):

    @patch('src.bioclip_vector_db.vector_db.datasets')
    @patch('src.bioclip_vector_db.vector_db.TreeOfLifeClassifier')
    def setUp(self, mock_datasets, mock_classifier):
        self.mock_classifier = mock_classifier.return_value
        self.mock_datasets = mock_datasets
        self.mock_collection = MagicMock()
        self.storage = mock_storage_impl.MockStorageInterface()

    def test_init_ok(self):
        vdb = BioclipVectorDatabase(
            dataset_type=storage_factory.HfDatasetType.BIRD,
            storage=self.storage,
            split="train"
        )


if __name__ == '__main__':
    unittest.main()