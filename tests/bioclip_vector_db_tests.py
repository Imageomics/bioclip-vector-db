import unittest
from unittest.mock import patch, MagicMock
from src.bioclip_vector_db.vector_db import BioclipVectorDatabase, HfDatasetType

class TestBioclipVectorDatabase(unittest.TestCase):

    @patch('src.bioclip_vector_db.vector_db.TreeOfLifeClassifier')
    @patch('src.bioclip_vector_db.vector_db.datasets')
    @patch('src.bioclip_vector_db.vector_db.chromadb')
    def setUp(self, mock_chromadb, mock_datasets, mock_classifier):
        self.mock_classifier = mock_classifier.return_value
        self.mock_datasets = mock_datasets
        self.mock_chromadb = mock_chromadb
        self.mock_client = MagicMock()
        self.mock_chromadb.PersistentClient.return_value = self.mock_client
        self.mock_collection = MagicMock()
        self.mock_client.get_or_create_collection.return_value = self.mock_collection

        self.dataset_type = HfDatasetType.BIRD
        self.collection_dir = "/tmp/vector_db"
        self.split = "train"

        self.vdb = BioclipVectorDatabase(
            dataset_type=self.dataset_type,
            collection_dir=self.collection_dir,
            split=self.split
        )

    def test_sample(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()