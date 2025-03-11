"""
Author: Sreejith Menon 
Executable script for setting up a database of vectors from the bioclip dataset
"""

import argparse
import chromadb
import datasets
import enum
import logging
import os
import torch 

from bioclip.predict import TreeOfLifeClassifier
from tqdm import tqdm
from typing import List

_DEFAULT_OUTPUT_DIR = os.path.join(os.getcwd(), 
                                   "vector_db")
_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger()

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("CUDA is available")
        return torch.device("cuda")
    elif torch.mps.is_available():
        logger.info("MPS is available")
        return torch.device("mps")
    else:
        logger.warning("CUDA and MPS are not available. Default to CPU")
        return torch.device("cpu")
    
class HfDatasetType(enum.Enum):
    BIRD = "Somnath01/Birds_Species"
    TREE_OF_LIFE = "imageomics/TreeOfLife-10M"

class BioclipVectorDatabase: 
    def __init__(self, dataset_type: HfDatasetType, 
                 collection_dir: str, 
                 split: str):
        self._dataset_type = dataset_type
        self._classifier = TreeOfLifeClassifier(
            device=_get_device())
        self._dataset = None
        self._collection_dir = collection_dir
        self._client = None
        self._collection = None
        
        self._prepare_dataset(
            split=split)
        self._init_collection()
        
    def _prepare_dataset(self, 
                         split: str) -> datasets.Dataset:
        """ Loads the dataset from Hugging Face to memory. """
        if split is None:
            logger.info(f"Loading entire dataset: {self._dataset_type.value}")
            self._dataset = datasets.load_dataset(self._dataset_type.value, 
                                         streaming=False)
        
        logger.info(f"Loading dataset: {self._dataset_type.value} for split: {split}")
        self._dataset = datasets.load_dataset(self._dataset_type.value, 
                                     split=split, 
                                     streaming=False)
        print(self._dataset[0])
    
    def _init_collection(self):
        """ Initializes the collection for storing the vectors. """
        self._client = chromadb.PersistentClient(path=self._collection_dir)

        self._collection = self._client.get_or_create_collection(
            name=self._dataset_type.name, 
            metadata={
                "hnsw:space": "cosine",
                "hnsw:search_ef": 10
            }
        )

    def _make_ids(self) -> List[str]:
        """ Generates unique ids for each record in the dataset. """
        if self._dataset_type.name == HfDatasetType.BIRD.name:
            return list(map(lambda x: str(x), range(len(self._dataset))))
        elif self._dataset_type.name == HfDatasetType.TREE_OF_LIFE.name:
            pass 

        raise ValueError(f"Dataset type: {self._dataset_type} not supported.")

    def _make_embeddings(self) -> List[List[float]]:
        """ Generates embeddings for each record in the dataset. """
        return list(
            map(
                lambda i: self._classifier.create_image_features_for_image(
                    self._dataset[i]["image"], 
                    normalize=True).tolist(),
                range(len(self._dataset)))
        )
    
    def load_database(self, reset: bool = False):
        if reset: 
            logger.info("Resetting the database.")
            self._client.delete_collection(self._collection.name)
            self._init_collection()
        
        ids = self._make_ids()
        embeddings = self._make_embeddings()
        assert len(ids) == len(embeddings), "Length of ids and embeddings should match."

        self._collection.add(embeddings=embeddings, ids=ids)
        logger.info(f"Database created with {len(ids)} records.")

    def get_vector_database(self):
        self._init_collection()
        return self._collection
    
    def query_neighbors(self, image, num_neightbors=5):
        pass 

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=lambda s: HfDatasetType[s.upper()],
        choices=list(HfDatasetType),
        required=True,
        help="Dataset to use for creating the vector database"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_DEFAULT_OUTPUT_DIR,
        help="Output directory to save the database"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Reset the entire vector database, if unset, only the new records are added"
    )

    args = parser.parse_args()
    dataset = args.dataset
    output_dir = args.output_dir

    logger.info(f"Creating database for dataset: {dataset}")
    logger.info(f"Creating database for dataset: {dataset.value}")
    logger.info(f"Output directory: {output_dir}")

    vdb = BioclipVectorDatabase(
        dataset_type=dataset, 
        collection_dir=output_dir, 
        split="")
    vdb.load_database(reset=args.reset)

if __name__ == "__main__":
    main()