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
            raise ValueError("Split cannot be None. Please provide a valid split.")
        
        logger.info(f"Loading dataset: {self._dataset_type.value} for split: {split}")
        self._dataset = datasets.load_dataset(self._dataset_type.value, 
                                     split=split, 
                                     streaming=False)
        logger.info(f"Dataset loaded with {len(self._dataset)} records.")
    
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

    def _get_id(self, index: int) -> str:
        """ Returns the id of the record at the given index. """
        if self._dataset_type.name == HfDatasetType.BIRD.name:
            return str(index)
        elif self._dataset_type.name == HfDatasetType.TREE_OF_LIFE.name:
            try:
                return self._dataset[index]["__key__"]
            except Exception as e:
                logger.error(f"Error while fetching id for index: {index}")
                logger.error(e)
                return None
        
        raise ValueError(f"Dataset type: {self._dataset_type} not supported.")
    
    def _get_embedding(self, index: int) -> List[float]:
        """ Returns the embedding of the record at the given index. """
        if self._dataset_type.name == HfDatasetType.BIRD.name:
            img_key = "image"
        elif self._dataset_type.name == HfDatasetType.TREE_OF_LIFE.name:
            img_key = "jpg"
        else:
            raise ValueError(f"Dataset type: {self._dataset_type} not supported.")
        
        try:
            return self._classifier.create_image_features_for_image(
                self._dataset[index][img_key], 
                normalize=True).tolist()
        except Exception as e:
            logger.error(f"Error while fetching embedding for index: {index}")
            logger.error(e)
            return None
    
    def load_database(self, reset: bool = False):
        if reset: 
            logger.info("Resetting the database.")
            self._client.delete_collection(self._collection.name)
            self._init_collection()
        
        num_records = 0
        for iter in tqdm(range(10000)):
            for i in tqdm(range(len(self._dataset))):
                id = self._get_id(iter + i)
                embedding = self._get_embedding(i)
                if id is None or embedding is None:
                    logger.warning(f"Skipping record with index: {i}")
                    continue
                self._collection.add(embeddings=[embedding], ids=[id])
                num_records += 1

        logger.info(f"Database created with {len(num_records)} records.")

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
    parser.add_argument(
        "--split", 
        type=str, 
        default="train", 
        help="Split of the dataset to use.")

    args = parser.parse_args()
    dataset = args.dataset
    output_dir = args.output_dir
    split = args.split

    logger.info(f"Creating database for dataset: {dataset} with split: {split}")
    logger.info(f"Creating database for dataset: {dataset.value}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Resetting the database: {args.reset}")

    vdb = BioclipVectorDatabase(
        dataset_type=dataset, 
        collection_dir=output_dir, 
        split=split)
    vdb.load_database(reset=args.reset)

if __name__ == "__main__":
    main()