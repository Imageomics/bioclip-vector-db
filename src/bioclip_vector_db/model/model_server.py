"""
A Flask server to handle image embedding and prediction requests using BioCLIP.

Usage:
    python src/bioclip_vector_db/model/model_server.py \
        --device cuda \
        --port 5002

Endpoints:
    POST /embed - Generate embeddings for images
    POST /predict - Predict species/classes for images
    GET /health - Health check endpoint
"""

import logging
import argparse
import time
import functools
import base64
import io
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

import numpy as np
import PIL.Image
from flask import Flask, request, jsonify


_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
logger = logging.getLogger(__name__)


def timer(func):
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished '{func.__name__}' in {run_time:.4f} secs")
        return value
    return wrapper


class BaseModelService(ABC):
    """
    Abstract base class for model services.
    
    Provides a common interface for embedding generation and prediction.
    """

    @abstractmethod
    def embed(
        self, 
        images: List[PIL.Image.Image], 
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of images.
        
        Args:
            images: List of PIL Image objects.
            normalize: Whether to L2-normalize the embeddings.
            
        Returns:
            numpy array of shape (N, embedding_dim) containing image embeddings.
        """
        pass

    @abstractmethod
    def predict(
        self,
        images: List[PIL.Image.Image],
        labels: Optional[List[str]] = None,
        rank: Optional[str] = None,
        k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Predict classes/species for a list of images.
        
        Args:
            images: List of PIL Image objects.
            labels: Optional custom labels for classification.
            rank: Taxonomic rank for TreeOfLife prediction (e.g., 'species', 'genus').
            k: Number of top predictions to return.
            
        Returns:
            List of prediction results for each image.
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model metadata.
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the model is loaded and ready for inference.
        
        Returns:
            True if model is ready, False otherwise.
        """
        pass


class BioCLIPModelService(BaseModelService):
    """
    BioCLIP model service using pybioclip for embedding and prediction.
    
    Supports:
        - Image embedding generation
        - TreeOfLife taxonomic classification
        - Custom label classification
    """

    def __init__(
        self,
        device: str = "cpu",
        model_str: Optional[str] = None
    ):
        """
        Initialize the BioCLIP model service.
        
        Args:
            device: Device to run inference on ('cpu', 'cuda', 'mps').
            model_str: Model identifier (defaults to BioCLIP v2).
        """
        if model_str is None:
            model_str = "hf-hub:imageomics/bioclip-2"
        self.device = device
        self.model_str = model_str
        self._tol_classifier = None
        self._custom_classifier_cache: Dict[str, Any] = {}
        
        self._load_model()

    def _load_model(self):
        """Load the BioCLIP model."""
        from bioclip import TreeOfLifeClassifier, Rank, CustomLabelsClassifier
        
        logger.info(f"Loading BioCLIP model on device: {self.device}")
        
        kwargs = {"device": self.device}
        if self.model_str:
            kwargs["model_str"] = self.model_str
            
        self._tol_classifier = TreeOfLifeClassifier(**kwargs)
        self._Rank = Rank
        self._CustomLabelsClassifier = CustomLabelsClassifier
        
        logger.info(f"BioCLIP model loaded successfully: {self._tol_classifier.model_str}")

    def _get_custom_classifier(self, labels: List[str]):
        """
        Get or create a custom labels classifier.
        
        Args:
            labels: List of class labels.
            
        Returns:
            CustomLabelsClassifier instance.
        """
        # Use tuple of labels as cache key
        cache_key = tuple(sorted(labels))
        
        if cache_key not in self._custom_classifier_cache:
            logger.info(f"Creating custom classifier for {len(labels)} labels")
            kwargs = {"device": self.device, "cls_ary": labels}
            if self.model_str:
                kwargs["model_str"] = self.model_str
            self._custom_classifier_cache[cache_key] = self._CustomLabelsClassifier(**kwargs)
            
        return self._custom_classifier_cache[cache_key]

    @timer
    def embed(
        self, 
        images: List[PIL.Image.Image], 
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for images using BioCLIP.
        
        Args:
            images: List of PIL Image objects.
            normalize: Whether to L2-normalize the embeddings.
            
        Returns:
            numpy array of shape (N, embedding_dim).
        """
        # Ensure RGB format
        rgb_images = [img.convert("RGB") for img in images]
        
        # Use the TreeOfLife classifier's create_image_features method
        features = self._tol_classifier.create_image_features(rgb_images, normalize=normalize)
        
        return features.cpu().numpy()

    @timer
    def predict(
        self,
        images: List[PIL.Image.Image],
        labels: Optional[List[str]] = None,
        rank: Optional[str] = None,
        k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Predict classes for images.
        
        Args:
            images: List of PIL Image objects.
            labels: Optional custom labels. If provided, uses CustomLabelsClassifier.
            rank: Taxonomic rank for TreeOfLife ('kingdom', 'phylum', 'class', 
                  'order', 'family', 'genus', 'species'). Used when labels is None.
            k: Number of top predictions to return.
            
        Returns:
            List of prediction results for each image, where each result is a list
            of dictionaries containing prediction details.
        """
        # Ensure RGB format
        rgb_images = [img.convert("RGB") for img in images]
        
        if labels:
            # Custom labels classification
            classifier = self._get_custom_classifier(labels)
            predictions = classifier.predict(rgb_images, k=k)
        else:
            # TreeOfLife taxonomic classification
            rank_enum = self._Rank[rank.upper()] if rank else self._Rank.SPECIES
            predictions = self._tol_classifier.predict(rgb_images, rank=rank_enum, k=k)
        
        # Group predictions by image
        # Predictions come as flat list, group by image index
        results = []
        predictions_per_image = k
        
        for i in range(len(images)):
            start_idx = i * predictions_per_image
            end_idx = start_idx + predictions_per_image
            image_predictions = predictions[start_idx:end_idx]
            results.append(image_predictions)
            
        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get BioCLIP model information."""
        return {
            "model_name": "BioCLIP",
            "model_str": self._tol_classifier.model_str,
            "device": self.device,
            "embedding_dim": 768,  # BioCLIP uses ViT-B/16 with 768-dim embeddings
        }

    def is_ready(self) -> bool:
        """Check if model is loaded."""
        return self._tol_classifier is not None


class ModelServer:
    """
    A Flask server class to handle embedding and prediction requests.
    """

    def __init__(self, service: BaseModelService):
        """
        Initialize the model server.
        
        Args:
            service: A model service instance implementing BaseModelService.
        """
        self._app = Flask(__name__)
        self._service = service
        self._register_routes()

    def _register_routes(self):
        """Register Flask routes."""
        self._app.add_url_rule(
            "/embed", "embed", self.handle_embed, methods=["POST"]
        )
        self._app.add_url_rule(
            "/predict", "predict", self.handle_predict, methods=["POST"]
        )
        self._app.add_url_rule(
            "/health", "health", self.handle_health, methods=["GET"]
        )

    def _success_response(self, data, status_code=200):
        """Create a success response."""
        return jsonify({"status": "success", "data": data}), status_code

    def _error_response(self, message, status_code=400):
        """Create an error response."""
        return (
            jsonify(
                {"status": "error", "error": {"code": status_code, "message": message}}
            ),
            status_code,
        )

    def _parse_images(self, data: Dict) -> List[PIL.Image.Image]:
        """
        Parse images from request data.
        
        Supports:
            - Base64 encoded images in 'images' field (list of base64 strings)
            - Image URLs in 'image_urls' field (list of URLs)
            
        Args:
            data: Request JSON data.
            
        Returns:
            List of PIL Image objects.
        """
        import requests as http_requests
        
        images = []
        
        # Parse base64 encoded images
        if "images" in data:
            for img_b64 in data["images"]:
                img_bytes = base64.b64decode(img_b64)
                img = PIL.Image.open(io.BytesIO(img_bytes))
                images.append(img)
                
        # Parse image URLs
        elif "image_urls" in data:
            for url in data["image_urls"]:
                response = http_requests.get(url, timeout=30)
                response.raise_for_status()
                img = PIL.Image.open(io.BytesIO(response.content))
                images.append(img)
        
        return images

    def handle_health(self):
        """Handle health check requests."""
        if self._service.is_ready():
            health_data = {
                "status": "ready",
                **self._service.get_model_info()
            }
            return self._success_response(health_data)
        return self._error_response("Model not loaded", 503)

    def handle_embed(self):
        """
        Handle embedding requests.
        
        Request JSON format:
        {
            "images": ["<base64_encoded_image>", ...],  # OR
            "image_urls": ["http://...", ...],
            "normalize": true  # optional, default true
        }
        
        Response format:
        {
            "status": "success",
            "data": {
                "embeddings": [[...], [...], ...],
                "embedding_dim": 768
            }
        }
        """
        data = request.get_json()

        if not data:
            return self._error_response("Missing JSON body", 400)
        
        if "images" not in data and "image_urls" not in data:
            return self._error_response(
                "Missing 'images' (base64) or 'image_urls' in JSON body", 400
            )

        try:
            images = self._parse_images(data)
            if not images:
                return self._error_response("No valid images provided", 400)
            
            normalize = data.get("normalize", True)
            
            embeddings = self._service.embed(images, normalize=normalize)
            
            return self._success_response({
                "embeddings": embeddings.tolist(),
                "embedding_dim": embeddings.shape[1],
                "num_images": len(images)
            })
            
        except Exception as e:
            logger.exception("Error during embedding")
            return self._error_response(f"Embedding error: {str(e)}", 500)

    def handle_predict(self):
        """
        Handle prediction requests.
        
        Request JSON format:
        {
            "images": ["<base64_encoded_image>", ...],  # OR
            "image_urls": ["http://...", ...],
            "labels": ["cat", "dog", ...],  # optional, for custom classification
            "rank": "species",  # optional, for TreeOfLife (default: species)
            "k": 5  # optional, number of top predictions (default: 5)
        }
        
        Response format:
        {
            "status": "success",
            "data": {
                "predictions": [
                    [{"classification": "...", "score": 0.95}, ...],
                    ...
                ],
                "num_images": 1
            }
        }
        """
        data = request.get_json()

        if not data:
            return self._error_response("Missing JSON body", 400)
        
        if "images" not in data and "image_urls" not in data:
            return self._error_response(
                "Missing 'images' (base64) or 'image_urls' in JSON body", 400
            )

        try:
            images = self._parse_images(data)
            if not images:
                return self._error_response("No valid images provided", 400)
            
            labels = data.get("labels")
            rank = data.get("rank", "species")
            k = data.get("k", 5)
            
            predictions = self._service.predict(
                images, 
                labels=labels,
                rank=rank,
                k=k
            )
            
            return self._success_response({
                "predictions": predictions,
                "num_images": len(images)
            })
            
        except Exception as e:
            logger.exception("Error during prediction")
            return self._error_response(f"Prediction error: {str(e)}", 500)

    def run(self, host: str, port: int):
        """Run the Flask server."""
        self._app.run(host=host, port=port)


def create_app(
    device: str = "cpu",
    model_str: Optional[str] = None
) -> Flask:
    """
    Create a Flask app with BioCLIP model service.
    
    Args:
        device: Device to run inference on.
        model_str: Model identifier.
        
    Returns:
        Flask application instance.
    """
    service = BioCLIPModelService(
        device=device,
        model_str=model_str
    )
    server = ModelServer(service)
    return server._app


def __main__():
    parser = argparse.ArgumentParser(description="BioCLIP Model Server")
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        help="Device to run inference on (cpu, cuda)"
    )
    parser.add_argument(
        "--model-str",
        type=str,
        default=None,
        help="Model identifier (e.g., 'hf-hub:imageomics/bioclip-2')"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=5002, 
        help="Port to run the server on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )

    args = parser.parse_args()
    
    app = create_app(
        device=args.device,
        model_str=args.model_str
    )

    logger.info(f"Starting BioCLIP Model Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    __main__()
