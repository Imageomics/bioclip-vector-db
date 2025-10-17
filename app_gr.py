"""BioCLIP Vector Database - Image Search Application.

A Gradio web interface for BioCLIP Vector DB
"""

import argparse
import io
import logging
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import gradio as gr
import open_clip
import pyarrow.dataset as ds
import torch
from PIL import Image

from src.bioclip_vector_db.client.nearest_neighbor_client import NearestNeighborClient
from src.app.server.image import embed_image, retrieve_images_hdf5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AppConfig:
    """Application configuration."""
    def __init__(
        self,
        server_url: str,
        lookup_table_path: str,
        model_name: str,
        device: Optional[str] = None
    ):
        self.server_url = server_url
        self.lookup_table_path = lookup_table_path
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.current_results: List[Image.Image] = []


class BioCLIPSearchApp:
    """BioCLIP image search application."""
    
    def __init__(self, config: AppConfig):
        """Initialize the application with configuration.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.model = None
        self.preprocess = None
        self.client = None
        self.lookup_tbl = None
        self._initialize()
    
    def _initialize(self):
        """Initialize models, client, and lookup table."""
        logger.info(f"Initializing model {self.config.model_name} on device: {self.config.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.model_name,
            device=self.config.device
        )
        
        logger.info(f"Initializing client with server: {self.config.server_url}")
        self.client = NearestNeighborClient([self.config.server_url])
        
        logger.info(f"Loading lookup table from: {self.config.lookup_table_path}")
        dataset = ds.dataset(self.config.lookup_table_path, format="parquet")
        self.lookup_tbl = dataset.to_table(use_threads=True, batch_readahead=4)
        
        logger.info("Initialization complete")
    
    def search(
        self,
        img: Optional[Image.Image],
        top_n: int,
        nprobe: int
    ) -> List[Image.Image]:
        """Perform image search.
        
        Args:
            img: Query image (PIL Image)
            top_n: Number of top results to return
            nprobe: Number of clusters to probe in the search
            
        Returns:
            List of similar images
        """
        # Handle case when image is deleted/cleared
        if img is None:
            self.config.current_results = []
            return []
        
        try:
            # Embed the query image
            img_embedded = embed_image(img, self.model, self.preprocess)
            
            # Search for similar images
            search_results = self.client.search(
                query_vector=img_embedded,
                top_n=top_n,
                nprobe=nprobe
            )
            
            # Retrieve images from results
            uuid_list = [item["id"] for item in search_results]
            images_dict, failed_dict = retrieve_images_hdf5(
                uuid_list,
                lookup_tbl=self.lookup_tbl
            )
            
            if failed_dict:
                logger.warning(
                    f"Failed to retrieve {len(failed_dict)} images: {failed_dict}"
                )
            
            self.config.current_results = list(images_dict.values())
            logger.info(f"Search completed. Found {len(self.config.current_results)} results")
            return self.config.current_results
            
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            gr.Warning(f"Search failed: {str(e)}")
            return []
    
    def export_results(self) -> Optional[str]:
        """Export current search results as a zip file.
        
        Returns:
            Path to the exported zip file, or None if no results
        """
        if not self.config.current_results:
            logger.warning("No results to export")
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create temporary file with proper cleanup
            temp_file = tempfile.NamedTemporaryFile(
                mode='wb',
                suffix='.zip',
                prefix=f'search_results_{timestamp}_',
                delete=False
            )
            
            with zipfile.ZipFile(temp_file, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for idx, img in enumerate(self.config.current_results, start=1):
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    zip_file.writestr(f"result_{idx}.png", img_buffer.getvalue())
            
            temp_file.close()
            logger.info(f"Exported {len(self.config.current_results)} results to {temp_file.name}")
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Error during export: {e}", exc_info=True)
            gr.Warning(f"Export failed: {str(e)}")
            return None
    
    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface.
        
        Returns:
            Configured Gradio Blocks interface
        """
        with gr.Blocks(title="BioCLIP Image Search") as demo:
            gr.Markdown("# BioCLIP Vector Database - Image Search")
            
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    img = gr.Image(type="pil", label="Upload Image", height=360)
                    top_n = gr.Slider(1, 40, value=8, step=1, label="Top N")
                    nprobe = gr.Slider(1, 128, value=16, step=1, label="Nprobe")
                    run = gr.Button("Run", variant="primary")
                    export_btn = gr.Button("Export Results", variant="secondary")
                    
                with gr.Column(scale=2):
                    gallery = gr.Gallery(
                        label="Search Output Gallery",
                        columns=5,
                        height=640
                    )
                    download_file = gr.File(label="Export", visible=True)
            
            # Event handlers
            run.click(
                self.search,
                inputs=[img, top_n, nprobe],
                outputs=[gallery]
            )
            img.change(
                self.search,
                inputs=[img, top_n, nprobe],
                outputs=[gallery]
            )
            export_btn.click(
                self.export_results,
                inputs=[],
                outputs=[download_file]
            )
        
        return demo


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="BioCLIP Vector Database - Image Search Application"
    )
    parser.add_argument(
        "--db-server-url",
        type=str,
        default="http://localhost:5001",
        help="Server URL for the nearest neighbor search (default: http://localhost:5001)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for the Gradio app server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the Gradio app server (default: 7860)"
    )
    parser.add_argument(
        "--lookup-table-path",
        type=str,
        default="/fs/scratch/PAS2136/TreeOfLife/lookup_tables/2024-05-01/hdf5/10M",
        help="Path to the lookup table directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="hf-hub:imageomics/bioclip",
        help="Model to embed the image (default: hf-hub:imageomics/bioclip)"
    )

    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Create configuration
    config = AppConfig(
        server_url=args.db_server_url,
        lookup_table_path=args.lookup_table_path,
        model_name=args.model
    )
    
    # Initialize and launch app
    app = BioCLIPSearchApp(config)
    demo = app.create_interface()
    
    logger.info(f"Launching app on {args.host}:{args.port}")
    demo.launch(
        server_name=args.host,
        server_port=args.port
    )


if __name__ == "__main__":
    main()
