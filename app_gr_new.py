"""BioCLIP - Image Search Application.

A Gradio web interface for BioCLIP Vector DB, connecting to:
1. Neighborhood Server (Vector Search)
2. Image Server (Image Retrieval)

Usage:
    python app_gr_new.py \
        --neighborhood-server http://localhost:5001 \
        --image-server http://localhost:5002 \
        --host 0.0.0.0 \
        --port 7860 \
        --model hf-hub:imageomics/bioclip-2 \
        --disable-export
        
"""

import argparse
import io
import logging
import tempfile
import zipfile
import base64
import requests
from datetime import datetime
from typing import List, Optional, Dict

import gradio as gr
import open_clip
import torch
from PIL import Image

from src.app.server.image import embed_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

css = '''
.custom-gallery { 
    height: 640px !important; 
    overflow-y: auto !important; 
}
'''

class AppConfig:
    """Application configuration."""
    def __init__(
        self,
        neighborhood_server_url: str,
        image_server_url: str,
        model_name: str,
        device: Optional[str] = None,
        enable_export: bool = True
    ):
        self.neighborhood_server_url = neighborhood_server_url
        self.image_server_url = image_server_url
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_export = enable_export
        self.current_results: List[Image.Image] = []
        self.current_metadata: List[Dict] = []


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
        self._initialize()
    
    def _initialize(self):
        """Initialize models."""
        logger.info(f"Initializing model {self.config.model_name} on device: {self.config.device}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.model_name,
            device=self.config.device
        )
        logger.info("Initialization complete")

    def _search_vectors(self, query_vector: List[float], top_n: int, nprobe: int) -> List[Dict]:
        """Calls the Neighborhood Server to find nearest neighbors."""
        url = f"{self.config.neighborhood_server_url}/search"
        payload = {
            "query_vector": query_vector,
            "top_n": top_n,
            "nprobe": nprobe
        }
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Handle different response formats if necessary
            # Assuming the server returns {"status": "success", "data": {0: [results]}}
            # or directly the results depending on implementation.
            # Based on previous context, it returns {"status": "success", "data": {0: [...]}}
            
            if "data" in data:
                # The monolithic server returns a dict where keys are query indices (strings)
                # We sent 1 query, so we want the results for index "0"
                results_map = data["data"]
                if "0" in results_map:
                    return results_map["0"]
                elif 0 in results_map:
                    return results_map[0]
                else:
                    # Fallback if structure is different (e.g. list of lists)
                    return list(results_map.values())[0]
            return []
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    def _retrieve_images(self, uuids: List[str]) -> Dict[str, Image.Image]:
        """Calls the Image Server to retrieve images by UUID."""
        url = f"{self.config.image_server_url}/images"
        payload = {"uuids": uuids}
        
        images_dict = {}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            
            if "images" in data:
                for uuid, b64_str in data["images"].items():
                    if b64_str:
                        try:
                            img_data = base64.b64decode(b64_str)
                            img = Image.open(io.BytesIO(img_data)).convert("RGB")
                            images_dict[uuid] = img
                        except Exception as e:
                            logger.error(f"Failed to decode image for {uuid}: {e}")
                            images_dict[uuid] = None # Or placeholder
                    else:
                        images_dict[uuid] = None
            
            return images_dict

        except Exception as e:
            logger.error(f"Image retrieval failed: {e}")
            raise

    def search(
        self,
        img: Optional[Image.Image],
        top_n: int,
        nprobe: int
    ) -> tuple:
        """Perform image search.
        
        Args:
            img: Query image (PIL Image)
            top_n: Number of top results to return
            nprobe: Number of clusters to probe in the search
            
        Returns:
            Tuple of (gallery images, tree summary)
        """
        # Handle case when image is deleted/cleared
        if img is None:
            self.config.current_results = []
            self.config.current_metadata = []
            return [], "No results."
        
        try:
            # 1. Embed the query image
            logger.info("Embedding query image...")
            img_embedded = embed_image(img, self.model, self.preprocess)
            
            # 2. Search for similar vectors
            logger.info(f"Searching vectors (top_n={top_n}, nprobe={nprobe})...")
            search_results = self._search_vectors(
                query_vector=img_embedded,
                top_n=top_n,
                nprobe=nprobe
            )
            
            if not search_results:
                logger.warning("No results found from vector search.")
                return []

            # 3. Extract UUIDs
            # Assuming result items have a "uuid" field. 
            # If the field is named differently (e.g. "id" or "image_id"), adjust here.
            # Based on previous context, metadata usually contains "uuid".
            uuid_list = []
            for item in search_results:
                if "uuid" in item:
                    uuid_list.append(item["uuid"])
                elif "id" in item: # Fallback if id is the uuid
                     uuid_list.append(str(item["id"]))
            
            # 4. Retrieve images
            logger.info(f"Retrieving {len(uuid_list)} images...")
            images_map = self._retrieve_images(uuid_list)
            
            # 5. Order results matching the search order, preserving metadata
            ordered_images = []
            ordered_metadata = []
            for i, uuid in enumerate(uuid_list):
                if uuid in images_map and images_map[uuid] is not None:
                    ordered_images.append(images_map[uuid])
                    ordered_metadata.append(search_results[i])
            
            self.config.current_results = ordered_images
            self.config.current_metadata = ordered_metadata
            logger.info(f"Search completed. Found {len(self.config.current_results)} images")
            return self.config.current_results, self._generate_tree_summary()
            
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            gr.Warning(f"Search failed: {str(e)}")
            return [], "Search failed."
    
    def on_gallery_select(self, evt: gr.SelectData) -> tuple:
        """Handle gallery image selection to display metadata.
        
        Args:
            evt: Gradio SelectData event containing the selected index
            
        Returns:
            Tuple of (selected_image, metadata_markdown)
        """
        if not self.config.current_results or evt.index >= len(self.config.current_results):
            return None, "*No image selected*"
        
        selected_image = self.config.current_results[evt.index]
        
        if evt.index < len(self.config.current_metadata):
            meta = self.config.current_metadata[evt.index]
            metadata_md = self._format_metadata(meta, evt.index + 1)
        else:
            metadata_md = "*No metadata available*"
        
        return selected_image, metadata_md
    
    def _format_metadata(self, meta: Dict, rank: int) -> str:
        """Format metadata dictionary as Markdown for display.
        
        Args:
            meta: Metadata dictionary from search results
            rank: The result rank (1-indexed)
            
        Returns:
            Formatted Markdown string
        """
        common_name = meta.get("common_name") or "Common Name Unknown"
        scientific_name = meta.get("scientific_name") or meta.get("species") or "Scientific Name Unknown"
        distance = meta.get("distance", 0)
        
        # Taxonomy as single line with header
        taxonomy_parts = []
        for key in ["kingdom", "phylum", "class", "order", "family", "genus"]:
            value = meta.get(key)
            taxonomy_parts.append(value if value else "-")
        taxonomy_str = " > ".join(taxonomy_parts)
        taxonomy_header = "Kingdom > Phylum > Class > Order > Family > Genus"
        
        # Source with GBIF link if applicable
        source = meta.get("source_dataset", "Unknown")
        source_id = meta.get("source_id", "")
        if source and source.lower() == "gbif" and source_id:
            source_display = f"[GBIF](https://gbif.org/occurrence/{source_id})"
        else:
            source_display = source or "Unknown"
        
        publisher = meta.get("publisher", "Unknown")
        img_type = meta.get("img_type", "Unknown")
        identifier = meta.get("identifier", "")
        url_link = f"[View Original]({identifier})" if identifier else ""
        
        md = f"""**#{rank} {common_name}**  
*{scientific_name}*  
**Distance:** {distance:.4f}

**Taxonomy:** {taxonomy_header}  
{taxonomy_str}

**Source:** {source_display}  
**Type:** {img_type}  
**Publisher:** {publisher}  
{url_link}"""
        return md.strip()
    
    def _generate_tree_summary(self) -> str:
        """Generate a taxonomic tree summary of search results.
        
        Returns:
            Tree-formatted string showing taxonomy frequency counts
        """
        if not self.config.current_metadata:
            return "No results to summarize."
        
        from collections import defaultdict
        
        # Build nested counts: kingdom > phylum > class > order > family
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))
        
        for meta in self.config.current_metadata:
            kingdom = meta.get("kingdom") or "Unknown"
            phylum = meta.get("phylum") or "Unknown"
            cls = meta.get("class") or "Unknown"
            order = meta.get("order") or "Unknown"
            family = meta.get("family") or "Unknown"
            tree[kingdom][phylum][cls][order][family] += 1
        
        # Format as tree
        lines = [f"Search Results: {len(self.config.current_metadata)} images", ""]
        
        for kingdom, phyla in sorted(tree.items()):
            k_count = sum(sum(sum(sum(f.values()) for f in o.values()) for o in c.values()) for c in phyla.values())
            lines.append(f"├── {kingdom} ({k_count})")
            
            phyla_list = list(sorted(phyla.items()))
            for p_idx, (phylum, classes) in enumerate(phyla_list):
                p_count = sum(sum(sum(f.values()) for f in o.values()) for o in classes.values())
                p_last = p_idx == len(phyla_list) - 1
                p_prefix = "│   └── " if p_last else "│   ├── "
                p_cont = "│       " if not p_last else "        "
                lines.append(f"{p_prefix}{phylum} ({p_count})")
                
                classes_list = list(sorted(classes.items()))
                for c_idx, (cls, orders) in enumerate(classes_list):
                    c_count = sum(sum(f.values()) for f in orders.values())
                    c_last = c_idx == len(classes_list) - 1
                    c_prefix = f"{p_cont}└── " if c_last else f"{p_cont}├── "
                    c_cont = f"{p_cont}    " if c_last else f"{p_cont}│   "
                    lines.append(f"{c_prefix}{cls} ({c_count})")
                    
                    orders_list = list(sorted(orders.items()))
                    for o_idx, (order, families) in enumerate(orders_list):
                        o_count = sum(families.values())
                        o_last = o_idx == len(orders_list) - 1
                        o_prefix = f"{c_cont}└── " if o_last else f"{c_cont}├── "
                        o_cont = f"{c_cont}    " if o_last else f"{c_cont}│   "
                        lines.append(f"{o_prefix}{order} ({o_count})")
                        
                        families_list = list(sorted(families.items()))
                        for f_idx, (family, count) in enumerate(families_list):
                            f_last = f_idx == len(families_list) - 1
                            f_prefix = f"{o_cont}└── " if f_last else f"{o_cont}├── "
                            lines.append(f"{f_prefix}{family} ({count})")
        
        return "\n".join(lines)

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
        with gr.Blocks(title="BioCLIP Image Search", css=css) as demo:
            gr.Markdown("# BioCLIP - Image Search")
            
            with gr.Row():
                # Left panel: Input controls
                with gr.Column(scale=1, min_width=280):
                    img = gr.Image(type="pil", label="Upload Image", height=300)
                    
                    nprobe = gr.Slider(
                        1, 128, value=16, step=1, 
                        label="Search Depth (nprobe)",
                        info="Number of cluster partitions to search.\nHigher values = more accurate but slower."
                    )
                    
                    top_n = gr.Slider(
                        1, 128, value=8, step=1, 
                        label="Top N Results",
                        info="Number of nearest neighbors to return."
                    )
        
                    run = gr.Button("Run", variant="primary")
                    export_btn = gr.Button(
                        "Export Results", 
                        variant="secondary",
                        visible=self.config.enable_export
                    )
                    download_file = gr.File(
                        label="Export", 
                        visible=self.config.enable_export
                    )
                
                # Middle panel: Gallery
                with gr.Column(scale=2):
                    gallery = gr.Gallery(
                        label="Search Output Gallery",
                        columns=4,
                        height=640,
                        elem_classes="custom-gallery"
                    )
                
                # Right panel: Selected image details with tabs
                with gr.Column(scale=1, min_width=300):
                    with gr.Tabs():
                        with gr.TabItem("Selected"):
                            selected_image = gr.Image(
                                label="Selected Image",
                                height=280,
                                show_label=False
                            )
                            metadata_display = gr.Markdown(
                                value="*Click an image to see details*"
                            )
                        with gr.TabItem("Summary"):
                            tree_summary = gr.Code(
                                label="Taxonomy Tree",
                                language=None,
                                lines=25,
                                value="Run a search to see summary."
                            )
            
            # Event handlers
            run.click(
                self.search,
                inputs=[img, top_n, nprobe],
                outputs=[gallery, tree_summary]
            )
            
            gallery.select(
                self.on_gallery_select,
                inputs=[],
                outputs=[selected_image, metadata_display]
            )
            
            if self.config.enable_export:
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
        "--neighborhood-server",
        type=str,
        default="http://localhost:5001",
        help="URL of the Neighborhood (Vector Search) Server (default: http://localhost:5001)"
    )
    parser.add_argument(
        "--image-server",
        type=str,
        default="http://localhost:5002",
        help="URL of the Image Retrieval Server (default: http://localhost:5002)"
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
        "--model",
        type=str,
        default="hf-hub:imageomics/bioclip-2",
        help="Model to embed the image (default: hf-hub:imageomics/bioclip-2). Make sure the model is compatible with OpenCLIP. Make sure the model is consistent with the one used to build the FAISS index."
    )
    parser.add_argument(
        "--disable-export",
        action="store_true",
        help="Disable export functionality (hides export button and download file)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # Create configuration
    config = AppConfig(
        neighborhood_server_url=args.neighborhood_server,
        image_server_url=args.image_server,
        model_name=args.model,
        enable_export=not args.disable_export
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
