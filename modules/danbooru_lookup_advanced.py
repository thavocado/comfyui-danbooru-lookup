"""
Advanced Danbooru lookup node that supports multiple input modes:
- Original mode: Images + text tags (like the original script)
- Conditioning mode: ComfyUI CLIP conditioning (current implementation)
- Hybrid mode: Combine all available inputs
"""

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any

from .danbooru_lookup import DanbooruFAISSLookup, ensure_dependencies
from .model_manager import ModelManager
from .wd14_embeddings import WD14Embeddings
from .tag_embeddings import TagEmbeddings

class DanbooruFAISSLookupAdvanced(DanbooruFAISSLookup):
    """
    Advanced ComfyUI node that supports multiple input modes for Danbooru lookup.
    """
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.wd14_embeddings = WD14Embeddings(self.model_manager)
        self.tag_embeddings = TagEmbeddings(self.model_manager)
        self._check_available_modes()
    
    def _check_available_modes(self):
        """Check which modes are available based on dependencies."""
        deps = self.model_manager.check_dependencies()
        self.modes_available = {
            "conditioning": True,  # Always available
            "tags_and_images": (
                WD14Embeddings.is_available() or 
                TagEmbeddings.is_available()
            ),
            "hybrid": True  # Can work with partial inputs
        }
        
        # Log available modes
        for mode, available in self.modes_available.items():
            if available:
                logging.info(f"[Danbooru Advanced] Mode '{mode}' is available")
            else:
                logging.info(f"[Danbooru Advanced] Mode '{mode}' is not available (missing dependencies)")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["conditioning", "tags_and_images", "hybrid"], {
                    "default": "conditioning"
                }),
            },
            "optional": {
                # Image inputs
                "positive_image": ("IMAGE",),
                "negative_image": ("IMAGE",),
                
                # Tag inputs
                "positive_tags": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Tags to search for (comma-separated)"
                }),
                "negative_tags": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Tags to exclude (comma-separated)"
                }),
                
                # Conditioning inputs
                "positive_conditioning": ("CONDITIONING",),
                "negative_conditioning": ("CONDITIONING",),
                
                # Model selection
                "embedding_model": (["CLIP", "SigLIP", "from_conditioning"], {
                    "default": "from_conditioning"
                }),
                
                # Other parameters
                "selected_ratings": ("STRING", {
                    "default": "General,Sensitive",
                    "multiline": False
                }),
                "n_neighbours": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "display": "number"
                }),
                "api_username": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("danbooru_id", "all_ids", "similarity_scores")
    FUNCTION = "lookup_advanced"
    CATEGORY = "conditioning/danbooru"
    
    def _process_image_embeddings(self, positive_image, negative_image) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process images to embeddings using WD14."""
        pos_img_emb = None
        neg_img_emb = None
        
        if positive_image is not None and self.wd14_embeddings.is_available():
            pos_img_emb = self.wd14_embeddings.extract_embeddings(positive_image)
            if pos_img_emb is not None:
                # Normalize
                faiss = ensure_dependencies()[1]
                faiss.normalize_L2(pos_img_emb)
        
        if negative_image is not None and self.wd14_embeddings.is_available():
            neg_img_emb = self.wd14_embeddings.extract_embeddings(negative_image)
            if neg_img_emb is not None:
                # Normalize
                faiss = ensure_dependencies()[1]
                faiss.normalize_L2(neg_img_emb)
        
        return pos_img_emb, neg_img_emb
    
    def _process_tag_embeddings(self, positive_tags, negative_tags, model_variant) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process tags to embeddings using CLIP/SigLIP."""
        pos_tag_emb = None
        neg_tag_emb = None
        
        if positive_tags and self.tag_embeddings.is_available():
            pos_tag_emb = self.tag_embeddings.encode_tags(positive_tags, model_variant)
            if pos_tag_emb is not None:
                # Normalize
                faiss = ensure_dependencies()[1]
                faiss.normalize_L2(pos_tag_emb)
        
        if negative_tags and self.tag_embeddings.is_available():
            neg_tag_emb = self.tag_embeddings.encode_tags(negative_tags, model_variant)
            if neg_tag_emb is not None:
                # Normalize
                faiss = ensure_dependencies()[1]
                faiss.normalize_L2(neg_tag_emb)
        
        return pos_tag_emb, neg_tag_emb
    
    def _combine_embeddings_advanced(self, 
                                   pos_img_emb: Optional[np.ndarray],
                                   pos_tag_emb: Optional[np.ndarray],
                                   pos_cond_emb: Optional[np.ndarray],
                                   neg_img_emb: Optional[np.ndarray],
                                   neg_tag_emb: Optional[np.ndarray],
                                   neg_cond_emb: Optional[np.ndarray]) -> np.ndarray:
        """Combine all types of embeddings following the original logic."""
        np_lib, faiss_lib, _ = ensure_dependencies()
        
        # Initialize with zeros if needed
        embedding_dim = None
        if self.knn_index is not None:
            embedding_dim = self.knn_index.d
        else:
            # Try to get from any available embedding
            for emb in [pos_img_emb, pos_tag_emb, pos_cond_emb, neg_img_emb, neg_tag_emb, neg_cond_emb]:
                if emb is not None:
                    embedding_dim = emb.shape[1]
                    break
            if embedding_dim is None:
                embedding_dim = 1024  # Default
        
        # Initialize embeddings
        pos_combined = np_lib.zeros((1, embedding_dim), dtype=np.float32)
        neg_combined = np_lib.zeros((1, embedding_dim), dtype=np.float32)
        
        # Add positive embeddings
        if pos_img_emb is not None:
            pos_combined = pos_combined + pos_img_emb
        if pos_tag_emb is not None:
            pos_combined = pos_combined + pos_tag_emb
        if pos_cond_emb is not None:
            pos_combined = pos_combined + pos_cond_emb
        
        # Add negative embeddings
        if neg_img_emb is not None:
            neg_combined = neg_combined + neg_img_emb
        if neg_tag_emb is not None:
            neg_combined = neg_combined + neg_tag_emb
        if neg_cond_emb is not None:
            neg_combined = neg_combined + neg_cond_emb
        
        # Normalize
        faiss_lib.normalize_L2(pos_combined)
        faiss_lib.normalize_L2(neg_combined)
        
        # Combine: positive - negative
        result = pos_combined - neg_combined
        faiss_lib.normalize_L2(result)
        
        return result
    
    def lookup_advanced(self, mode="conditioning", 
                       positive_image=None, negative_image=None,
                       positive_tags="", negative_tags="",
                       positive_conditioning=None, negative_conditioning=None,
                       embedding_model="from_conditioning",
                       selected_ratings="General,Sensitive", n_neighbours=5,
                       api_username="", api_key=""):
        """
        Perform advanced FAISS lookup with multiple input modes.
        Returns the top matching Danbooru post ID, all IDs, and similarity scores.
        """
        try:
            # Ensure dependencies are available
            ensure_dependencies()
            
            # Check if mode is available
            if not self.modes_available.get(mode, False):
                error_msg = f"Mode '{mode}' is not available. Missing dependencies."
                logging.error(f"[Danbooru Advanced] {error_msg}")
                return ("ERROR: " + error_msg, "", "")
            
            # Load index if not already loaded
            self._load_index()
            
            # Process inputs based on mode
            pos_img_emb = neg_img_emb = None
            pos_tag_emb = neg_tag_emb = None
            pos_cond_emb = neg_cond_emb = None
            
            if mode == "tags_and_images":
                # Original mode - use images and tags
                if positive_image is not None or negative_image is not None:
                    pos_img_emb, neg_img_emb = self._process_image_embeddings(positive_image, negative_image)
                
                if positive_tags or negative_tags:
                    model_variant = embedding_model if embedding_model != "from_conditioning" else "CLIP"
                    pos_tag_emb, neg_tag_emb = self._process_tag_embeddings(positive_tags, negative_tags, model_variant)
                
            elif mode == "conditioning":
                # Current mode - use conditioning
                pos_cond_emb = self._extract_embeddings_from_conditioning(positive_conditioning)
                neg_cond_emb = self._extract_embeddings_from_conditioning(negative_conditioning)
                
            elif mode == "hybrid":
                # New mode - use all available inputs
                if positive_image is not None or negative_image is not None:
                    pos_img_emb, neg_img_emb = self._process_image_embeddings(positive_image, negative_image)
                
                if positive_tags or negative_tags:
                    model_variant = embedding_model if embedding_model != "from_conditioning" else "CLIP"
                    pos_tag_emb, neg_tag_emb = self._process_tag_embeddings(positive_tags, negative_tags, model_variant)
                
                if positive_conditioning is not None or negative_conditioning is not None:
                    pos_cond_emb = self._extract_embeddings_from_conditioning(positive_conditioning)
                    neg_cond_emb = self._extract_embeddings_from_conditioning(negative_conditioning)
            
            # Combine all embeddings
            embeddings = self._combine_embeddings_advanced(
                pos_img_emb, pos_tag_emb, pos_cond_emb,
                neg_img_emb, neg_tag_emb, neg_cond_emb
            )
            
            # Parse ratings
            ratings_list = [r.strip() for r in selected_ratings.split(",")]
            
            # Perform search
            dists, indexes = self.knn_index.search(embeddings, k=n_neighbours)
            neighbours_ids = self.images_ids[indexes][0]
            neighbours_ids = [int(x) for x in neighbours_ids]
            
            # Format results
            all_ids = []
            all_scores = []
            valid_id = None
            
            for idx, (image_id, dist) in enumerate(zip(neighbours_ids, dists[0])):
                all_ids.append(str(image_id))
                all_scores.append(float(dist))
                
                # Check if this ID is valid based on ratings
                if valid_id is None:
                    url = self.danbooru_id_to_url(image_id, ratings_list, api_username, api_key)
                    if url is not None:
                        valid_id = str(image_id)
            
            # If no valid ID found, use the first one
            if valid_id is None and all_ids:
                valid_id = all_ids[0]
            
            # Format outputs
            top_id = valid_id if valid_id else ""
            all_ids_str = ",".join(all_ids)
            scores_str = ",".join([f"{s:.4f}" for s in all_scores])
            
            return (top_id, all_ids_str, scores_str)
            
        except ImportError as e:
            error_msg = f"Missing dependency: {e}"
            logging.error(f"[Danbooru Advanced] {error_msg}")
            return (f"ERROR: {error_msg}", "", "")
        except Exception as e:
            error_msg = f"Error during lookup: {e}"
            logging.error(f"[Danbooru Advanced] {error_msg}")
            return (f"ERROR: {error_msg}", "", "")


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "DanbooruFAISSLookupAdvanced": DanbooruFAISSLookupAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DanbooruFAISSLookupAdvanced": "Danbooru FAISS Lookup (Advanced)"
}