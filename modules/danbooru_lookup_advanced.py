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
    
    # Class-level cache for dependency check results
    _deps_cache = None
    _modes_cache = None
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.wd14_embeddings = WD14Embeddings(self.model_manager)
        self.tag_embeddings = TagEmbeddings(self.model_manager)
        self._check_available_modes()
    
    def _check_available_modes(self):
        """Check which modes are available based on dependencies."""
        # Use cached results if available to avoid re-checking dependencies
        if DanbooruFAISSLookupAdvanced._modes_cache is not None:
            self.modes_available = DanbooruFAISSLookupAdvanced._modes_cache
            self.can_process_images = self.modes_available.get("_can_process_images", False)
            self.can_process_tags = self.modes_available.get("_can_process_tags", False)
            return
        
        try:
            deps = self.model_manager.check_dependencies()
            DanbooruFAISSLookupAdvanced._deps_cache = deps
        except Exception as e:
            # If dependency check fails due to PyTreeDef, use fallback
            logging.warning(f"[Danbooru Advanced] Dependency check error: {e}")
            if "PyTreeDef" in str(e):
                # Assume JAX/FLAX are available if we get PyTreeDef error
                deps = {
                    "jax": True,
                    "flax": True,
                    "dghs_imgutils": WD14Embeddings.is_available(),
                    "torch": True,  # Usually available in ComfyUI
                    "huggingface_hub": True
                }
            else:
                deps = {}
        
        # Check individual capabilities
        self.can_process_images = WD14Embeddings.is_available()
        self.can_process_tags = TagEmbeddings.is_available()
        
        self.modes_available = {
            "conditioning": True,  # Always available
            "tags_and_images": (self.can_process_images or self.can_process_tags),
            "hybrid": True,  # Can work with partial inputs
            "_can_process_images": self.can_process_images,  # Store for cache
            "_can_process_tags": self.can_process_tags  # Store for cache
        }
        
        # Cache the results
        DanbooruFAISSLookupAdvanced._modes_cache = self.modes_available
        
        # Log available modes and capabilities
        logging.info(f"[Danbooru Advanced] Image processing available: {self.can_process_images}")
        logging.info(f"[Danbooru Advanced] Tag processing available: {self.can_process_tags}")
        
        for mode, available in self.modes_available.items():
            if not mode.startswith("_") and available:
                logging.info(f"[Danbooru Advanced] Mode '{mode}' is available")
            elif not mode.startswith("_") and not available:
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
                "hf_token": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "HuggingFace token (optional)"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("danbooru_id", "all_ids", "similarity_scores")
    FUNCTION = "lookup_advanced"
    CATEGORY = "conditioning/danbooru"
    OUTPUT_NODE = False
    
    def _process_image_embeddings(self, positive_image, negative_image, hf_token=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process images to embeddings using WD14."""
        pos_img_emb = None
        neg_img_emb = None
        
        # Check if WD14 is available
        if (positive_image is not None or negative_image is not None) and not self.wd14_embeddings.is_available():
            logging.error("[Danbooru Advanced] Images provided but WD14 embeddings not available!")
            logging.error("[Danbooru Advanced] Please install dghs-imgutils:")
            logging.error("[Danbooru Advanced]   For GPU: pip install dghs-imgutils[gpu]")
            logging.error("[Danbooru Advanced]   For CPU: pip install dghs-imgutils")
            logging.error("[Danbooru Advanced] Or run the installer: python install.py")
            return None, None
        
        if positive_image is not None and self.wd14_embeddings.is_available():
            pos_img_emb = self.wd14_embeddings.extract_embeddings(positive_image, hf_token=hf_token)
            if pos_img_emb is not None:
                logging.info(f"[Danbooru Advanced] Positive image embedding shape: {pos_img_emb.shape}")
            else:
                logging.error("[Danbooru Advanced] Failed to extract positive image embeddings")
        
        if negative_image is not None and self.wd14_embeddings.is_available():
            neg_img_emb = self.wd14_embeddings.extract_embeddings(negative_image, hf_token=hf_token)
            if neg_img_emb is not None:
                logging.info(f"[Danbooru Advanced] Negative image embedding shape: {neg_img_emb.shape}")
            else:
                logging.error("[Danbooru Advanced] Failed to extract negative image embeddings")
        
        return pos_img_emb, neg_img_emb
    
    def _process_tag_embeddings(self, positive_tags, negative_tags, model_variant) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process tags to embeddings using CLIP/SigLIP."""
        pos_tag_emb = None
        neg_tag_emb = None
        
        try:
            if positive_tags and self.tag_embeddings.is_available():
                pos_tag_emb = self.tag_embeddings.encode_tags(positive_tags, model_variant)
                if pos_tag_emb is not None:
                    logging.info(f"[Danbooru Advanced] Positive tag embedding shape: {pos_tag_emb.shape}")
            
            if negative_tags and self.tag_embeddings.is_available():
                neg_tag_emb = self.tag_embeddings.encode_tags(negative_tags, model_variant)
                if neg_tag_emb is not None:
                    logging.info(f"[Danbooru Advanced] Negative tag embedding shape: {neg_tag_emb.shape}")
                    
        except (RuntimeError, ValueError) as e:
            # Re-raise with more context
            error_msg = f"Tag processing failed: {str(e)}"
            logging.error(f"[Danbooru Advanced] {error_msg}")
            raise RuntimeError(error_msg)
        
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
        
        # Process positive embeddings following original logic
        if pos_img_emb is not None:
            faiss_lib.normalize_L2(pos_img_emb)
            pos_combined = pos_combined + pos_img_emb
        if pos_tag_emb is not None:
            faiss_lib.normalize_L2(pos_tag_emb)
            pos_combined = pos_combined + pos_tag_emb
        if pos_cond_emb is not None:
            faiss_lib.normalize_L2(pos_cond_emb)
            pos_combined = pos_combined + pos_cond_emb
        
        # Process negative embeddings following original logic
        if neg_img_emb is not None:
            faiss_lib.normalize_L2(neg_img_emb)
            neg_combined = neg_combined + neg_img_emb
        if neg_tag_emb is not None:
            faiss_lib.normalize_L2(neg_tag_emb)
            neg_combined = neg_combined + neg_tag_emb
        if neg_cond_emb is not None:
            faiss_lib.normalize_L2(neg_cond_emb)
            neg_combined = neg_combined + neg_cond_emb
        
        # Normalize combined embeddings
        faiss_lib.normalize_L2(pos_combined)
        faiss_lib.normalize_L2(neg_combined)
        
        logging.info(f"[Danbooru Advanced] Pos combined shape: {pos_combined.shape}, Neg combined shape: {neg_combined.shape}")
        
        # Combine: positive - negative
        result = pos_combined - neg_combined
        faiss_lib.normalize_L2(result)
        
        logging.info(f"[Danbooru Advanced] Final embedding shape: {result.shape}, norm: {np_lib.linalg.norm(result)}")
        
        return result
    
    def lookup_advanced(self, mode="conditioning", 
                       positive_image=None, negative_image=None,
                       positive_tags="", negative_tags="",
                       positive_conditioning=None, negative_conditioning=None,
                       embedding_model="from_conditioning",
                       selected_ratings="General,Sensitive", n_neighbours=5,
                       api_username="", api_key="", hf_token=""):
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
                return {"ui": {"text": ["ERROR: " + error_msg, "", ""]}, "result": ("ERROR: " + error_msg, "", "")}
            
            # Load index if not already loaded
            self._load_index()
            
            # Process inputs based on mode
            pos_img_emb = neg_img_emb = None
            pos_tag_emb = neg_tag_emb = None
            pos_cond_emb = neg_cond_emb = None
            
            if mode == "tags_and_images":
                # Original mode - use images and tags
                if positive_image is not None or negative_image is not None:
                    pos_img_emb, neg_img_emb = self._process_image_embeddings(positive_image, negative_image, hf_token)
                
                if positive_tags or negative_tags:
                    model_variant = embedding_model if embedding_model != "from_conditioning" else "CLIP"
                    try:
                        pos_tag_emb, neg_tag_emb = self._process_tag_embeddings(positive_tags, negative_tags, model_variant)
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "text_projection" in error_msg:
                            error_msg = "Tag encoding failed. The CLIP/SigLIP models may need to be re-downloaded. Try deleting the models folder and restarting."
                        return {"ui": {"text": [f"ERROR: {error_msg}", "", ""]}, "result": (f"ERROR: {error_msg}", "", "")}
                
            elif mode == "conditioning":
                # Current mode - use conditioning
                pos_cond_emb = self._extract_embeddings_from_conditioning(positive_conditioning)
                neg_cond_emb = self._extract_embeddings_from_conditioning(negative_conditioning)
                
            elif mode == "hybrid":
                # New mode - use all available inputs
                if positive_image is not None or negative_image is not None:
                    pos_img_emb, neg_img_emb = self._process_image_embeddings(positive_image, negative_image, hf_token)
                
                if positive_tags or negative_tags:
                    model_variant = embedding_model if embedding_model != "from_conditioning" else "CLIP"
                    try:
                        pos_tag_emb, neg_tag_emb = self._process_tag_embeddings(positive_tags, negative_tags, model_variant)
                    except RuntimeError as e:
                        error_msg = str(e)
                        if "text_projection" in error_msg:
                            error_msg = "Tag encoding failed. The CLIP/SigLIP models may need to be re-downloaded. Try deleting the models folder and restarting."
                        return {"ui": {"text": [f"ERROR: {error_msg}", "", ""]}, "result": (f"ERROR: {error_msg}", "", "")}
                
                if positive_conditioning is not None or negative_conditioning is not None:
                    pos_cond_emb = self._extract_embeddings_from_conditioning(positive_conditioning)
                    neg_cond_emb = self._extract_embeddings_from_conditioning(negative_conditioning)
            
            # Check if we have any inputs
            if (pos_img_emb is None and pos_tag_emb is None and pos_cond_emb is None and
                neg_img_emb is None and neg_tag_emb is None and neg_cond_emb is None):
                
                # Provide specific error based on what was attempted
                if mode == "tags_and_images":
                    if (positive_image is not None or negative_image is not None) and not self.can_process_images:
                        error_msg = "Images provided but dghs-imgutils not installed. Install with: pip install dghs-imgutils[gpu]"
                    elif (positive_tags or negative_tags) and not self.can_process_tags:
                        error_msg = "Tags provided but CLIP/SigLIP models not available. Check JAX/FLAX installation."
                    elif (positive_image is not None or negative_image is not None):
                        # Images were provided and dghs-imgutils is installed, but embeddings failed
                        error_msg = "Failed to process images. Check console for details (may be HF auth issue)."
                    else:
                        error_msg = f"No valid inputs provided for mode '{mode}'. Please provide images or tags."
                else:
                    error_msg = f"No valid inputs provided for mode '{mode}'. Please provide at least one input."
                
                logging.error(f"[Danbooru Advanced] {error_msg}")
                return {"ui": {"text": ["ERROR: " + error_msg, "", ""]}, "result": ("ERROR: " + error_msg, "", "")}
            
            # Combine all embeddings
            embeddings = self._combine_embeddings_advanced(
                pos_img_emb, pos_tag_emb, pos_cond_emb,
                neg_img_emb, neg_tag_emb, neg_cond_emb
            )
            
            # Parse ratings
            ratings_list = [r.strip() for r in selected_ratings.split(",")]
            
            # Debug: Check if embeddings are all zeros
            np_lib = ensure_dependencies()[0]
            if np_lib.allclose(embeddings, 0):
                logging.warning("[Danbooru Advanced] WARNING: Final embeddings are all zeros!")
            
            # Perform search
            logging.info(f"[Danbooru Advanced] Searching with embedding shape: {embeddings.shape}")
            
            try:
                # Ensure embeddings are the right format for FAISS
                if not embeddings.flags['C_CONTIGUOUS']:
                    logging.info("[Danbooru Advanced] Making embeddings C-contiguous for FAISS")
                    embeddings = np_lib.ascontiguousarray(embeddings, dtype=np_lib.float32)
                
                # Check index dimension
                if hasattr(self.knn_index, 'd'):
                    index_dim = self.knn_index.d
                    if index_dim != embeddings.shape[1]:
                        error_msg = f"Dimension mismatch: index expects {index_dim} but got {embeddings.shape[1]}"
                        logging.error(f"[Danbooru Advanced] {error_msg}")
                        return {"ui": {"text": [f"ERROR: {error_msg}", "", ""]}, "result": (f"ERROR: {error_msg}", "", "")}
                
                # Perform the search
                dists, indexes = self.knn_index.search(embeddings, k=n_neighbours)
                
                # Validate results
                if indexes is None or len(indexes) == 0:
                    logging.error("[Danbooru Advanced] FAISS search returned no results")
                    return {"ui": {"text": ["ERROR: No results from search", "", ""]}, "result": ("ERROR: No results from search", "", "")}
                
                neighbours_ids = self.images_ids[indexes][0]
                neighbours_ids = [int(x) for x in neighbours_ids]
                
                logging.info(f"[Danbooru Advanced] Found IDs: {neighbours_ids[:5]}, distances: {dists[0][:5]}")
                
            except Exception as e:
                error_msg = f"FAISS search failed: {str(e)}"
                logging.error(f"[Danbooru Advanced] {error_msg}")
                import traceback
                traceback.print_exc()
                return {"ui": {"text": [f"ERROR: {error_msg}", "", ""]}, "result": (f"ERROR: {error_msg}", "", "")}
            
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
            
            return {"ui": {"text": [top_id, all_ids_str, scores_str]}, "result": (top_id, all_ids_str, scores_str)}
            
        except ImportError as e:
            error_msg = f"Missing dependency: {e}"
            logging.error(f"[Danbooru Advanced] {error_msg}")
            return {"ui": {"text": [f"ERROR: {error_msg}", "", ""]}, "result": (f"ERROR: {error_msg}", "", "")}
        except Exception as e:
            error_msg = f"Error during lookup: {e}"
            logging.error(f"[Danbooru Advanced] {error_msg}")
            return {"ui": {"text": [f"ERROR: {error_msg}", "", ""]}, "result": (f"ERROR: {error_msg}", "", "")}


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "DanbooruFAISSLookupAdvanced": DanbooruFAISSLookupAdvanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DanbooruFAISSLookupAdvanced": "Danbooru FAISS Lookup (Advanced)"
}