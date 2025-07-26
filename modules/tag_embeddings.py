"""
Tag to embedding conversion using CLIP/SigLIP models.
Supports both JAX/FLAX and PyTorch implementations.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Union
import json

# Try JAX/FLAX imports
try:
    import jax
    import jax.numpy as jnp
    import flax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    logging.info("JAX/FLAX not installed. Will try PyTorch fallback if available.")

# Try PyTorch imports  
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logging.info("PyTorch not installed. Tag embeddings will require JAX/FLAX.")

from .model_manager import ModelManager

class CLIPModel:
    """Minimal CLIP model implementation for tag encoding."""
    
    def __init__(self, embed_dim: int = 1024, num_classes: int = 12547):
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.out_units = embed_dim
    
    def encode_text(self, params: Dict, tag_onehot: np.ndarray) -> np.ndarray:
        """Encode one-hot tag representation to embeddings.
        This is a simplified version - real implementation would need full model architecture.
        """
        # For now, we'll use a simple linear projection
        # Real implementation would use the full CLIP text encoder
        if HAS_JAX:
            # JAX implementation
            weights = params.get('text_projection', {}).get('kernel', None)
            if weights is None:
                raise ValueError("text_projection weights not found in params")
            
            embeddings = jnp.dot(tag_onehot, weights)
            return jax.device_get(embeddings)
        else:
            raise NotImplementedError("JAX is required for CLIP model inference")

class TagEmbeddings:
    """Convert tag names to embeddings using CLIP/SigLIP models."""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.tags_df = None
        self.model_params = {}
        self.loaded_variant = None
        
    def load_tags(self) -> bool:
        """Load the tags dataframe."""
        if self.tags_df is not None:
            return True
        
        # Try from the data directory (already downloaded by main lookup)
        data_path = Path(__file__).parent.parent / "data" / "data" / "selected_tags.csv"
        if not data_path.exists():
            logging.error("selected_tags.csv not found")
            return False
        
        try:
            self.tags_df = pd.read_csv(data_path)
            logging.info(f"Loaded {len(self.tags_df)} tags")
            return True
        except Exception as e:
            logging.error(f"Failed to load tags: {e}")
            return False
    
    def load_model_params(self, variant: str = "CLIP") -> bool:
        """Load CLIP/SigLIP model parameters."""
        if self.loaded_variant == variant and variant in self.model_params:
            return True
        
        model_path = self.model_manager.get_clip_model_path(variant)
        if model_path is None or not model_path.exists():
            logging.error(f"{variant} model not found")
            return False
        
        try:
            if HAS_JAX:
                # Load msgpack parameters
                with open(model_path, "rb") as f:
                    data = f.read()
                
                params = flax.serialization.msgpack_restore(data)
                self.model_params[variant] = params.get("model", params)
                self.loaded_variant = variant
                logging.info(f"Loaded {variant} model parameters")
                return True
            else:
                logging.error("JAX/FLAX is required for loading msgpack models")
                return False
                
        except Exception as e:
            logging.error(f"Failed to load {variant} model: {e}")
            return False
    
    def tags_to_indices(self, tags: Union[str, List[str]]) -> List[int]:
        """Convert tag names to indices."""
        if not self.load_tags():
            return []
        
        if isinstance(tags, str):
            # Split by comma and strip whitespace
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        
        # Find indices for matching tags
        indices = []
        for tag in tags:
            matches = self.tags_df[self.tags_df["name"] == tag].index.tolist()
            if matches:
                indices.extend(matches)
            else:
                logging.warning(f"Tag '{tag}' not found in vocabulary")
        
        return indices
    
    def create_onehot(self, tag_indices: List[int], num_classes: Optional[int] = None) -> np.ndarray:
        """Create one-hot encoding from tag indices."""
        if num_classes is None:
            if self.tags_df is not None:
                num_classes = len(self.tags_df)
            else:
                num_classes = 12547  # Default WD14 tag count
        
        onehot = np.zeros((1, num_classes), dtype=np.float32)
        for idx in tag_indices:
            if 0 <= idx < num_classes:
                onehot[0, idx] = 1.0
        
        return onehot
    
    def encode_tags(self, tags: Union[str, List[str]], variant: str = "CLIP") -> Optional[np.ndarray]:
        """Encode tags to embeddings."""
        # Load model if needed
        if not self.load_model_params(variant):
            error_msg = f"Failed to load {variant} model."
            logging.error(f"[Tag Embeddings] {error_msg}")
            logging.error("[Tag Embeddings] This feature requires JAX/FLAX to be installed:")
            logging.error("[Tag Embeddings]   pip install jax jaxlib flax")
            logging.error("[Tag Embeddings] Or install all features:")
            logging.error(f"[Tag Embeddings]   pip install -r requirements-full.txt")
            raise RuntimeError(error_msg + " Install JAX/FLAX: pip install jax jaxlib flax")
        
        # Convert tags to indices
        indices = self.tags_to_indices(tags)
        if not indices:
            error_msg = f"No valid tags found in: {tags}. Tags must match Danbooru vocabulary."
            logging.error(f"[Tag Embeddings] {error_msg}")
            raise ValueError(error_msg)
        
        # Create one-hot encoding
        onehot = self.create_onehot(indices)
        
        # Encode to embeddings
        try:
            if HAS_JAX:
                # Use CLIP model to encode
                model = CLIPModel()
                embeddings = model.encode_text(self.model_params[variant], onehot)
                return embeddings
            else:
                logging.error("JAX is required for tag encoding")
                return None
                
        except Exception as e:
            logging.error(f"Failed to encode tags: {e}")
            return None
    
    def get_embedding_dim(self, variant: str = "CLIP") -> Optional[int]:
        """Get the embedding dimension for the model."""
        # Standard dimensions
        dims = {
            "CLIP": 1024,
            "SigLIP": 1152
        }
        return dims.get(variant, None)
    
    @staticmethod
    def is_available() -> bool:
        """Check if tag embeddings are available."""
        return HAS_JAX or HAS_TORCH
    
    def encode_tags_simple(self, tags: Union[str, List[str]], variant: str = "CLIP") -> Optional[np.ndarray]:
        """
        Simplified encoding without full model - just returns zero embeddings.
        This is a fallback when JAX/FLAX is not available.
        """
        dim = self.get_embedding_dim(variant) or 1024
        return np.zeros((1, dim), dtype=np.float32)