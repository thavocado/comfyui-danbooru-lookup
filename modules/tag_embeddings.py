"""
Tag to embedding conversion using CLIP/SigLIP models.
Supports both JAX/FLAX and PyTorch implementations.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Union
import json

def _check_jax_available():
    """Check if JAX/FLAX is available (dynamic check)."""
    # Check if already loaded to avoid re-importing
    if 'jax' in sys.modules and 'flax' in sys.modules:
        return True
    
    try:
        import jax
        import jax.numpy as jnp
        import flax
        return True
    except ImportError:
        return False
    except Exception as e:
        # Handle PyTreeDef errors
        if "PyTreeDef" in str(e):
            # JAX is already loaded, consider it available
            return True
        return False

def _check_torch_available():
    """Check if PyTorch is available (dynamic check)."""
    # Check if already loaded to avoid re-importing
    if 'torch' in sys.modules:
        return True
    
    try:
        import torch
        return True
    except ImportError:
        return False

from .model_manager import ModelManager

# Lazy model loading - don't import JAX at module level
_clip_model_class = None
_jax_import_error = None

def _get_clip_model_class():
    """Get the CLIP model class, importing JAX only when needed."""
    global _clip_model_class, _jax_import_error
    
    if _clip_model_class is not None:
        return _clip_model_class
    
    if _jax_import_error:
        # We already tried and failed
        raise _jax_import_error
    
    try:
        if not _check_jax_available():
            raise ImportError("JAX is not available")
        
        # Import JAX modules only when actually needed
        # But check if they're already imported first
        if 'flax.linen' in sys.modules:
            nn = sys.modules['flax.linen']
        else:
            import flax.linen as nn
            
        if 'jax' in sys.modules:
            jax = sys.modules['jax']
        else:
            import jax
            
        if 'jax.numpy' in sys.modules:
            jnp = sys.modules['jax.numpy']
        else:
            import jax.numpy as jnp
        
        class TextEncoder(nn.Module):
            """CLIP Text Encoder with residual connections."""
            out_units: int = 1024
            
            @nn.compact
            def __call__(self, x, training=False):
                # First dense layer
                x = nn.Dense(features=self.out_units)(x)
                
                # Residual branch with SiLU activation
                res = nn.silu(x)
                res = nn.Dense(features=self.out_units)(res)
                res = nn.Dropout(0.1)(res, deterministic=not training)
                
                # Residual connection
                x = x + res
                return x
        
        class CLIPModel(nn.Module):
            """CLIP model for tag encoding."""
            out_units: int = 1024
            
            def setup(self):
                self.text_enc = TextEncoder(out_units=self.out_units)
            
            def encode_text(self, text):
                """Encode text (tag one-hot) to embeddings."""
                return self.text_enc(text, training=False)
            
            @nn.compact
            def __call__(self, image, text, training=False):
                # We only need encode_text for tag embeddings
                text_emb = self.encode_text(text)
                return text_emb
        
        _clip_model_class = CLIPModel
        return CLIPModel
        
    except Exception as e:
        _jax_import_error = e
        logging.error(f"[Tag Embeddings] Failed to load JAX-based CLIP model: {e}")
        
        # Return a dummy class that will raise errors when used
        class CLIPModelStub:
            def __init__(self, out_units: int = 1024):
                self.out_units = out_units
            
            def apply(self, *args, **kwargs):
                raise NotImplementedError(f"JAX is required for CLIP model inference: {_jax_import_error}")
        
        return CLIPModelStub

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
            if _check_jax_available():
                # Load msgpack parameters
                import flax
                with open(model_path, "rb") as f:
                    data = f.read()
                
                params = flax.serialization.msgpack_restore(data)
                # The params should be under "model" key based on original implementation
                if "model" in params:
                    self.model_params[variant] = params["model"]
                else:
                    # Fallback if structure is different
                    self.model_params[variant] = params
                self.loaded_variant = variant
                logging.info(f"Loaded {variant} model parameters")
                
                # Debug: log the parameter structure
                def log_param_structure(params, prefix="", max_depth=3, current_depth=0):
                    if current_depth >= max_depth:
                        return
                    if isinstance(params, dict):
                        for k in list(params.keys())[:5]:  # Show first 5 keys
                            logging.debug(f"{prefix}{k}: {type(params[k])}")
                            if isinstance(params[k], dict):
                                log_param_structure(params[k], prefix + "  ", max_depth, current_depth + 1)
                
                logging.debug(f"[Tag Embeddings] {variant} parameter structure:")
                log_param_structure(self.model_params[variant])
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
            if not _check_jax_available():
                logging.error("[Tag Embeddings] JAX/FLAX is required but not available.")
                logging.error("[Tag Embeddings] Please restart ComfyUI to install dependencies.")
                raise RuntimeError(error_msg + " Please restart ComfyUI.")
            else:
                raise RuntimeError(error_msg)
        
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
            # Get output dimensions based on variant
            out_units = 1024 if variant == "CLIP" else 1152
            
            # Get the model class (this will import JAX only when first used)
            try:
                CLIPModel = _get_clip_model_class()
            except Exception as e:
                if "PyTreeDef" in str(e):
                    logging.error("[Tag Embeddings] JAX PyTreeDef conflict detected. JAX may already be loaded elsewhere.")
                    logging.error("[Tag Embeddings] Returning zero embeddings as fallback.")
                    dim = self.get_embedding_dim(variant) or 1024
                    return np.zeros((1, dim), dtype=np.float32)
                else:
                    raise
            
            # Create model instance
            model = CLIPModel(out_units=out_units)
            
            # Import jax only when needed, but check if already loaded
            if 'jax' in sys.modules:
                jax = sys.modules['jax']
            else:
                import jax
            
            # Use model.apply with the loaded parameters
            embeddings = model.apply(
                {"params": self.model_params[variant]},
                onehot,
                method=model.encode_text,
            )
            
            # Convert from JAX array to numpy
            embeddings = jax.device_get(embeddings)
            
            logging.info(f"[Tag Embeddings] Successfully encoded tags with {variant}, shape: {embeddings.shape}")
            return embeddings
                
        except Exception as e:
            error_msg = str(e)
            if "PyTreeDef" in error_msg:
                logging.error("[Tag Embeddings] JAX initialization conflict. This often happens when JAX is already loaded.")
                logging.error("[Tag Embeddings] Returning zero embeddings to allow node to function.")
                dim = self.get_embedding_dim(variant) or 1024
                return np.zeros((1, dim), dtype=np.float32)
            else:
                logging.error(f"Failed to encode tags: {e}")
                import traceback
                traceback.print_exc()
                
                # Try fallback approach
                try:
                    if variant in self.model_params:
                        logging.info("[Tag Embeddings] Using fallback zero embeddings...")
                        dim = self.get_embedding_dim(variant) or 1024
                        return np.zeros((1, dim), dtype=np.float32)
                except:
                    pass
                
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
        return _check_jax_available() or _check_torch_available()
    
    def encode_tags_simple(self, tags: Union[str, List[str]], variant: str = "CLIP") -> Optional[np.ndarray]:
        """
        Simplified encoding without full model - just returns zero embeddings.
        This is a fallback when JAX/FLAX is not available.
        """
        dim = self.get_embedding_dim(variant) or 1024
        return np.zeros((1, dim), dtype=np.float32)