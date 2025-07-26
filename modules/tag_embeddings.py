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

# Try to import flax once at module level to avoid duplicate registrations
_flax_module = None
_flax_serialization = None
_flax_linen = None
_jax_module = None

try:
    if 'flax' in sys.modules:
        _flax_module = sys.modules['flax']
        if 'flax.serialization' in sys.modules:
            _flax_serialization = sys.modules['flax.serialization']
        if 'flax.linen' in sys.modules:
            _flax_linen = sys.modules['flax.linen']
    else:
        import flax
        _flax_module = flax
        import flax.serialization
        _flax_serialization = flax.serialization
        import flax.linen
        _flax_linen = flax.linen
        
    if 'jax' in sys.modules:
        _jax_module = sys.modules['jax']
    else:
        import jax
        _jax_module = jax
except ImportError:
    pass
except Exception as e:
    if "PyTreeDef" not in str(e):
        logging.debug(f"[Tag Embeddings] Module-level import error: {e}")

def _check_jax_available():
    """Check if JAX/FLAX is available (dynamic check)."""
    # Check our module-level imports first
    if _flax_module is not None and _jax_module is not None:
        return True
    
    # Check if already loaded in sys.modules
    if 'jax' in sys.modules and 'flax' in sys.modules:
        return True
    
    try:
        if _jax_module is None:
            import jax
        if _flax_module is None:
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
_models_defined = False

def apply_clip_model(params, x, out_units=1024):
    """Apply CLIP model using pure JAX functions without Flax nn.Module.
    
    This implements the same architecture as the original CLIP model:
    - First dense layer
    - Residual branch with SiLU activation and second dense layer
    - Dropout (set to 0 for inference)
    - Residual connection
    """
    if not _check_jax_available():
        raise ImportError("JAX is not available")
    
    # Get JAX functions
    if _jax_module is not None:
        jax = _jax_module
        jnp = jax.numpy
    else:
        import jax
        import jax.numpy as jnp
    
    # Extract text encoder parameters
    if "text_enc" in params:
        text_enc_params = params["text_enc"]
    else:
        # Fallback if structure is different
        text_enc_params = params
    
    # Apply first dense layer
    # Dense layer: y = x @ W.T + b
    dense0_kernel = text_enc_params["Dense_0"]["kernel"]  # shape: (input_dim, out_units)
    dense0_bias = text_enc_params["Dense_0"]["bias"]      # shape: (out_units,)
    
    x = jnp.dot(x, dense0_kernel) + dense0_bias
    
    # Residual branch
    # SiLU activation (x * sigmoid(x))
    res = x * jax.nn.sigmoid(x)
    
    # Second dense layer
    dense1_kernel = text_enc_params["Dense_1"]["kernel"]  # shape: (out_units, out_units)
    dense1_bias = text_enc_params["Dense_1"]["bias"]      # shape: (out_units,)
    
    res = jnp.dot(res, dense1_kernel) + dense1_bias
    
    # Note: Dropout is 0 during inference, so we skip it
    
    # Residual connection
    x = x + res
    
    return x

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
                # Load msgpack parameters using module-level import
                if _flax_serialization is not None:
                    serialization = _flax_serialization
                elif 'flax.serialization' in sys.modules:
                    serialization = sys.modules['flax.serialization']
                else:
                    import flax.serialization
                    serialization = flax.serialization
                
                with open(model_path, "rb") as f:
                    data = f.read()
                
                params = serialization.msgpack_restore(data)
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
            
            # Apply the model using pure JAX functions
            try:
                embeddings = apply_clip_model(
                    self.model_params[variant], 
                    onehot, 
                    out_units=out_units
                )
            except Exception as e:
                logging.error(f"[Tag Embeddings] Failed to apply CLIP model: {e}")
                raise
            
            # Use module-level jax import if available
            if _jax_module is not None:
                jax = _jax_module
            elif 'jax' in sys.modules:
                jax = sys.modules['jax']
            else:
                import jax
            
            # Convert from JAX array to numpy
            embeddings = jax.device_get(embeddings)
            
            logging.info(f"[Tag Embeddings] Successfully encoded tags with {variant}, shape: {embeddings.shape}")
            return embeddings
                
        except Exception as e:
            logging.error(f"Failed to encode tags: {e}")
            import traceback
            traceback.print_exc()
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