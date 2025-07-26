"""
WD14 tagger embeddings extraction for image inputs using dghs-imgutils.
"""

import logging
import numpy as np
from typing import Optional, Union
import torch

# Try imports
try:
    from imgutils.tagging import wd14
    HAS_IMGUTILS = True
except ImportError:
    HAS_IMGUTILS = False
    logging.warning("dghs-imgutils not installed. WD14 image embeddings will not be available.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("Pillow not installed. Image processing will be limited.")

class WD14Embeddings:
    """Extract embeddings from images using WD14 ConvNext model via dghs-imgutils."""
    
    def __init__(self, model_manager=None):
        # model_manager is kept for compatibility but not used with dghs-imgutils
        pass
        
    def extract_embeddings(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Optional[np.ndarray]:
        """Extract embeddings from an image using dghs-imgutils."""
        if not HAS_IMGUTILS:
            logging.error("dghs-imgutils is required for WD14 embeddings")
            return None
        
        try:
            # Convert to PIL Image if needed
            if isinstance(image, torch.Tensor):
                # Assume ComfyUI format: [B, H, W, C] with values in [0, 1]
                if len(image.shape) == 4:
                    image = image[0]  # Take first image from batch
                image = (image.cpu().numpy() * 255).astype(np.uint8)
                if HAS_PIL:
                    image = Image.fromarray(image)
            elif isinstance(image, np.ndarray):
                # Assume numpy array in [0, 255] range
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                if HAS_PIL:
                    image = Image.fromarray(image)
            
            # Use dghs-imgutils to get embeddings
            # The library handles all preprocessing internally
            embeddings = wd14.get_wd14_tags(
                image,
                model_name="ConvNext",
                fmt="embedding"  # This returns embeddings instead of tags
            )
            
            # Match original behavior - always expand dims
            embeddings = np.expand_dims(embeddings, 0)
            
            logging.info(f"[WD14] Raw embedding shape: {embeddings.shape}")
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logging.error(f"Failed to extract embeddings: {e}")
            return None
    
    def get_embedding_dim(self) -> Optional[int]:
        """Get the dimension of the embeddings."""
        # WD14 ConvNext embeddings are typically 1024-dimensional
        return 1024
    
    @staticmethod
    def is_available() -> bool:
        """Check if WD14 embeddings are available."""
        return HAS_IMGUTILS