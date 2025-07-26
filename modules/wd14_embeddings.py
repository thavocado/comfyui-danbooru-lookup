"""
WD14 tagger embeddings extraction for image inputs using dghs-imgutils.
"""

import logging
import numpy as np
import os
from typing import Optional, Union
import torch

# Try imports
try:
    from imgutils.tagging import wd14
    HAS_IMGUTILS = True
    logging.info("[WD14] dghs-imgutils successfully imported")
except ImportError as e:
    HAS_IMGUTILS = False
    logging.error(f"[WD14] dghs-imgutils not installed: {e}")
    logging.error("[WD14] Install with: pip install dghs-imgutils[gpu]")
    logging.error("[WD14] WD14 image embeddings will not be available without this dependency.")

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
        
    def extract_embeddings(self, image: Union[Image.Image, torch.Tensor, np.ndarray], hf_token: Optional[str] = None) -> Optional[np.ndarray]:
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
            
            # Temporarily set HF token if provided
            old_token = os.environ.get('HF_TOKEN')
            if hf_token:
                os.environ['HF_TOKEN'] = hf_token
                logging.info("[WD14] Using provided HuggingFace token")
            
            try:
                # Use dghs-imgutils to get embeddings
                # The library handles all preprocessing internally
                embeddings = wd14.get_wd14_tags(
                    image,
                    model_name="ConvNext",
                    fmt="embedding"  # This returns embeddings instead of tags
                )
            finally:
                # Restore original token
                if hf_token:
                    if old_token:
                        os.environ['HF_TOKEN'] = old_token
                    else:
                        os.environ.pop('HF_TOKEN', None)
            
            # Match original behavior - always expand dims
            embeddings = np.expand_dims(embeddings, 0)
            
            logging.info(f"[WD14] Raw embedding shape: {embeddings.shape}")
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            error_str = str(e)
            if "401" in error_str or "authentication" in error_str.lower() or "unauthorized" in error_str.lower():
                logging.error("[WD14] HuggingFace authentication error detected!")
                logging.error("[WD14] The WD14 model may require authentication.")
                logging.error("[WD14] Please set up HuggingFace authentication:")
                logging.error("[WD14] 1. Get a token from https://huggingface.co/settings/tokens")
                logging.error("[WD14] 2. Run: huggingface-cli login")
                logging.error("[WD14] OR set environment variable: HF_TOKEN=your_token_here")
            else:
                logging.error(f"[WD14] Failed to extract embeddings: {e}")
            return None
    
    def get_embedding_dim(self) -> Optional[int]:
        """Get the dimension of the embeddings."""
        # WD14 ConvNext embeddings are typically 1024-dimensional
        return 1024
    
    @staticmethod
    def is_available() -> bool:
        """Check if WD14 embeddings are available."""
        return HAS_IMGUTILS