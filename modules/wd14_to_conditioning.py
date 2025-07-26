"""
WD14 to Conditioning node - converts images to ComfyUI conditioning using WD14 embeddings.
"""

import logging
import torch
from typing import Optional, Tuple

from .wd14_embeddings import WD14Embeddings


class WD14ToConditioning:
    """
    Convert images to ComfyUI conditioning format using WD14 embeddings.
    This allows using image embeddings as conditioning in other nodes.
    """
    
    def __init__(self):
        self.wd14_embeddings = WD14Embeddings()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "hf_token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "HuggingFace token (optional)"
                }),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "conditioning/danbooru"
    
    def encode(self, image, hf_token=""):
        """
        Convert an image to conditioning using WD14 embeddings.
        
        Args:
            image: ComfyUI IMAGE format
            hf_token: Optional HuggingFace token for authentication
            
        Returns:
            ComfyUI CONDITIONING format
        """
        try:
            # Check if WD14 is available
            if not self.wd14_embeddings.is_available():
                error_msg = "WD14 embeddings not available. Please install: pip install dghs-imgutils[gpu]"
                logging.error(f"[WD14ToConditioning] {error_msg}")
                # Return empty conditioning with error
                empty_tensor = torch.zeros((1, 1024), dtype=torch.float32)
                return ([(empty_tensor, {"error": error_msg})],)
            
            # Extract embeddings from image
            embeddings = self.wd14_embeddings.extract_embeddings(image, hf_token=hf_token)
            
            if embeddings is None:
                error_msg = "Failed to extract embeddings. Check console for details."
                logging.error(f"[WD14ToConditioning] {error_msg}")
                # Return empty conditioning with error
                empty_tensor = torch.zeros((1, 1024), dtype=torch.float32)
                return ([(empty_tensor, {"error": error_msg})],)
            
            # Convert numpy array to torch tensor
            # embeddings shape should be (1, 1024) from WD14
            tensor = torch.from_numpy(embeddings).float()
            
            logging.info(f"[WD14ToConditioning] Created conditioning with shape: {tensor.shape}")
            
            # Return in ComfyUI conditioning format
            # Format: [(tensor, dict), ...]
            conditioning = [(tensor, {})]
            
            return (conditioning,)
            
        except Exception as e:
            error_msg = f"Error creating conditioning: {str(e)}"
            logging.error(f"[WD14ToConditioning] {error_msg}")
            # Return empty conditioning with error
            empty_tensor = torch.zeros((1, 1024), dtype=torch.float32)
            return ([(empty_tensor, {"error": error_msg})],)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "WD14ToConditioning": WD14ToConditioning
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WD14ToConditioning": "WD14 to Conditioning"
}