"""
WD14 tagger embeddings extraction for image inputs.
Handles both standalone usage and integration with existing ComfyUI WD14 nodes.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import torch

# Try imports
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logging.warning("onnxruntime not installed. WD14 image embeddings will not be available.")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.warning("Pillow not installed. Image processing will be limited.")

from .model_manager import ModelManager

class WD14Embeddings:
    """Extract embeddings from images using WD14 ConvNext model."""
    
    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.model_manager = model_manager or ModelManager()
        self.model_session = None
        self.input_shape = (448, 448)  # WD14 ConvNext input size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
    def load_model(self) -> bool:
        """Load the ONNX model."""
        if self.model_session is not None:
            return True
        
        if not HAS_ONNX:
            logging.error("onnxruntime is required for WD14 embeddings")
            return False
        
        model_path = self.model_manager.get_wd14_model_path()
        if model_path is None or not model_path.exists():
            logging.error("WD14 model not found")
            return False
        
        try:
            # Create ONNX session with GPU if available
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.model_session = ort.InferenceSession(str(model_path), providers=providers)
            logging.info(f"WD14 model loaded with providers: {self.model_session.get_providers()}")
            return True
        except Exception as e:
            logging.error(f"Failed to load WD14 model: {e}")
            return False
    
    def preprocess_image(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Optional[np.ndarray]:
        """Preprocess image for WD14 model."""
        if not HAS_PIL and not isinstance(image, (torch.Tensor, np.ndarray)):
            logging.error("Pillow is required for image preprocessing")
            return None
        
        try:
            # Convert to PIL Image if needed
            if isinstance(image, torch.Tensor):
                # Assume ComfyUI format: [B, H, W, C] with values in [0, 1]
                if len(image.shape) == 4:
                    image = image[0]  # Take first image from batch
                image = (image.cpu().numpy() * 255).astype(np.uint8)
                image = Image.fromarray(image)
            elif isinstance(image, np.ndarray):
                # Assume numpy array in [0, 255] range
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            # Resize and convert to RGB
            image = image.convert('RGB')
            image = image.resize(self.input_shape, Image.Resampling.LANCZOS)
            
            # Convert to numpy and normalize
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # Apply ImageNet normalization
            image_np = (image_np - self.mean) / self.std
            
            # Transpose to NCHW format for ONNX
            image_np = np.transpose(image_np, (2, 0, 1))
            
            # Add batch dimension
            image_np = np.expand_dims(image_np, 0).astype(np.float32)
            
            return image_np
            
        except Exception as e:
            logging.error(f"Failed to preprocess image: {e}")
            return None
    
    def extract_embeddings(self, image: Union[Image.Image, torch.Tensor, np.ndarray]) -> Optional[np.ndarray]:
        """Extract embeddings from an image."""
        if not self.load_model():
            return None
        
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        if preprocessed is None:
            return None
        
        try:
            # Run inference
            input_name = self.model_session.get_inputs()[0].name
            outputs = self.model_session.run(None, {input_name: preprocessed})
            
            # The last output should be the embeddings (before classification layer)
            # For ConvNext, this is typically the second-to-last output
            if len(outputs) > 1:
                embeddings = outputs[-2]  # Penultimate layer
            else:
                embeddings = outputs[0]
            
            # Ensure shape is (1, embedding_dim)
            if len(embeddings.shape) > 2:
                embeddings = embeddings.reshape(embeddings.shape[0], -1)
            
            return embeddings
            
        except Exception as e:
            logging.error(f"Failed to extract embeddings: {e}")
            return None
    
    def get_embedding_dim(self) -> Optional[int]:
        """Get the dimension of the embeddings."""
        if not self.load_model():
            return None
        
        # Get output shape from model
        outputs = self.model_session.get_outputs()
        if len(outputs) > 1:
            # Use penultimate layer
            output_shape = outputs[-2].shape
        else:
            output_shape = outputs[0].shape
        
        # Last dimension is embedding size
        if isinstance(output_shape, list) and len(output_shape) >= 2:
            return output_shape[-1]
        
        return None
    
    @staticmethod
    def is_available() -> bool:
        """Check if WD14 embeddings are available."""
        return HAS_ONNX and HAS_PIL