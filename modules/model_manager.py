"""
Model manager for downloading and caching models used by Danbooru lookup.
Handles WD14 tagger and CLIP/SigLIP models.
"""

import os
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Try imports but don't fail if missing
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    logging.warning("huggingface_hub not installed. Model downloads will be limited.")

from .data_loader import DataLoader

class ModelManager:
    """Manages model downloads and caching."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        
        self.base_dir = base_dir
        self.models_dir = base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            "clip_model": {
                "direct_url": "https://huggingface.co/spaces/SmilingWolf/danbooru2022_embeddings_playground/resolve/main/data/wd-v1-4-convnext-tagger-v2/clip.msgpack",
                "filename": "clip.msgpack",
                "type": "msgpack"
            },
            "siglip_model": {
                "direct_url": "https://huggingface.co/spaces/SmilingWolf/danbooru2022_embeddings_playground/resolve/main/data/wd-v1-4-convnext-tagger-v2/siglip.msgpack",
                "filename": "siglip.msgpack",
                "type": "msgpack"
            }
        }
        
        # Cache for loaded models
        self._model_cache = {}
    
    def ensure_model(self, model_name: str, force_download: bool = False) -> Path:
        """Ensure a model is downloaded and return its path."""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = self.model_configs[model_name]
        model_dir = self.models_dir / model_name
        
        # Check if already downloaded
        if not force_download and self._is_model_complete(model_dir, config):
            return model_dir
        
        logging.info(f"Downloading {model_name} model...")
        model_dir.mkdir(exist_ok=True)
        
        # Use direct download for msgpack files
        if "direct_url" in config:
            try:
                import requests
                local_path = model_dir / config["filename"]
                
                if force_download or not local_path.exists():
                    logging.info(f"Downloading from: {config['direct_url']}")
                    response = requests.get(config['direct_url'], stream=True)
                    response.raise_for_status()
                    
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    logging.info(f"Model {model_name} downloaded successfully")
                
                return model_dir
                
            except Exception as e:
                logging.error(f"Failed to download {model_name}: {e}")
                logging.error(f"You may need to manually download the file from:")
                logging.error(f"  {config['direct_url']}")
                logging.error(f"And place it at: {model_dir / config['filename']}")
                raise
        else:
            # Original HF hub download logic (not used for msgpack files)
            raise NotImplementedError("HF hub download not supported for this model type")
    
    def _is_model_complete(self, model_dir: Path, config: Dict[str, Any]) -> bool:
        """Check if all model files are present."""
        if not model_dir.exists():
            return False
        
        # Check for direct download models
        if "filename" in config:
            local_path = model_dir / config["filename"]
            return local_path.exists()
        
        # Legacy check for HF hub models
        if "files" in config:
            for file_path in config["files"]:
                local_path = model_dir / Path(file_path).name
                if not local_path.exists():
                    return False
        
        return True
    
    def get_clip_model_path(self, variant: str = "CLIP") -> Optional[Path]:
        """Get path to CLIP/SigLIP model weights."""
        model_name = "clip_model" if variant == "CLIP" else "siglip_model"
        
        try:
            self.ensure_model(model_name)
            # The msgpack files are downloaded to a specific structure
            if variant == "CLIP":
                return self.models_dir / model_name / "clip.msgpack"
            else:
                return self.models_dir / model_name / "siglip.msgpack"
        except Exception as e:
            logging.error(f"Failed to get {variant} model: {e}")
            return None
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check which dependencies are available."""
        deps = {
            "huggingface_hub": HAS_HF_HUB,
            "dghs_imgutils": False,
            "flax": False,
            "jax": False,
            "torch": False,
        }
        
        try:
            import imgutils
            deps["dghs_imgutils"] = True
        except ImportError:
            pass
        
        try:
            import flax
            deps["flax"] = True
        except ImportError:
            pass
        
        try:
            import jax
            deps["jax"] = True
        except ImportError:
            pass
        
        try:
            import torch
            deps["torch"] = True
        except ImportError:
            pass
        
        return deps