import os
import json
import torch
import logging
from pathlib import Path
import requests

# Delay imports to avoid errors when dependencies are not installed
numpy = None
faiss = None
pandas = None

def ensure_dependencies():
    """Ensure all dependencies are loaded"""
    global numpy, faiss, pandas
    
    if numpy is None:
        try:
            import numpy as np
            numpy = np
        except ImportError:
            raise ImportError("numpy not installed. Please run install.py or: pip install numpy")
    
    if faiss is None:
        try:
            import faiss as _faiss
            faiss = _faiss
        except ImportError:
            raise ImportError("faiss-cpu not installed. Please run install.py or: pip install faiss-cpu")
    
    if pandas is None:
        try:
            import pandas as pd
            pandas = pd
        except ImportError:
            raise ImportError("pandas not installed. Please run install.py or: pip install pandas")
    
    return numpy, faiss, pandas

# Import DataLoader but handle if it fails
try:
    from .data_loader import DataLoader
except ImportError:
    DataLoader = None

class DanbooruFAISSLookup:
    """
    A ComfyUI node that performs FAISS cosine similarity lookup on Danbooru embeddings
    using positive and negative CLIP conditioning inputs.
    """
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # DataLoader might not be available if dependencies are missing
        if DataLoader is not None:
            self.data_loader = DataLoader(self.data_dir)
        else:
            self.data_loader = None
            
        self.loaded_variant = None
        self.knn_index = None
        self.images_ids = None
        self.tags_df = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_conditioning": ("CONDITIONING",),
                "negative_conditioning": ("CONDITIONING",),
            },
            "optional": {
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
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("danbooru_id",)
    FUNCTION = "lookup"
    CATEGORY = "conditioning/danbooru"
    
    def _load_index(self):
        """Load FAISS index and related data."""
        # Ensure dependencies are available
        np, faiss_lib, pd = ensure_dependencies()
        
        if self.knn_index is None:
            # Ensure data is downloaded
            if self.data_loader is None:
                raise RuntimeError("DataLoader not available. Please install dependencies.")
            self.data_loader.ensure_data_downloaded()
            
            # Load the FAISS index
            index_path = self.data_dir / "index/cosine_knn.index"
            self.knn_index = faiss_lib.read_index(str(index_path))
            
            # Load the image IDs
            ids_path = self.data_dir / "index/cosine_ids.npy"
            self.images_ids = np.load(str(ids_path))
            
            # Load and configure the index
            infos_path = self.data_dir / "index/cosine_infos.json"
            with open(infos_path, 'r') as f:
                config = json.loads(f.read())["index_param"]
            faiss_lib.ParameterSpace().set_index_parameters(self.knn_index, config)
            
            # Load tags dataframe
            tags_path = self.data_dir / "data/selected_tags.csv"
            self.tags_df = pd.read_csv(str(tags_path))
            
            logging.info("FAISS index loaded successfully")
    
    def _extract_embeddings_from_conditioning(self, conditioning):
        """Extract embeddings from ComfyUI conditioning format."""
        # Ensure numpy is available
        np = ensure_dependencies()[0]
        
        if conditioning is None or len(conditioning) == 0:
            return None
        
        # ComfyUI conditioning format is a list of tuples: [(tensor, dict), ...]
        # The tensor contains the actual embeddings
        embeddings = []
        for cond in conditioning:
            if isinstance(cond, tuple) and len(cond) >= 1:
                tensor = cond[0]
                if isinstance(tensor, torch.Tensor):
                    # Convert to numpy and average over sequence dimension if needed
                    emb = tensor.cpu().numpy()
                    if len(emb.shape) == 3:  # [batch, seq, dim]
                        emb = emb.mean(axis=1)  # Average over sequence
                    elif len(emb.shape) == 2:  # [batch, dim]
                        pass
                    else:
                        emb = emb.reshape(1, -1)  # Ensure 2D
                    embeddings.append(emb)
        
        if embeddings:
            # Concatenate and average all embeddings
            combined = np.concatenate(embeddings, axis=0)
            return combined.mean(axis=0, keepdims=True)
        
        return None
    
    def _combine_embeddings(self, pos_emb, neg_emb):
        """Combine positive and negative embeddings."""
        # Ensure dependencies
        np, faiss_lib, _ = ensure_dependencies()
        
        # Ensure we have the right dimensionality
        if self.knn_index is None:
            self._load_index()
        
        embedding_dim = self.knn_index.d
        
        if pos_emb is None:
            pos_emb = np.zeros((1, embedding_dim), dtype=np.float32)
        if neg_emb is None:
            neg_emb = np.zeros((1, embedding_dim), dtype=np.float32)
        
        # Ensure correct shape
        if pos_emb.shape[1] != embedding_dim:
            # Pad or truncate to match expected dimension
            if pos_emb.shape[1] < embedding_dim:
                pos_emb = np.pad(pos_emb, ((0, 0), (0, embedding_dim - pos_emb.shape[1])), mode='constant')
            else:
                pos_emb = pos_emb[:, :embedding_dim]
        
        if neg_emb.shape[1] != embedding_dim:
            # Pad or truncate to match expected dimension
            if neg_emb.shape[1] < embedding_dim:
                neg_emb = np.pad(neg_emb, ((0, 0), (0, embedding_dim - neg_emb.shape[1])), mode='constant')
            else:
                neg_emb = neg_emb[:, :embedding_dim]
        
        # Normalize embeddings
        faiss_lib.normalize_L2(pos_emb)
        faiss_lib.normalize_L2(neg_emb)
        
        # Combine: positive - negative
        result = pos_emb - neg_emb
        faiss_lib.normalize_L2(result)
        
        return result
    
    def danbooru_id_to_url(self, image_id, selected_ratings, api_username="", api_key=""):
        """Convert Danbooru ID to URL with rating filtering."""
        headers = {"User-Agent": "comfyui_danbooru_node"}
        ratings_to_letters = {
            "General": "g",
            "Sensitive": "s",
            "Questionable": "q",
            "Explicit": "e",
        }
        
        acceptable_ratings = []
        for rating in selected_ratings:
            if rating in ratings_to_letters:
                acceptable_ratings.append(ratings_to_letters[rating])
        
        image_url = f"https://danbooru.donmai.us/posts/{image_id}.json"
        if api_username != "" and api_key != "":
            image_url = f"{image_url}?api_key={api_key}&login={api_username}"
        
        try:
            r = requests.get(image_url, headers=headers, timeout=10)
            if r.status_code != 200:
                return None
            
            content = json.loads(r.text)
            if content.get("rating") in acceptable_ratings:
                return content.get("large_file_url")
        except Exception as e:
            logging.warning(f"Failed to fetch Danbooru URL for ID {image_id}: {e}")
        
        return None
    
    def lookup(self, positive_conditioning, negative_conditioning, 
               selected_ratings="General,Sensitive", n_neighbours=5,
               api_username="", api_key=""):
        """
        Perform FAISS lookup using positive and negative conditioning.
        Returns the top matching Danbooru post ID.
        """
        try:
            # Ensure dependencies are available
            ensure_dependencies()
            
            # Parse ratings string
            ratings_list = [r.strip() for r in selected_ratings.split(",")]
            
            # Load index if not already loaded
            self._load_index()
            
            # Extract embeddings from conditioning
            pos_emb = self._extract_embeddings_from_conditioning(positive_conditioning)
            neg_emb = self._extract_embeddings_from_conditioning(negative_conditioning)
            
            # Combine embeddings
            embeddings = self._combine_embeddings(pos_emb, neg_emb)
            
            # Perform search
            dists, indexes = self.knn_index.search(embeddings, k=n_neighbours)
            neighbours_ids = self.images_ids[indexes][0]
            neighbours_ids = [int(x) for x in neighbours_ids]
            
            # Find the first valid ID based on ratings
            valid_id = None
            for image_id in neighbours_ids:
                url = self.danbooru_id_to_url(image_id, ratings_list, api_username, api_key)
                if url is not None:
                    valid_id = str(image_id)
                    break
            
            # If no valid ID found, return the first one anyway
            if valid_id is None and len(neighbours_ids) > 0:
                valid_id = str(neighbours_ids[0])
            
            return (valid_id if valid_id else "",)
            
        except ImportError as e:
            error_msg = f"Missing dependency: {e}"
            logging.error(f"[Danbooru Lookup] {error_msg}")
            return (f"ERROR: {error_msg}",)
        except Exception as e:
            error_msg = f"Error during lookup: {e}"
            logging.error(f"[Danbooru Lookup] {error_msg}")
            return (f"ERROR: {error_msg}",)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "DanbooruFAISSLookup": DanbooruFAISSLookup
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DanbooruFAISSLookup": "Danbooru FAISS Lookup"
}