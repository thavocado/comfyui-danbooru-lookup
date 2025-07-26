# ComfyUI Danbooru FAISS Lookup

A ComfyUI custom node that performs FAISS cosine similarity lookup on Danbooru embeddings using multiple input modes: CLIP conditioning, images with WD14 tagging, or text tags.

## Features

### Basic Node (Danbooru FAISS Lookup)
- Takes positive and negative CLIP conditioning as inputs
- Performs FAISS similarity search on Danbooru image embeddings
- Returns the best matching Danbooru post ID
- Supports rating filtering (General, Sensitive, Questionable, Explicit)
- Optional Danbooru API authentication for better access
- Automatic download of required data files from HuggingFace

### Advanced Node (Danbooru FAISS Lookup Advanced)
- **Multiple input modes**:
  - `conditioning`: Use CLIP conditioning (same as basic node)
  - `tags_and_images`: Use images + text tags (like original Danbooru search)
  - `hybrid`: Combine all available inputs for best results
- **Image inputs**: Process images through WD14 tagger for embeddings
- **Text tag inputs**: Convert Danbooru tags to embeddings using CLIP/SigLIP
- **Extended outputs**: Returns top match ID, all matching IDs, and similarity scores
- **Model selection**: Choose between CLIP and SigLIP for tag encoding

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/thavocado/comfyui-danbooru-lookup
   ```

2. Install the required dependencies (if auto install fails):
   ```bash
   pip install -r comfyui-danbooru-lookup/requirements.txt
   ```

3. Restart ComfyUI

The node will automatically download required data files (~7GB) from HuggingFace on first use. Additional models for the advanced node will be downloaded on demand.

## Usage

### Basic Node
1. Find the "Danbooru FAISS Lookup" node under the "conditioning/danbooru" category
2. Connect positive and negative CLIP Text Encode nodes to the conditioning inputs
3. Configure optional parameters:
   - `selected_ratings`: Comma-separated list of ratings to include (default: "General,Sensitive")
   - `n_neighbours`: Number of similar images to search (default: 5)
   - `api_username`: Optional Danbooru API username
   - `api_key`: Optional Danbooru API key

The node outputs a Danbooru post ID as a string.

### Advanced Node
1. Find the "Danbooru FAISS Lookup (Advanced)" node under the "conditioning/danbooru" category
2. Select your input mode:
   - `conditioning`: Connect CLIP conditioning (like basic node)
   - `tags_and_images`: Connect images and/or enter text tags
   - `hybrid`: Use any combination of inputs
3. Connect/configure inputs based on mode:
   - **Images**: Connect image outputs from LoadImage, VAE Decode, etc.
   - **Tags**: Enter comma-separated Danbooru tags (e.g., "1girl, white_hair, blue_eyes")
   - **Conditioning**: Connect CLIP Text Encode outputs
4. Optional: Select embedding model (CLIP or SigLIP) for tag encoding

The advanced node outputs:
- `danbooru_id`: Best matching post ID
- `all_ids`: Comma-separated list of all matching IDs
- `similarity_scores`: Comma-separated similarity scores

## Data Source

This node uses embeddings and indices from the [Danbooru2022 Embeddings Playground](https://huggingface.co/spaces/SmilingWolf/danbooru2022_embeddings_playground) by SmilingWolf.

## Requirements

- ComfyUI
- Python 3.8+
- ~7GB disk space for FAISS index files
- Additional space for optional models:
  - WD14 ConvNext model: ~400MB
  - CLIP/SigLIP models: ~50MB each

### Dependencies
Core dependencies (auto-installed):
- `faiss-cpu`: FAISS similarity search
- `pandas`, `numpy`: Data handling
- `tqdm`: Progress bars
- `requests`: File downloads

Optional dependencies for advanced features:
- `onnxruntime` or `onnxruntime-gpu`: WD14 image tagging
- `huggingface-hub`: Model downloads
- `Pillow`: Image processing
- `jax`, `flax`: CLIP/SigLIP tag encoding (optional, will fallback if not available)

## Troubleshooting

### Installation Issues
- If auto-install fails, manually run: `pip install -r requirements.txt` in the node directory
- For CUDA users: Install `onnxruntime-gpu` instead of `onnxruntime` for better performance
- On Mac: JAX may have compatibility issues; the node will work without tag encoding features

### Download Issues
- The node automatically downloads large files on first use
- If downloads fail, check your internet connection and try again
- Files are cached after successful download

### Memory Issues
- The FAISS index requires ~6-7GB RAM when loaded
- Close other applications if you encounter out-of-memory errors

## Examples

### Finding similar anime artwork
1. Use "Danbooru FAISS Lookup (Advanced)" in `tags_and_images` mode
2. Load your reference image
3. Add descriptive tags like "1girl, fantasy, detailed_background"
4. The node will find similar Danbooru posts

### Exploring CLIP prompt variations
1. Use basic node or advanced node in `conditioning` mode
2. Connect CLIP text prompts describing the style/content you want
3. Use negative prompts to exclude unwanted elements
4. Check returned Danbooru IDs for inspiration

## License

This project is released under the MIT License. 