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

### WD14 to Conditioning Node
- **Purpose**: Convert images to conditioning using WD14 embeddings
- **Input**: Any image (from LoadImage, VAE Decode, etc.)
- **Output**: CONDITIONING that can be used with lookup nodes
- **Use cases**:
  - Preprocess images once and reuse the conditioning
  - Combine multiple image embeddings using conditioning nodes
  - Create cleaner workflows with the conditioning mode

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/thavocado/comfyui-danbooru-lookup
   ```

2. Restart ComfyUI - all dependencies will be installed automatically

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
4. Optional settings:
   - **Embedding Model**: Choose CLIP or SigLIP for tag encoding
   - **HF Token**: Enter your HuggingFace token if you get authentication errors

The advanced node outputs:
- `danbooru_id`: Best matching post ID
- `all_ids`: Comma-separated list of all matching IDs
- `similarity_scores`: Comma-separated similarity scores

### WD14 to Conditioning
1. Find the "WD14 to Conditioning" node under the "conditioning/danbooru" category
2. Connect an image from LoadImage or any other image source
3. Optional: Add HuggingFace token if you get authentication errors
4. Connect the CONDITIONING output to:
   - Basic or Advanced Danbooru lookup nodes (conditioning inputs)
   - Other conditioning manipulation nodes
   - Conditioning Combine nodes to merge multiple images

Example workflow:
- LoadImage → WD14 to Conditioning → DanbooruFAISSLookup (positive_conditioning)
- Multiple images → Multiple WD14 nodes → Conditioning Average → Lookup

## Data Source

This node uses embeddings and indices from the [Danbooru2022 Embeddings Playground](https://huggingface.co/spaces/SmilingWolf/danbooru2022_embeddings_playground) by SmilingWolf.

## Requirements

- ComfyUI
- Python 3.8+
- ~7GB disk space for FAISS index files
- Additional space for optional models:
  - WD14 ConvNext model: ~400MB (auto-downloaded by dghs-imgutils)
  - CLIP/SigLIP models: ~50MB each

### Dependencies
All dependencies are installed automatically when you restart ComfyUI:
- `faiss-cpu`: FAISS similarity search
- `pandas`, `numpy`: Data handling
- `tqdm`: Progress bars
- `requests`: File downloads
- `dghs-imgutils`: WD14 image tagging (handles model downloads automatically)
- `huggingface-hub`: Model downloads
- `Pillow`: Image processing
- `jax`, `jaxlib`, `flax`: CLIP/SigLIP tag encoding

## Troubleshooting

### Installation Issues
- If dependencies fail to install, restart ComfyUI and check the console for error messages
- Make sure you have an active internet connection for downloading packages
- For CUDA users: `dghs-imgutils` will automatically use GPU acceleration if available
- On Mac: JAX installation may take longer but should complete successfully

### HuggingFace Authentication
If you get a 401 authentication error when using image inputs:
1. Get a HuggingFace token from https://huggingface.co/settings/tokens
2. Set up authentication using one of these methods:
   - **In ComfyUI**: Enter your token in the "HF Token" field of the advanced node
   - **CLI**: Run `huggingface-cli login` and enter your token
   - **Environment**: Set `export HF_TOKEN=your_token_here` (Linux/Mac) or `set HF_TOKEN=your_token_here` (Windows)

### Download Issues
- The node automatically downloads large files on first use
- If downloads fail, check your internet connection and try again
- Files are cached after successful download

### CLIP/SigLIP Model Issues
If you get errors about CLIP/SigLIP models when using tags:
- The models are hosted on HuggingFace Spaces and may fail to download
- Error: "Failed to load CLIP model. Please ensure the model files are downloaded."
- You can manually download the files:
  - CLIP: https://huggingface.co/spaces/SmilingWolf/danbooru2022_embeddings_playground/resolve/main/data/wd-v1-4-convnext-tagger-v2/clip.msgpack
  - SigLIP: https://huggingface.co/spaces/SmilingWolf/danbooru2022_embeddings_playground/resolve/main/data/wd-v1-4-convnext-tagger-v2/siglip.msgpack
- Place them in `ComfyUI/custom_nodes/comfyui-danbooru-lookup/models/clip_model/` or `models/siglip_model/`

### Memory Issues
- The FAISS index requires ~6-7GB RAM when loaded
- Close other applications if you encounter out-of-memory errors

### OpenMP Library Conflicts (Windows)
If ComfyUI crashes with an OpenMP error:
- **Error**: "Initializing libomp140.x86_64.dll, but found libiomp5md.dll already initialized"
- **Cause**: Multiple libraries (FAISS, NumPy, etc.) loading different OpenMP runtimes
- **Solution**: The node automatically sets `KMP_DUPLICATE_LIB_OK=TRUE` to handle this
- **Manual fix** (if needed): Set environment variable before starting ComfyUI:
  ```batch
  set KMP_DUPLICATE_LIB_OK=TRUE
  python main.py
  ```

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