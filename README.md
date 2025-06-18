![Cherry Logo](media/cherry_logo.png)
# Cherry Nodes for ComfyUI
## Overview
Cherry Nodes is a collection of custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

---

## Installation

1. **Download or Clone** this repository into your ComfyUI custom nodes directory:
   ```bash
   git clone <this-repo-url> ComfyUI/custom_nodes/cherry_nodes
   # or manually copy the 'cherry_nodes' folder into ComfyUI/custom_nodes/
   ```

2. **Restart ComfyUI** to load the new nodes.

---

## Cherry Ambient Occlusion

### What It Does
- High-quality, tileable ambient occlusion map generation from texture maps:
  - **Color Map** (Albedo)
  - **Depth Map**
  - **Normal Map**
- Supports seamless texture processing.
- Handles large images (e.g., 4K) by processing in tiles, with automatic overlap and blending to avoid seams.

### How It Works
- Uses a Horizon-Based Ambient Occlusion (HBAO) algorithm, sampling the depth and normal maps in multiple directions to estimate occlusion.
- Processes images in tiles with overlap and feathered blending to avoid visible seams.
- Dynamically adjusts tile overlap to match the AO radius, and prevents settings that would cause artifacts.

### Node Inputs
- **Depth Map**: The depth texture (grayscale or RGB).
- **Base Color Map**: The albedo/color texture.
- **Normal Map**: The normal map (RGB, tangent space).
- **Sample Count**: Number of AO directions (quality vs. speed).
- **Radius (px)**: AO sampling radius in pixels.
- **Strength**: AO intensity.
- **Height Bias**: AO height bias (for fine-tuning contact shadows).
- **Bilateral Blur**: Smooths AO using a bilateral filter.
- **Seamless Textures Input**: Set to `True` if your textures are seamless/tileable.
- **Use CPU**: Forces computation on CPU.
- **Blur Source**: Which map to use as the guide for bilateral blur.
- **Tile Size**: Optional tiling supported (384, 512, 1024) for low GPU memory situations. The node will warn and stop if the AO radius is too large for the selected tile size.

### Node Output
- **Ambient Occlusion Map**: An AO map matching the input resolution.

---

## Usage Tips
- To maintain the seamlessness of seamless texture inputs in the Generated ambient occlusion output, enable **Seamless Textures Input**.
- For large AO radii, use larger tile sizes to avoid artifacts.
- If you see a warning about tile size vs. radius, increase the tile size or reduce the radius.

---

## Example Workflow
1. Add the **Cherry Ambient Occlusion** node to your ComfyUI workflow.
2. Connect your Color, Depth, and Normal maps to the node.
3. Adjust the AO parameters as needed.
4. Run the workflow to generate a high-quality AO map.

---

## Support & License
- This project is open source and provided as-is.
- For issues or feature requests, please open an issue on the repository.

---

