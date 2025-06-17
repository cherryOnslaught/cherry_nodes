import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import comfy.utils           # progress bar hook


def to_tensor(img: Union[torch.Tensor, np.ndarray], device, channels: int):
    """ComfyUI IMAGE → torch.float32 tensor on <device>, shape [H,W] or [H,W,3]."""
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img).float()
    elif isinstance(img, torch.Tensor):
        t = img.detach().float()
    else:
        raise TypeError("expected torch.Tensor or np.ndarray")

    # Unify layout
    if t.dim() == 4 and t.shape[0] == 1:           # [1,H,W,C] → [H,W,C]
        t = t.squeeze(0)
    if t.dim() == 3 and t.shape[0] in (1, 3):      # [C,H,W]  → [H,W,C]
        t = t.permute(1, 2, 0)

    if channels == 1 and t.dim() == 3:             # rgb → luma
        t = t.mean(-1)
    if channels == 3:
        if t.dim() == 2:                           # grey → rgb
            t = t.unsqueeze(-1).repeat(1, 1, 3)
        elif t.shape[-1] == 1:
            t = t.repeat(1, 1, 3)

    return t.to(device)


def bilateral(img, guide, sigma_s=2, sigma_r=0.05):
    """Fast bilateral blur on a single-channel [H,W] tensor."""
    k = sigma_s * 2 + 1
    grid = torch.arange(-sigma_s, sigma_s + 1, device=img.device)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    g_spatial = torch.exp(-(xx**2 + yy**2) / (2 * sigma_s**2))

    pad = sigma_s
    ip = F.pad(img.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="reflect")[0, 0]
    gp = F.pad(guide.unsqueeze(0).unsqueeze(0), (pad, pad, pad, pad), mode="reflect")[0, 0]

    out  = torch.zeros_like(img)
    norm = torch.zeros_like(img)

    for dy in range(k):
        for dx in range(k):
            w_s = g_spatial[dy, dx]
            patch_g = gp[dy:dy + img.shape[0], dx:dx + img.shape[1]]
            w_r = torch.exp(-((guide - patch_g) ** 2) / (2 * sigma_r ** 2))
            w = w_s * w_r
            patch_i = ip[dy:dy + img.shape[0], dx:dx + img.shape[1]]
            out  += patch_i * w
            norm += w
    return out / norm.clamp_min_(1e-6)


class AccurateAO_HBAO:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map":  ("IMAGE",),
                "base_color_map": ("IMAGE",),
                "normal_map": ("IMAGE",),
            },
            "optional": {
                "sample_count":  ("INT",   {"default": 64, "min": 4,  "max": 64}),
                "radius_px":     ("INT",   {"default": 128, "min": 4,  "max": 256}),
                "strength":      ("FLOAT", {"default": 2, "min": 0.1, "max": 4.0, "step": 0.01}),
                "height_bias":   ("FLOAT", {"default": 0.05, "min": 0.00, "max": 1.0, "step": 0.01}),
                "bilateral_blur":("BOOLEAN", {"default": True}),
                "seamless":      ("BOOLEAN", {"default": True}),
                "use_cpu":       ("BOOLEAN", {"default": False}),
                "blur_source": (   
                    [
                        "Color map", 
                        "Depth map", 
                        "Normal map (Z Axis)"
                    ],
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "hbao"
    CATEGORY     = "Cherry Nodes / Imaging"

    def hbao(
        self,
        depth_map, 
        base_color_map, 
        normal_map,
        sample_count      = 24,
        radius_px         = 32,
        strength          = 1.5,
        height_bias       = 0.05,
        bilateral_blur    = True,
        seamless          = True,
        use_cpu           = False,
        blur_source      = "Color Map",
    ):
        device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

        depth   = to_tensor(depth_map,  device, 1)      # [H,W]
        normals = to_tensor(normal_map, device, 3)      # [H,W,3]

        # Normalise depth to 0-1
        min_d, max_d = depth.amin(), depth.amax()
        depth = (depth - min_d) / (max_d - min_d + 1e-6)

        H, W  = depth.shape
        steps = torch.arange(1, radius_px + 1, device=device)  # [S]
        thetas = torch.arange(sample_count, device=device) * (2 * math.pi / sample_count)

        n = normals * 2 - 1                                    # −1‥1 normals
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]

        occl = torch.zeros_like(depth)

        # Pre-compute base grids (broadcasted later)
        y0 = torch.arange(H, device=device)[:, None, None]     # (H,1,1)
        x0 = torch.arange(W, device=device)[None, :, None]     # (1,W,1)

        # Progress bar — one tick per angle
        pbar = comfy.utils.ProgressBar(sample_count)

        for theta in thetas:
            dx, dy = math.cos(theta), math.sin(theta)

            yi = y0 + (steps * dy).round().long()              # (H,1,S)
            xi = x0 + (steps * dx).round().long()              # (1,W,S)

            if seamless:
                yi %= H; xi %= W
            else:
                yi.clamp_(0, H - 1); xi.clamp_(0, W - 1)

            samples = depth[yi, xi]                            # (H,W,S)

            ndot = (nx * dx + ny * dy + nz * 0.2).clamp_(0.0, 1.0).unsqueeze(-1)
            samples = samples * ndot

            horizon = samples.min(-1)[0]                       # nearest blocker
            diff    = (depth - horizon + height_bias).clamp_min_(0)
            occl   += diff

            pbar.update(1)                                     # tick

        pbar.update_absolute(pbar.total)                       # snap to 100 %

        occl = (occl / sample_count).pow(1.2)                  # fall-off curve
        ao   = (occl * strength).clamp_(0, 1)                  # white = occluded
        ao  *= (1 - height_bias * depth)                       # keep peaks bright

        if bilateral_blur:
            if blur_source == "Color Map":
                guide = to_tensor(base_color_map, device, 3)[..., 0]
            elif blur_source == "Depth Map":
                guide = depth        # already normalised
            else:  # Normal map (Z Axis)
                guide = (normals[..., 2] * 0.5 + 0.5)  # map −1…1 → 0…1
            ao = bilateral(ao, guide)

        img = ao.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0)    # [1,H,W,3]
        return (img,)


NODE_CLASS_MAPPINGS        = {"AccurateAO_HBAO": AccurateAO_HBAO}
NODE_DISPLAY_NAME_MAPPINGS = {"AccurateAO_HBAO": "Cherry Ambient Occlusion"}