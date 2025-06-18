"""
Cherry Ambient Occlusion
• Accurate Horizon-Based Ambient Occlusion (HBAO)
• Uses bilateral filtering for noise reduction
• Supports seamless textures
• Tiling support for large images
"""

import math
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import comfy.utils


def to_tensor(img: Union[torch.Tensor, np.ndarray], device, channels: int):
    
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img).float()
    elif isinstance(img, torch.Tensor):
        t = img.detach().float()
    else:
        raise TypeError("expected torch.Tensor or np.ndarray")

    if t.dim() == 4 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.dim() == 3 and t.shape[0] in (1, 3):
        t = t.permute(1, 2, 0)

    if channels == 1 and t.dim() == 3:
        t = t.mean(-1)
    if channels == 3:
        if t.dim() == 2:
            t = t.unsqueeze(-1).repeat(1, 1, 3)
        elif t.shape[-1] == 1:
            t = t.repeat(1, 1, 3)

    return t.to(device)


def bilateral(img, guide, sigma_s=2, sigma_r=0.05):
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


def run_on_tiles(
    kernel,
    depth, normals, base_color,
    tile       = 1024,
    overlap    = 32,
    **kargs
):
    H, W = depth.shape
    stride  = tile
    margin  = overlap

    ao_acc  = torch.zeros((H, W), dtype=depth.dtype, device=depth.device)
    weights = torch.zeros_like(ao_acc)

    n_tiles_y = (H + stride - 1) // stride
    n_tiles_x = (W + stride - 1) // stride
    total_tiles = n_tiles_y * n_tiles_x
    pbar = comfy.utils.ProgressBar(total_tiles)

    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y0 + tile, H)
            x1 = min(x0 + tile, W)

            yp0 = y0 - margin
            yp1 = y1 + margin
            xp0 = x0 - margin
            xp1 = x1 + margin

            y_idx = torch.arange(yp0, yp1) % H
            x_idx = torch.arange(xp0, xp1) % W
            d_crop = depth[y_idx][:, x_idx]
            n_crop = normals[y_idx][:, x_idx, ...]
            c_crop = base_color[y_idx][:, x_idx, ...]

            ao_crop = kernel(d_crop, c_crop, n_crop, **kargs, pbar=pbar)\
                        .squeeze(0)[..., 0]       

            h, w = ao_crop.shape
            mask = torch.ones_like(ao_crop)
            if margin:
                yy = torch.linspace(0, 1, h, device=ao_crop.device).unsqueeze(1)
                xx = torch.linspace(0, 1, w, device=ao_crop.device).unsqueeze(0)
                border_y = torch.minimum(yy, 1 - yy)
                border_x = torch.minimum(xx, 1 - xx)
                feather  = torch.minimum(border_x, border_y)
                inner_ratio = tile / (tile + 2 * margin)
                mask *= (feather / inner_ratio).clamp(0, 1)

            ao_acc[y_idx[:, None], x_idx] += ao_crop * mask
            weights[y_idx[:, None], x_idx] += mask

            pbar.update(1)

    pbar.update_absolute(pbar.total)
    return (ao_acc / weights.clamp_min_(1e-6))\
        .unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0)

class AccurateAO_HBAO:
    TILE_SIZE   = 1024
    TILE_OVERLP = 32

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth_map":      ("IMAGE",),
                "base_color_map": ("IMAGE",),
                "normal_map":     ("IMAGE",),
            },
            "optional": {
                "sample_count":  ("INT",   {"default": 24,  "min": 4,  "max": 64}),
                "radius_px":     ("INT",   {"default": 32,  "min": 4,  "max": 128}),
                "strength":      ("FLOAT", {"default": 1.5, "min": 0.1,"max": 4.0,"step":0.01}),
                "height_bias":   ("FLOAT", {"default": 0.05,"min": 0.0,"max": 1.0,"step":0.01}),
                "bilateral_blur":("BOOLEAN", {"default": True}),
                "seamless_textures_input": ("BOOLEAN", {"default": True}),
                "use_cpu":       ("BOOLEAN", {"default": False}),
                "blur_source":   ([
                    "Color Map",
                    "Depth Map",
                    "Normal Map (Z Axis)"
                ],),
                "tile_size": ([
                    "No Tiling",
                    384, 512, 1024
                ], {"default": "No Tiling"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION     = "hbao"
    CATEGORY     = "Cherry Nodes / Imaging"

    def hbao(
        self,
        depth_map, base_color_map, normal_map,
        sample_count      = 24,
        radius_px         = 32,
        strength          = 1.5,
        height_bias       = 0.05,
        bilateral_blur    = True,
        seamless_textures_input = True,
        use_cpu           = False,
        blur_source       = "Color Map",
        tile_size         = "No Tiling",
    ):
        device = "cpu" if use_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

        depth   = to_tensor(depth_map,  device, 1)
        normals = to_tensor(normal_map, device, 3)
        color   = to_tensor(base_color_map, device, 3)

        def _kernel(d, c, n, pbar=None):
            return self._hbao_single_tile(
                d, c, n,
                sample_count=sample_count, radius_px=radius_px,
                strength=strength, height_bias=height_bias,
                bilateral_blur=bilateral_blur, blur_source=blur_source,
                seamless=seamless_textures_input,
                pbar=pbar
            )

        if tile_size == "No Tiling":
            img = _kernel(depth, color, normals)
        else:
            tile_size_int = int(tile_size)
            min_tile = 384
            if tile_size_int < min_tile:
                tile_size_int = min_tile
            overlap = max(radius_px, 32)
            if radius_px * 2 >= tile_size_int:
                raise ValueError(f"AO radius ({radius_px}) is too large for the selected tile size ({tile_size_int}). Please choose a larger tile size or reduce the radius.")
            else:
                img = run_on_tiles(
                    _kernel, depth, normals, color,
                    tile=tile_size_int, overlap=overlap
                )
        return (img,)

    def _hbao_single_tile(
        self,
        depth, base_color, normals,
        sample_count, radius_px,
        strength, height_bias,
        bilateral_blur, blur_source,
        seamless,
        pbar=None
    ):
        device = depth.device

        depth = (depth - depth.amin()) / (depth.amax() - depth.amin() + 1e-6)

        H, W  = depth.shape
        steps = torch.arange(1, radius_px + 1, device=device)   # [S]
        thetas = torch.arange(sample_count, device=device) * (2 * math.pi / sample_count)

        n = normals * 2 - 1                                    
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]

        occl = torch.zeros_like(depth)

        y0 = torch.arange(H, device=device)[:, None, None]
        x0 = torch.arange(W, device=device)[None, :, None]

        if pbar is not None:
            update_pbar = False
        else:
            pbar = comfy.utils.ProgressBar(len(thetas))
            update_pbar = True

        for theta in thetas:
            dx, dy = math.cos(theta), math.sin(theta)

            yi = y0 + (steps * dy).round().long()
            xi = x0 + (steps * dx).round().long()

            if seamless:
                yi %= H; xi %= W
            else:
                yi.clamp_(0, H - 1); xi.clamp_(0, W - 1)

            samples = depth[yi, xi]                          

            ndot = (nx * dx + ny * dy + nz * 0.2).clamp_(0.0, 1.0).unsqueeze(-1)
            samples = samples * ndot

            horizon = samples.min(-1)[0]
            diff    = (depth - horizon + height_bias).clamp_min_(0)
            occl   += diff

            if update_pbar:
                pbar.update(1)

        if update_pbar:
            pbar.update_absolute(pbar.total)

        occl = (occl / sample_count).pow(1.2)
        ao   = (occl * strength).clamp_(0, 1)
        ao  *= (1 - height_bias * depth)

        if bilateral_blur:
            if blur_source == "Color Map":
                guide = base_color[..., 0]
            elif blur_source == "Depth Map":
                guide = depth
            else:  # Normal map (Z Axis)
                guide = (normals[..., 2] * 0.5 + 0.5)
            ao = bilateral(ao, guide)

        return ao.unsqueeze(-1).repeat(1, 1, 3).unsqueeze(0)


NODE_CLASS_MAPPINGS        = {"AccurateAO_HBAO": AccurateAO_HBAO}
NODE_DISPLAY_NAME_MAPPINGS = {"AccurateAO_HBAO": "Cherry Ambient Occlusion"}
