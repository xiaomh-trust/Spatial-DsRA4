import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Union
from einops import rearrange

from vggt.heads.utils import create_uv_grid, position_grid_to_embed


def _interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)

def _apply_pos_embed(x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
    """
    Apply positional embedding to tensor x.
    """
    patch_w = x.shape[-1]
    patch_h = x.shape[-2]
    pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
    pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
    pos_embed = pos_embed * ratio
    pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
    return x + pos_embed

def interpolate_pooling(hidden, patch_hw, img_hw, reference, pooling_func, use_vggt_pe):
    (patch_h, patch_w) = patch_hw
    (img_h, img_w) = img_hw
    bs, N, S, D = hidden.shape
    # [Spatial-DsRA v4]: Explicit per-view square VLA grid avoids scale-factor rounding errors.
    import math
    tokens_per_view, remainder = divmod(reference.shape[1], N)
    side = math.isqrt(tokens_per_view)
    if remainder or side * side != tokens_per_view or S != patch_h * patch_w:
        raise ValueError("Teacher patch grid and per-view VLA token grid do not match")

    _hidden = hidden.permute(0, 1, 3, 2)
    _hidden = _hidden.reshape(bs*N, D, patch_h, patch_w)
    hidden_pooling = _interpolate(
        _hidden, size=(side, side), mode=pooling_func, align_corners=True
    )
    # [Spatial-DsRA v4]: Y = bilinear(Z) + E on the destination visual grid.
    if use_vggt_pe:
        hidden_pooling = _apply_pos_embed(hidden_pooling, img_w, img_h)
    hidden_pooling = hidden_pooling.reshape(bs, N, D, -1).permute(0, 1, 3, 2).reshape(bs, -1, D)
    return hidden_pooling


def custom_pooling(hidden, patch_hw, img_hw, reference, pooling_func, use_vggt_pe):
    if pooling_func in ['bilinear']:
        return interpolate_pooling(hidden, patch_hw, img_hw, reference, pooling_func, use_vggt_pe)
    else:
        raise NotImplementedError(f"Pooling function {pooling_func} is not implemented.")

