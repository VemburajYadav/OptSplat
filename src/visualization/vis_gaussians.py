from pathlib import Path
from random import randrange
import numpy as np
import torch
import torch.nn.functional as F
from pyparsing import opAssoc
from torch import Tensor
from einops import einsum, rearrange, repeat
from jaxtyping import Float

from .color_map import apply_color_map
from .layout import add_border, hcat, vcat
from .annotation import add_label


def box(
    image: Float[Tensor, "3 height width"],
) -> Float[Tensor, "3 new_height new_width"]:
    return add_border(add_border(image), 1, 0)

def visualize_gaussians(
        context_images: Float[Tensor, "batch view 3 height width"],
        opacities: Float[Tensor, "batch vrspp"],
        covariances: Float[Tensor, "batch vrspp 3 3"],
        colors: Float[Tensor, "batch vrspp 3"],
        gaussian_sampling_resolution: int,
) -> Float[Tensor, "3 vis_height vis_width"]:
    b, v, _, h, w = context_images.shape
    hg, wg = h // gaussian_sampling_resolution, w // gaussian_sampling_resolution

    rb = randrange(b)
    context_images = context_images[rb]
    opacities = repeat(
        opacities[rb], "(v h w spp) -> spp v c h w", v=v, c=3, h=hg, w=wg
    )

    colors = rearrange(colors[rb], "(v h w spp) c -> spp v c h w", v=v, h=hg, w=wg)

    # Color-map Gaussian covariawnces.
    det = covariances[rb].det()
    det = rearrange(det, "(v h w spp) -> (spp v) 1 h w", v=v, h=hg, w=wg)
    det = F.interpolate(det, size=(h, w), mode="bilinear", align_corners=True)
    det = rearrange(det, "(spp v) c h w -> (v c h w spp)", v=v, h=h, w=w, c=1)
    det = apply_color_map(det / det.max(), "inferno")
    det = rearrange(det, "(v h w spp) c -> spp v c h w", v=v, h=hg, w=wg)

    # upsample the images
    x_up = []
    for x in (opacities, colors):
        x = rearrange(x, "spp v c h w -> (spp v) c h w")
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        x = rearrange(x, "(spp v) c h w -> spp v c h w", v=v)
        x_up.append(x)

    opacities, colors = x_up

    return add_border(
        hcat(
            add_label(box(hcat(*context_images)), "Context"),
            add_label(box(vcat(*[hcat(*x) for x in opacities])), "Opacities"),
            add_label(
                box(vcat(*[hcat(*x) for x in (colors * opacities)])), "Colors"
            ),
            add_label(box(vcat(*[hcat(*x) for x in colors])), "Colors (Raw)"),
            add_label(box(vcat(*[hcat(*x) for x in det])), "Determinant"),
        )
    )