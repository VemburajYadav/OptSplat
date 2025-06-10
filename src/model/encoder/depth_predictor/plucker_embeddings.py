import torch
import os

from einops import rearrange, repeat

from ..backbone.unimatch.geometry import coords_grid

def generate_plucker_embeddings(intrinsics, extrinsics, height, width):
    # pixel coordinates
    vb, _, _ = intrinsics.shape

    with torch.no_grad():
        grid = coords_grid(vb, height, width, homogeneous=True, device=extrinsics.device)  # [B, 3, H, W]

        intrinsics = intrinsics.clone().detach()
        extrinsics = extrinsics.clone().detach()

        intrinsics[:, 0, :] *= float(width)
        intrinsics[:, 1, :] *= float(height)

        # back project to 3D
        points = torch.inverse(intrinsics).bmm(grid.view(vb, 3, -1))  # [B, 3, H*W]

        # compute the direction vectors for Plucker coordinates
        # directions = torch.bmm(torch.transpose(extrinsics[:, :3, :3], 1, 2), points) # poses in w2c format
        directions = torch.bmm(extrinsics[:, :3, :3], points) # poses in c2w format

        # normalize the direction vectors
        directions = directions / directions.norm(dim=1, keepdim=True)

        # compute the moment vectors for Plucker coordinates
        # camera_origins = -torch.bmm(torch.transpose(extrinsics[:, :3, :3], 1, 2), extrinsics[:, :3, 3:]) # poses in w2c format
        camera_origins = extrinsics[:, :3, 3:] # poses in c2w format
        camera_origins = camera_origins.repeat(1, 1, height * width)
        moments = torch.linalg.cross(camera_origins, directions, dim=1)

        # form plucker embeddings from ray directions and moments
        plucker_embeddings = torch.cat([directions, moments], dim=1)
        plucker_embeddings = rearrange(plucker_embeddings, "vb c (h w) -> vb c h w", h=height)

    return plucker_embeddings



