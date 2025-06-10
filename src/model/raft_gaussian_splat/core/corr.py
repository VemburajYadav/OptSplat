from itertools import repeat

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from .utils.utils import bilinear_sampler, coords_grid
from ...encoder.costvolume.depth_predictor_multiview import prepare_feat_proj_data_lists, warp_with_pose_depth_candidates

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CostVolumeBlock:
    def __init__(self, features, intrinsics, extrinsics,
                 near, far, num_levels=4, radius=4, num_depth_candidates=512):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.near = near
        self.far = far
        self.num_depth_candidates = num_depth_candidates

        b, v, c, h, w = features.shape
        
        # prepare features and camera parameters for warping
        feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = (
            prepare_feat_proj_data_lists(
                features,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=self.num_depth_candidates,
            )
        )

        # sample feat01 from feat10 via camera projection
        feat01, feat10 = feat_comb_lists
        feat01_warped = warp_with_pose_depth_candidates(
            feat10,
            intr_curr,
            pose_curr_lists[0],
            1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
            warp_padding_mode="zeros",
        )  # [B, C, D, H, W]

        # calculate similarity
        corr = (feat01.unsqueeze(2) * feat01_warped).sum(1) / (c ** 0.5)  # [vB, D, H, W]
        corr = rearrange(corr, "vb d h w -> (vb h w) 1 d")

        # cost-volume at different levels
        self.corr_pyramid = [corr.view(v * b * h * w, 1, 1, self.num_depth_candidates)]
        for i in range(self.num_levels - 1):
            corr = F.avg_pool1d(corr, kernel_size=2, stride=2)
            num_depth_candidates_i = int(self.num_depth_candidates / (2**(i+1)))
            self.corr_pyramid.append(corr.view(v * b * h * w, 1, 1, num_depth_candidates_i))

        # pixel-wise near and far depth values
        self.near = near.view(v * b, 1, 1, 1).repeat(1, 1, h, w)
        self.far = far.view(v * b, 1, 1, 1).repeat(1, 1, h, w)

        # disparity range
        self.disparity_near = 1 / self.near
        self.disparity_far = 1 / self.far

    def __call__(self, disparities):
        vb, _, h, w = disparities.shape
        r = self.radius

        # normalize the disparities
        disparities_normalised = (disparities - self.disparity_far) / (self.disparity_near - self.disparity_far)

        # cost volume lookups
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]

            # index the cost volume using normalised disparities
            num_depth_candidates_i = int(self.num_depth_candidates / (2**i))
            indices = disparities_normalised * (num_depth_candidates_i - 1)
            indices = rearrange(indices, "vb 1 h w -> (vb h w) 1 1 1").repeat(1, 1, 2 * r + 1, 1)

            # lookup coordinates to index cost volume in the neighborhood of the predicted disparity
            di = torch.linspace(-r, r, 2 * r + 1, device=disparities.device)
            di = di.view(1, 1, -1, 1).repeat(vb * h * w, 1, 1, 1)
            coords_x = indices + di
            coords_y = torch.zeros_like(coords_x)
            coords = torch.cat([coords_x, coords_y], dim=-1)

            # sample the correlation values
            out = bilinear_sampler(corr, coords)
            out = rearrange(out, "(vb h w) 1 1 n -> vb n h w", h=h, w=w)
            out_pyramid.append(out)

        out_tensor = torch.cat(out_pyramid, dim=1)
        return out_tensor


class MultiViewCostVolumeBlock:
    def __init__(self, features, intrinsics, extrinsics,
                 near, far, num_levels=4, radius=4, num_depth_candidates=512):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.near = near
        self.far = far
        self.num_depth_candidates = num_depth_candidates

        b, v, c, h, w = features.shape

        # prepare features and camera parameters for warping
        feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = (
            prepare_feat_proj_data_lists(
                features,
                intrinsics,
                extrinsics,
                near,
                far,
                num_samples=self.num_depth_candidates,
            )
        )

        # cost volume constructions
        feat01 = feat_comb_lists[0]

        raw_correlation_in_lists = []
        for feat10, pose_curr in zip(feat_comb_lists[1:], pose_curr_lists):
            # sample feat01 from feat10 via camera projection
            feat01_warped = warp_with_pose_depth_candidates(
                feat10,
                intr_curr,
                pose_curr,
                1.0 / disp_candi_curr.repeat([1, 1, *feat10.shape[-2:]]),
                warp_padding_mode="zeros",
            )  # [B, C, D, H, W]
            # calculate similarity
            raw_correlation_in = (feat01.unsqueeze(2) * feat01_warped).sum(1) / (c ** 0.5)  # [vB, D, H, W]
            raw_correlation_in_lists.append(raw_correlation_in)

        # average all cost volumes
        # corr = torch.mean(torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False)  # [vxb d, h, w]
        corr = torch.max(torch.stack(raw_correlation_in_lists, dim=0), dim=0, keepdim=False)[0]  # [vxb d, h, w]
        corr = rearrange(corr, "vb d h w -> (vb h w) 1 d")

        # cost-volume at different levels
        self.corr_pyramid = [corr.view(v * b * h * w, 1, 1, self.num_depth_candidates)]
        for i in range(self.num_levels - 1):
            corr = F.avg_pool1d(corr, kernel_size=2, stride=2)
            num_depth_candidates_i = int(self.num_depth_candidates / (2 ** (i + 1)))
            self.corr_pyramid.append(corr.view(v * b * h * w, 1, 1, num_depth_candidates_i))

        # pixel-wise near and far depth values
        self.near = near.view(v * b, 1, 1, 1).repeat(1, 1, h, w)
        self.far = far.view(v * b, 1, 1, 1).repeat(1, 1, h, w)

        # disparity range
        self.disparity_near = 1 / self.near
        self.disparity_far = 1 / self.far

    def __call__(self, disparities):
        vb, _, h, w = disparities.shape
        r = self.radius

        # normalize the disparities
        disparities_normalised = (disparities - self.disparity_far) / (self.disparity_near - self.disparity_far)

        # cost volume lookups
        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]

            # index the cost volume using normalised disparities
            num_depth_candidates_i = int(self.num_depth_candidates / (2 ** i))
            indices = disparities_normalised * (num_depth_candidates_i - 1)
            indices = rearrange(indices, "vb 1 h w -> (vb h w) 1 1 1").repeat(1, 1, 2 * r + 1, 1)

            # lookup coordinates to index cost volume in the neighborhood of the predicted disparity
            di = torch.linspace(-r, r, 2 * r + 1, device=disparities.device)
            di = di.view(1, 1, -1, 1).repeat(vb * h * w, 1, 1, 1)
            coords_x = indices + di
            coords_y = torch.zeros_like(coords_x)
            coords = torch.cat([coords_x, coords_y], dim=-1)

            # sample the correlation values
            out = bilinear_sampler(corr, coords)
            out = rearrange(out, "(vb h w) 1 1 n -> vb n h w", h=h, w=w)
            out_pyramid.append(out)

        out_tensor = torch.cat(out_pyramid, dim=1)
        return out_tensor


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr_12 = corr.reshape(batch*h1*w1, dim, h2, w2)
        # corr_21 = corr.permute(0, 4, 5, 3, 1, 2).reshape(batch * h2 * w2, din, h1, w1)
        corr_21 = rearrange(corr, "b h1 w1 d h2 w2 -> (b h2 w2) d h1 w1")
        corr = torch.cat([corr_12 , corr_21], dim=0)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1, device=coords.device)
            dy = torch.linspace(-r, r, 2*r+1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), dim=-1)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd) 
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())
