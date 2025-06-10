from dataclasses import dataclass
from pyexpat import features
from collections import OrderedDict

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from networkx.algorithms.bipartite.basic import density

from ....geometry.projection import sample_image_grid
from ...encoder.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from ...encoder.backbone import BackboneMultiview
from .update import BasicUpdateBlock, SmallUpdateBlock, BasicUpdateBlockWoMask, BasicUpdateBlockSingleHead
from .extractor import BasicEncoder, SmallEncoder, ResidualDownBlock, ResNetEncoder, GaussianUpsampler
from .corr import CorrBlock, AlternateCorrBlock, CostVolumeBlock, MultiViewCostVolumeBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8
from ...types import Gaussians
from ....global_cfg import get_cfg

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int

@dataclass
class RaftCfg:
    dropout: float
    alternate_corr: bool
    corr_levels: int
    corr_radius: int
    small: bool
    iters: int
    mixed_precision: bool
    num_surfaces: int
    num_depth_candidates: int
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    gaussian_sampling_resolution: int
    corr_resolution: int
    optimization_search_space_dim: int
    wo_convex_upsampling: bool
    unet_backbone: bool
    unimatch_weights_path: str | None
    wo_backbone_cross_attn: bool
    use_epipolar_trans: bool
    multiview_trans_attn_split: int
    disparity_init: str
    near_depth_plane: float
    far_depth_plane: float
    scale_disp_update: float
    scale_density_update: float
    scale_gaussian_update: float
    upsampler_normalization: str
    context_based_upsampling_iters: str
    scale_multiplier_up_mask: float

class RAFT(nn.Module):
    def __init__(self, cfg: RaftCfg):
        super(RAFT, self).__init__()
        self.cfg = cfg

        if cfg.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128

        # parameters for raw gaussians
        # feature network, context network, and update block
        if cfg.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=cfg.ropout)
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=cfg.dropout)
            self.update_block = SmallUpdateBlock(self.cfg, hidden_dim=hdim)
        else:
            if cfg.unet_backbone:
                self.backbone = self.get_backbone(feature_dim=128)
                self.fnet = self.get_feature_encoder_from_backbone_features(input_dim=128, output_dim=128)
                self.cnet = self.get_feature_encoder_from_backbone_features(input_dim=256, output_dim=256)
            else:
                self.fnet = BasicEncoder(output_dim=128, norm_fn='instance', dropout=cfg.dropout, fmap_res=cfg.corr_resolution)
                self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=cfg.dropout, fmap_res=cfg.corr_resolution)

            self.gaussian_adapter = GaussianAdapter(self.cfg.gaussian_adapter)
            self.corr_resolution = cfg.corr_resolution
            self.raw_gaussian_channels = 3 * ((self.cfg.gaussian_adapter.sh_degree + 1)**2) + 3 + 4

            self.update_block_no_mask = BasicUpdateBlockSingleHead(self.cfg, hidden_dim=hdim, gru_res=cfg.corr_resolution,
                                                               output_dim=128, upsample_res=cfg.gaussian_sampling_resolution)

            # self.update_block = BasicUpdateBlock(self.cfg, hidden_dim=hdim, gru_res=cfg.corr_resolution)

            self.num_upsampling_levels = int(math.log2(self.corr_resolution))
            self.context_channels_up = [32, 64, 128] if self.num_upsampling_levels == 2 else [32, 64, 128, 256]
            self.context_encoder_up = ResNetEncoder(in_channels=3, channel_dims=self.context_channels_up,
                                                    activation="gelu", normalization="group")
            self.gaussian_upsampler = GaussianUpsampler(input_dim=128, output_dim=128,
                                                        context_dims_up=self.context_channels_up,
                                                        scale_factor=self.corr_resolution,
                                                        normalization=None if self.cfg.upsampler_normalization == "none" else "group")

            gaussian_head_in_channels = 0
            density_head_in_channels = 0
            disparity_head_in_channels = 0
            if self.cfg.context_based_upsampling_iters == "last":
                gaussian_head_in_channels = self.raw_gaussian_channels
                density_head_in_channels = 1
                disparity_head_in_channels = 1

            self.gaussian_head = nn.Sequential(
                nn.Conv2d(128 + gaussian_head_in_channels, 128, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 128) if self.cfg.upsampler_normalization == "group" else nn.Sequential(),
                nn.GELU(),
                nn.Conv2d(128, self.raw_gaussian_channels, kernel_size=1, stride=1, padding=0)
            )

            self.disparity_head = nn.Sequential(
                nn.Conv2d(128 + density_head_in_channels, 128, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 128) if self.cfg.upsampler_normalization == "group" else nn.Sequential(),
                nn.GELU(),
                nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
            )

            self.density_head = nn.Sequential(
                nn.Conv2d(128 + disparity_head_in_channels, 128, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, 128) if self.cfg.upsampler_normalization == "group" else nn.Sequential(),
                nn.GELU(),
                nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            )

            if self.cfg.context_based_upsampling_iters == "last":
                self.gaussian_head_iter = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(128, self.raw_gaussian_channels, kernel_size=1, stride=1, padding=0)
                )

                self.disparity_head_iter = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
                )

                self.density_head_iter = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
                )

        self.scale_disp_update = self.cfg.scale_disp_update
        self.scale_density_update = self.cfg.scale_density_update
        self.scale_gaussian_update = self.cfg.scale_gaussian_update
        self.search_space_dim = self.cfg.optimization_search_space_dim
        self.convex_upsampling = True if (self.cfg.context_based_upsampling_iters == "last") else False

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        #         if m.weight is not None:
        #             nn.init.constant_(m.weight, 1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)


    def get_backbone(self, feature_dim=128):
        backbone = BackboneMultiview(
            feature_channels=feature_dim,
            downscale_factor=4,
            no_cross_attn=self.cfg.wo_backbone_cross_attn,
            use_epipolar_trans=self.cfg.use_epipolar_trans,
        )
        ckpt_path = self.cfg.unimatch_weights_path
        if get_cfg().mode == 'train':
            if self.cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {
                        k: v
                        for k, v in unimatch_pretrained_model.items()
                        if k in backbone.state_dict()
                    }
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not self.cfg.wo_backbone_cross_attn
                backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

        return backbone

    def get_feature_encoder_from_backbone_features(self, input_dim=128, output_dim=256):
        fnet = ResidualDownBlock(in_channels=input_dim, out_channels=output_dim, norm_channels=8, num_layers=3,
                                 downsample=True if self.cfg.corr_resolution == 8 else False,
                                 activation="gelu", normalization="group")

        return fnet

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, b, h, w, device):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        coords0 = coords_grid(b, h, w, device=device)
        coords1 = coords_grid(b, h, w, device=device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def initialize_disaprity(self, disparity_near, disparity_far, h, w):
        b, v = disparity_near.shape

        if self.cfg.disparity_init == "near":
            disparity_init = disparity_near.view(v * b, 1, 1, 1).repeat(1, 1, h, w)
        elif self.cfg.disparity_init == "far":
            disparity_init = disparity_far.view(v * b, 1, 1, 1).repeat(1, 1, h, w)
        elif self.cfg.disparity_init == "center":
            disparity_avg = (disparity_near + disparity_far) * 0.5
            disparity_init = disparity_avg.view(v * b, 1, 1, 1).repeat(1, 1, h, w)
        elif self.cfg.disparity_init == "random":
            disparity_init = torch.rand((v * b, 1, h, w), dtype=disparity_near.dtype, device=disparity_near.device)
            disparity_near = disparity_near.view(v * b, 1, 1, 1).repeat(1, 1, h, w)
            disparity_far = disparity_far.view(v * b, 1, 1, 1).repeat(1, 1, h, w)
            disparity_init = disparity_far + disparity_init * (disparity_near - disparity_far)
        else:
            raise ValueError("Unsupported argument type '{}' for dispsrity_init".format(self.cfg.disparity_init))

        return disparity_init

    def upsample_tensor(self, inp, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, C, H, W = inp.shape
        scale = self.corr_resolution // self.cfg.gaussian_sampling_resolution

        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_inp = F.unfold(inp, [3,3], padding=1)
        up_inp = up_inp.view(N, C, 9, 1, 1, H, W)

        up_inp = torch.sum(mask * up_inp, dim=2)
        up_inp = up_inp.permute(0, 1, 4, 2, 5, 3)

        return up_inp.reshape(N, C, scale*H, scale*W)

    def map_pdf_to_opacity(self, pdf, global_step):
        # https://www.desmos.com/calculator/opvwti3ba9
        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def convert_to_gaussians(self, depths, densities, raw_gaussians, extrinsics, intrinsics, h, w):
        b, v, _, _ = raw_gaussians.shape

        # rays passing through pixels
        xy_ray, _ = sample_image_grid((h, w), raw_gaussians.device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        xy_ray = repeat(xy_ray, "hw c xy -> b v hw c xy", b=b, v=v)

        gaussians = rearrange(raw_gaussians,"... (srf c) -> ... srf c", srf=1)

        gaussians = self.gaussian_adapter.forward(
            rearrange(extrinsics, "b v i j -> b v () () () i j"),
            rearrange(intrinsics, "b v i j -> b v () () () i j"),
            rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
            depths,
            densities,
            rearrange(gaussians,"b v r srf c -> b v r srf () c"),
            (h, w),
        )

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = 1

        visualization_dump = {}
        visualization_dump["depth"] = depths.clone().detach()
        visualization_dump["scales"] = rearrange(
            gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
        )
        visualization_dump["rotations"] = rearrange(
            gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
        )

        return Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                opacity_multiplier * gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        ), visualization_dump

    def get_indexing_coords(self, disparities, intrinsics, extrinsics):
        # pixel coordinates

        grid = coords_grid(
            b, h, w, depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]


    def forward(self, images, intrinsics, extrinsics, near, far, iters=12, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # image size
        b, v, c, h_img, w_img = images.shape

        h = h_img // self.corr_resolution
        w = w_img // self.corr_resolution

        h_g = h_img // self.cfg.gaussian_sampling_resolution
        w_g = w_img // self.cfg.gaussian_sampling_resolution

        hdim = self.hidden_dim
        cdim = self.context_dim

        near = torch.tensor([self.cfg.near_depth_plane], dtype=near.dtype, device=near.device).view(1, 1).repeat(b, v)
        far = torch.tensor([self.cfg.far_depth_plane], dtype=far.dtype, device=near.device).view(1, 1).repeat(b, v)

        # only works with 2 context views
        if self.cfg.unet_backbone:
            trans_features, cnn_features = self.backbone(
                images,
                attn_splits=self.cfg.multiview_trans_attn_split,
                return_cnn_features=True,
                epipolar_kwargs=None,
            )

            trans_cnn_features_cat = torch.cat([trans_features, cnn_features], dim=2)
            trans_cnn_features_cat = rearrange(trans_cnn_features_cat, "b v c h w -> (v b) c h w")

            fnet_in = rearrange(trans_features, "b v c h w -> (v b) c h w")
            features = self.fnet(fnet_in)
            features = rearrange(features, "(v b) c h w -> b v c h w", b=b)

            cfeatures = self.cnet(trans_cnn_features_cat)
            net, inp = torch.split(cfeatures, [hdim, cdim], dim=1)
        else:
            image1, image2 = torch.split(images, [1, 1], dim=1)
            image1 = image1.squeeze(dim=1)
            image2 = image2.squeeze(dim=1)

            image1 = 2 * (image1) - 1.0
            image2 = 2 * (image2) - 1.0

            image1 = image1.contiguous()
            image2 = image2.contiguous()

            # run the feature network
            fmap1, fmap2 = self.fnet([image1, image2])
            features = torch.cat([fmap1.unsqueeze(dim=1), fmap2.unsqueeze(dim=1)], dim=1)

            # run the context network
            cnet1, cnet2 = self.cnet([image1, image2])
            net1, inp1 = torch.split(cnet1, [hdim, cdim], dim=1)
            net2, inp2 = torch.split(cnet2, [hdim, cdim], dim=1)

            net = torch.cat([net1.unsqueeze(dim=1), net2.unsqueeze(dim=1)], dim=1)
            inp = torch.cat([inp1.unsqueeze(dim=1), inp2.unsqueeze(dim=1)], dim=1)

            net = rearrange(net, "b v c h w -> (v b) c h w")
            inp = rearrange(inp, "b v c h w -> (v b) c h w")

        context_encoder_up_in = rearrange(images, "b v c h w -> (v b) c h w")
        context_encoder_up_in = 2. * context_encoder_up_in - 1.
        list_context_features_up = self.context_encoder_up(context_encoder_up_in)
        # print("Context feature levels: ", len(list_context_features_up))
        # print(f"Context features up: {[t.shape for t in list_context_features_up]}")

        if self.search_space_dim == 1:
            corr_fn = CostVolumeBlock(features, intrinsics, extrinsics, near, far,
                                      num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius,
                                      num_depth_candidates=self.cfg.num_depth_candidates)
        else:
            features1, features2 = torch.split(features, [1, 1], dim=1)
            corr_fn = CorrBlock(features1, features2, num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius)

        net = torch.tanh(net)
        inp = torch.relu(inp)

        disparities_full = self.initialize_disaprity(1 / near, 1 / far, h_g, w_g)
        densities_full = torch.zeros_like(disparities_full)

        disp_near = 1. / self.cfg.far_depth_plane
        disp_far = 1. / self.cfg.near_depth_plane

        # disparities = self.initialize_disaprity(1 / near, 1 / far, h, w)
        # densities = torch.zeros_like(disparities)

        depth_predictions = []
        density_predictions = []
        raw_gaussian_predictions = []
        gaussian_predictions = []
        visualization_dump_list = []
        stats_list = []

        for itr in range(iters):
            disparities_full = disparities_full.detach()
            densities_full = densities_full.detach()
            #
            # disparities = disparities.detach()
            # densities = densities.detach()

            disparities = F.interpolate(disparities_full, size=(h, w), align_corners=True, mode="bilinear")
            densities = F.interpolate(densities_full, size=(h, w), align_corners=True, mode="bilinear")

            # index correlation volume
            if self.search_space_dim == 1:
                corr = corr_fn(disparities)
            else:
                coords = self.get_indexing_coords(disparities, intrinsics, extrinsics)
                corr = corr_fn(coords)

            convex_upsample = (itr < (iters - 1)) and self.convex_upsampling
            with ((autocast(enabled=self.cfg.mixed_precision))):
                if convex_upsample:
                    net, out, up_mask = self.update_block_no_mask(net, inp, corr, disparities,
                                                                  densities, scale=self.cfg.scale_multiplier_up_mask)
                    gres_h, gres_w = h_g, w_g
                else:
                    net, out, _ = self.update_block_no_mask(net, inp, corr, disparities, densities)
                    gres_h, gres_w = h_img, w_img
                # net, up_mask_disp, up_mask_density, up_mask_gaussian, delta_disparities, densities, raw_gaussians = self.update_block(net, inp, corr, disparities, densities)

            # upsample predictions
            # F(t+1) = F(t) + \Delta(t)
            if convex_upsample and (self.corr_resolution != self.cfg.gaussian_sampling_resolution):
                out_up = self.upsample_tensor(out, up_mask)
            else:
                out_up = out

            if convex_upsample:
                delta_disparities_full = self.disparity_head_iter(out_up)
                densities_full = (self.scale_density_update * self.density_head_iter(out_up)).clamp(min=0.0, max=1.0)
                raw_gaussians_full = self.scale_gaussian_update * self.gaussian_head_iter(out_up)
            else:
                out_up = self.gaussian_upsampler(out, list_context_features_up)
                disparities_full = F.interpolate(disparities_full, size=(h_img, w_img), align_corners=True, mode="bilinear")
                densities_full = F.interpolate(densities_full, size=(h_img, w_img), align_corners=True, mode="bilinear")
                raw_gaussians_full = F.interpolate(raw_gaussians_full, size=(h_img, w_img), align_corners=True, mode="bilinear")

                delta_disparities_full = self.disparity_head(torch.cat([disparities_full, out_up], dim=1))
                densities_full = (self.scale_density_update * self.density_head(torch.cat([densities_full, out_up], dim=1))).clamp(min=0.0, max=1.0)
                raw_gaussians_full = self.scale_gaussian_update * self.gaussian_head(torch.cat([raw_gaussians_full, out_up], dim=1))

            disparities_full = (disparities_full + delta_disparities_full * self.scale_disp_update).clamp(min=disp_near, max=disp_far)
            depths_full = 1. / disparities_full

            depth_predictions.append(depths_full)
            density_predictions.append(densities_full)
            raw_gaussian_predictions.append(raw_gaussians_full)

            # rearrange the upsampled predictions
            densities_out = repeat(densities_full,"(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1)
            depths_out = repeat(depths_full,"(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1)
            raw_gaussians_out = rearrange(raw_gaussians_full, "(v b) c h w -> b v (h w) c", v=v, b=b)

            # print(f"itr: {itr}, depths; {depths_out.shape}, densities: {densities_out.shape}, raw_gaussians: {raw_gaussians_out.shape}")
            gaussians, visualization_dump = self.convert_to_gaussians(depths_out, densities_out, raw_gaussians_out,
                                                                      extrinsics, intrinsics, gres_h, gres_w)
            gaussian_predictions.append(gaussians)
            visualization_dump_list.append(visualization_dump)
            
            # Track statistics of disparity updates, disparities, densities and scales for each iteration
            with torch.no_grad():
                scales = visualization_dump['scales']  # Last dim has xyz components
                delta_stats = {
                    'iter': itr,
                    'delta_disparities': {
                        'mean': delta_disparities_full.mean().item(),
                        'std': delta_disparities_full.std().item(),
                        'min': delta_disparities_full.min().item(),
                        'max': delta_disparities_full.max().item()
                    },
                    'disparities': {
                        'mean': disparities_full.mean().item(),
                        'std': disparities_full.std().item(),
                        'min': disparities_full.min().item(),
                        'max': disparities_full.max().item()
                    },
                    'densities': {
                        'mean': densities_full.mean().item(),
                        'std': densities_full.std().item(),
                        'min': densities_full.min().item(),
                        'max': densities_full.max().item()
                    },
                    'scales': {
                        'x': {
                            'mean': scales[..., 0].mean().item(),
                            'std': scales[..., 0].std().item(),
                            'min': scales[..., 0].min().item(),
                            'max': scales[..., 0].max().item()
                        },
                        'y': {
                            'mean': scales[..., 1].mean().item(),
                            'std': scales[..., 1].std().item(),
                            'min': scales[..., 1].min().item(),
                            'max': scales[..., 1].max().item()
                        },
                        'z': {
                            'mean': scales[..., 2].mean().item(),
                            'std': scales[..., 2].std().item(),
                            'min': scales[..., 2].min().item(),
                            'max': scales[..., 2].max().item()
                        }
                    }
                }
                stats_list.append(delta_stats)

        return gaussian_predictions, depth_predictions, density_predictions, raw_gaussian_predictions, visualization_dump_list, stats_list


    def multiview_forward(self, images, intrinsics, extrinsics, near, far, iters=12, upsample=True, test_mode=False):
        # image size
        b, v, c, h_img, w_img = images.shape

        h = h_img // self.corr_resolution
        w = w_img // self.corr_resolution

        h_g = h_img // self.cfg.gaussian_sampling_resolution
        w_g = w_img // self.cfg.gaussian_sampling_resolution

        hdim = self.hidden_dim
        cdim = self.context_dim

        near = torch.tensor([self.cfg.near_depth_plane], dtype=near.dtype, device=near.device).view(1, 1).repeat(b, v)
        far = torch.tensor([self.cfg.far_depth_plane], dtype=far.dtype, device=near.device).view(1, 1).repeat(b, v)

        # only works with 2 context views
        if self.cfg.unet_backbone:
            trans_features, cnn_features = self.backbone(
                images,
                attn_splits=self.cfg.multiview_trans_attn_split,
                return_cnn_features=True,
                epipolar_kwargs=None,
            )

            trans_cnn_features_cat = torch.cat([trans_features, cnn_features], dim=2)
            trans_cnn_features_cat = rearrange(trans_cnn_features_cat, "b v c h w -> (v b) c h w")

            fnet_in = rearrange(trans_features, "b v c h w -> (v b) c h w")
            features = self.fnet(fnet_in)
            features = rearrange(features, "(v b) c h w -> b v c h w", b=b)

            cfeatures = self.cnet(trans_cnn_features_cat)
            net, inp = torch.split(cfeatures, [hdim, cdim], dim=1)
        else:
            image1, image2 = torch.split(images, [1, 1], dim=1)
            image1 = image1.squeeze(dim=1)
            image2 = image2.squeeze(dim=1)

            image1 = 2 * (image1) - 1.0
            image2 = 2 * (image2) - 1.0

            image1 = image1.contiguous()
            image2 = image2.contiguous()

            # run the feature network
            fmap1, fmap2 = self.fnet([image1, image2])
            features = torch.cat([fmap1.unsqueeze(dim=1), fmap2.unsqueeze(dim=1)], dim=1)

            # run the context network
            cnet1, cnet2 = self.cnet([image1, image2])
            net1, inp1 = torch.split(cnet1, [hdim, cdim], dim=1)
            net2, inp2 = torch.split(cnet2, [hdim, cdim], dim=1)

            net = torch.cat([net1.unsqueeze(dim=1), net2.unsqueeze(dim=1)], dim=1)
            inp = torch.cat([inp1.unsqueeze(dim=1), inp2.unsqueeze(dim=1)], dim=1)

            net = rearrange(net, "b v c h w -> (v b) c h w")
            inp = rearrange(inp, "b v c h w -> (v b) c h w")

        context_encoder_up_in = rearrange(images, "b v c h w -> (v b) c h w")
        context_encoder_up_in = 2. * context_encoder_up_in - 1.
        list_context_features_up = self.context_encoder_up(context_encoder_up_in)
        # print("Context feature levels: ", len(list_context_features_up))
        # print(f"Context features up: {[t.shape for t in list_context_features_up]}")

        corr_fn = MultiViewCostVolumeBlock(features, intrinsics, extrinsics, near, far,
                                           num_levels=self.cfg.corr_levels, radius=self.cfg.corr_radius,
                                           num_depth_candidates=self.cfg.num_depth_candidates)

        net = torch.tanh(net)
        inp = torch.relu(inp)

        disparities_full = self.initialize_disaprity(1 / near, 1 / far, h_g, w_g)
        densities_full = torch.zeros_like(disparities_full)

        disp_near = 1. / self.cfg.far_depth_plane
        disp_far = 1. / self.cfg.near_depth_plane

        # disparities = self.initialize_disaprity(1 / near, 1 / far, h, w)
        # densities = torch.zeros_like(disparities)

        depth_predictions = []
        density_predictions = []
        raw_gaussian_predictions = []
        gaussian_predictions = []
        visualization_dump_list = []
        stats_list = []

        for itr in range(iters):
            disparities_full = disparities_full.detach()
            densities_full = densities_full.detach()
            #
            # disparities = disparities.detach()
            # densities = densities.detach()

            disparities = F.interpolate(disparities_full, size=(h, w), align_corners=True, mode="bilinear")
            densities = F.interpolate(densities_full, size=(h, w), align_corners=True, mode="bilinear")

            # index correlation volume
            if self.search_space_dim == 1:
                corr = corr_fn(disparities)
            else:
                coords = self.get_indexing_coords(disparities, intrinsics, extrinsics)
                corr = corr_fn(coords)

            convex_upsample = (itr < (iters - 1)) and self.convex_upsampling
            with ((autocast(enabled=self.cfg.mixed_precision))):
                if convex_upsample:
                    net, out, up_mask = self.update_block_no_mask(net, inp, corr, disparities,
                                                                  densities, scale=self.cfg.scale_multiplier_up_mask)
                    gres_h, gres_w = h_g, w_g
                else:
                    net, out, _ = self.update_block_no_mask(net, inp, corr, disparities, densities)
                    gres_h, gres_w = h_img, w_img
                # net, up_mask_disp, up_mask_density, up_mask_gaussian, delta_disparities, densities, raw_gaussians = self.update_block(net, inp, corr, disparities, densities)

            # upsample predictions
            # F(t+1) = F(t) + \Delta(t)
            if convex_upsample and (self.corr_resolution != self.cfg.gaussian_sampling_resolution):
                out_up = self.upsample_tensor(out, up_mask)
            else:
                out_up = out

            if convex_upsample:
                delta_disparities_full = self.disparity_head_iter(out_up)
                densities_full = (self.scale_density_update * self.density_head_iter(out_up)).clamp(min=0.0, max=1.0)
                raw_gaussians_full = self.scale_gaussian_update * self.gaussian_head_iter(out_up)
            else:
                out_up = self.gaussian_upsampler(out, list_context_features_up)
                disparities_full = F.interpolate(disparities_full, size=(h_img, w_img), align_corners=True,
                                                 mode="bilinear")
                densities_full = F.interpolate(densities_full, size=(h_img, w_img), align_corners=True, mode="bilinear")
                raw_gaussians_full = F.interpolate(raw_gaussians_full, size=(h_img, w_img), align_corners=True,
                                                   mode="bilinear")

                delta_disparities_full = self.disparity_head(torch.cat([disparities_full, out_up], dim=1))
                densities_full = (self.scale_density_update * self.density_head(
                    torch.cat([densities_full, out_up], dim=1))).clamp(min=0.0, max=1.0)
                raw_gaussians_full = self.scale_gaussian_update * self.gaussian_head(
                    torch.cat([raw_gaussians_full, out_up], dim=1))

            disparities_full = (disparities_full + delta_disparities_full * self.scale_disp_update).clamp(min=disp_near,
                                                                                                          max=disp_far)
            depths_full = 1. / disparities_full

            depth_predictions.append(depths_full)
            density_predictions.append(densities_full)
            raw_gaussian_predictions.append(raw_gaussians_full)

            # rearrange the upsampled predictions
            densities_out = repeat(densities_full, "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1)
            depths_out = repeat(depths_full, "(v b) dpt h w -> b v (h w) srf dpt", b=b, v=v, srf=1)
            raw_gaussians_out = rearrange(raw_gaussians_full, "(v b) c h w -> b v (h w) c", v=v, b=b)

            # print(f"itr: {itr}, depths; {depths_out.shape}, densities: {densities_out.shape}, raw_gaussians: {raw_gaussians_out.shape}")
            gaussians, visualization_dump = self.convert_to_gaussians(depths_out, densities_out, raw_gaussians_out,
                                                                      extrinsics, intrinsics, gres_h, gres_w)
            gaussian_predictions.append(gaussians)
            visualization_dump_list.append(visualization_dump)

        return gaussian_predictions, depth_predictions, density_predictions, raw_gaussian_predictions, visualization_dump_list, stats_list
