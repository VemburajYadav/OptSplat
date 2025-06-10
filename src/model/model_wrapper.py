from dataclasses import dataclass
from pathlib import Path
from random import gauss
from typing import Optional, Protocol, runtime_checkable

from PIL import Image
import moviepy.editor as mpy
import torch
import torch.nn.functional as F
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from markdown_it.rules_inline import image
from pathspec import iter_tree
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json

from torch.ao.quantization import default_quint8_weight_qconfig

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from .ply_export import export_ply
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization import layout
from ..visualization.validation_in_3d import render_cameras, render_projections
from ..visualization.vis_gaussians import visualize_gaussians
from ..visualization.vis_depth import viz_depth_tensor, viz_error_tensor
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .raft_gaussian_splat.core.raft import RAFT
from .point_matching_loss import PointMatchingLoss
from .types import Gaussians

@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    test_iters: int
    test_wo_gaussian_upsampler: bool


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    wo_gs_rendering_loss_context: bool
    loss_reduction: str
    seq_loss_gamma: float
    point_matching_loss: bool
    point_matching_loss_weight: float


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: RAFT,
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_visualizer = None
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)
        self.iters = self.encoder.cfg.iters
        print("iters: ", self.iters)
        print(self.encoder.cfg)

        self.w_point_matching_loss = self.train_cfg.point_matching_loss

        if self.w_point_matching_loss:
            self.point_matcher = PointMatchingLoss()
            self.point_matcher.requires_grad_(requires_grad=False)
            self.point_matcher.eval()

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.psnr_outputs = {}
            self.ssim_outputs = {}
            self.lpips_outputs = {}
            self.mse_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, n_tv, _, h, w = batch["target"]["image"].shape
        _, n_cv, _, _, _ = batch["context"]["image"].shape

        # print("Image Input: ", batch["context"]["image"].shape)
        # print("Image Intrinsics: ", batch["context"]["intrinsics"].shape, batch["context"]["intrinsics"])
        # print("Image Extrinsics: ", batch["context"]["extrinsics"].shape, batch["context"]["extrinsics"])
        # exit()

        # only works with 2 context views
        image1, image2 = torch.split(batch["context"]["image"], [1, 1], dim=1)

        # Run the model.
        gaussians_list, depths_list, _, _, _, stats_list = self.encoder(batch["context"]["image"],
                                         batch["context"]["intrinsics"],
                                         batch["context"]["extrinsics"],
                                         batch["context"]["near"], batch["context"]["far"], iters=self.iters)

        if not self.train_cfg.wo_gs_rendering_loss_context:
            render_viewpoint_extrinsics = torch.cat([batch["target"]["extrinsics"], batch["context"]["extrinsics"]], dim=1)
            render_viewpoint_intrinsics = torch.cat([batch["target"]["intrinsics"], batch["context"]["intrinsics"]], dim=1)
            render_viewpoint_near = torch.cat([batch["target"]["near"], batch["context"]["near"]], dim=1)
            render_viewpoint_far = torch.cat([batch["target"]["far"], batch["context"]["far"]], dim=1)
            target_gt = torch.cat([batch["target"]["image"], batch["context"]["image"]], dim=1)
        else:
            render_viewpoint_extrinsics = batch["target"]["extrinsics"]
            render_viewpoint_intrinsics = batch["target"]["intrinsics"]
            render_viewpoint_near = batch["target"]["near"]
            render_viewpoint_far = batch["target"]["far"]
            target_gt = batch["target"]["image"]

        # render the context and target frames from the gaussians predicted over the iterations of optimization
        n_predictions = len(gaussians_list)

        output_list = []
        psnr_list = []
        loss_list = []

        if self.w_point_matching_loss:
            # p_context_ids = torch.randint(0, n_cv, [n_tv], dtype=torch.int32, device=image1.device)
            mast3r_matches = self.point_matcher.inference_and_matching(batch["context"]["image"].detach()[:, [0, 1], ...],
                                                                       batch["context"]["image"].detach()[:, [1, 0], ...])


        for i in range(self.iters):
            gaussians = gaussians_list[i]
            output = self.decoder.forward(
                gaussians,
                render_viewpoint_extrinsics,
                render_viewpoint_intrinsics,
                render_viewpoint_near,
                render_viewpoint_far,
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
            output_list.append(output)

            # Compute metrics.
            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(output.color, "b v c h w -> (b v) c h w"),
            )
            psnr_list.append(psnr_probabilistic)

            # Compute and log loss.
            total_loss = 0
            for loss_fn in self.losses:
                loss = loss_fn.forward(output, target_gt, batch, gaussians, self.global_step)
                self.log(f"loss/{loss_fn.name}", loss)
                total_loss = total_loss + loss

            if self.w_point_matching_loss:
                depth_mvs = depths_list[i]
                depth_mvs = rearrange(depth_mvs, "(v b) c h w -> b v c h w", b=b)
                # p_match_loss = self.point_matcher.compute_matching_loss(batch["context"]["image"].detach()[:, p_context_ids, ...],
                #                                          output.color,
                #                                          batch["context"]["intrinsics"].clone().detach()[:, p_context_ids, ...],
                #                                          batch["target"]["intrinsics"].clone().detach(),
                #                                          batch["context"]["extrinsics"].clone().detach()[:, p_context_ids, ...],
                #                                          batch["target"]["extrinsics"].clone().detach(),
                #                                          depth_mvs[:, p_context_ids, ...])

                p_match_loss = self.point_matcher.compute_loss_from_depths_and_matches(depth_mvs,
                                                                                       mast3r_matches,
                                                                                       batch["context"]["intrinsics"].clone().detach()[:, [0, 1], ...],
                                                                                       batch["context"]["intrinsics"].clone().detach()[:, [1, 0], ...],
                                                                                       batch["context"]["extrinsics"].clone().detach()[:, [0, 1], ...],
                                                                                       batch["context"]["extrinsics"].clone().detach()[:, [1, 0], ...]
                                                                                       )
                total_loss = total_loss + self.train_cfg.point_matching_loss_weight * p_match_loss


            loss_list.append(total_loss)

        # compute the sequence loss
        seq_loss = 0.0
        seq_loss_gamma = self.train_cfg.seq_loss_gamma
        for i in range(n_predictions):
            i_weight = seq_loss_gamma ** (n_predictions - i - 1)
            seq_loss += i_weight * loss_list[i]

        if self.train_cfg.loss_reduction == "avg":
            seq_loss /= n_predictions

        self.log("loss/total", seq_loss)
        self.log("train/psnr_probabilistic", psnr_list[-1].mean())

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"iter loss = {[x.item() for x in loss_list]}; "
                f"loss = {seq_loss:.6f}"
            )
            print("stats: ", stats_list)
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return seq_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        _, n_cv, _, _, _ = batch["context"]["image"].shape

        assert b == 1

        # if batch_idx < 6000:
        #     return
        # context_images = batch["context"]["image"]
        # context_images = rearrange(context_images, "b v c h w -> (v b) c h w")
        # context_images = F.interpolate(context_images, size=(256, 256), mode="bilinear", align_corners=True).clamp(min=0.0, max=1.0)
        # context_images = rearrange(context_images, "(v b) c h w -> b v c h w", b=b)
        # batch["context"]["image"] = context_images
        # batch["context"]["intrinsics"][:, :, 0, :] = batch["context"]["intrinsics"][:, :, 0, :] * 448 / 256
        # batch["target"]["intrinsics"][:, :, 0, :] = batch["target"]["intrinsics"][:, :, 0, :] * 448 / 256

        # Tiled execution of encoder
        # if h == 256 and w == 448:
        #     context_images = batch["context"]["image"]
        #     context_images_left = context_images[:, :, :, :, :256]
        #     context_images_right = context_images[:, :, :, :, 192:]
        #
        #     context_intrinsics = batch["context"]["intrinsics"]
        #     context_intrinsics_left = context_intrinsics.clone()
        #     context_intrinsics_right = context_intrinsics.clone()
        #
        #     context_intrinsics_left[:, :, 0, 0] = context_intrinsics[:, :, 0, 0] * 448 / 256
        #     context_intrinsics_left[:, :, 0, 2] = context_intrinsics[:, :, 0, 2] * 448 / 256
        #
        #     context_intrinsics_right[:, :, 0, 0] = context_intrinsics[:, :, 0, 0] * 448 / 256
        #     context_intrinsics_right[:, :, 0, 2] = (context_intrinsics[:, :, 0, 2] * 448 - 192) / 256
        #
        #     context_extrinsics = batch["context"]["extrinsics"]
        #     context_extrinsics_left = context_extrinsics.clone()
        #     context_extrinsics_right = context_extrinsics.clone()
        #
        #     batch["context"]["image"] = torch.cat([context_images_left, context_images_right], dim=0)
        #     batch["context"]["intrinsics"] = torch.cat([context_intrinsics_left, context_intrinsics_right], dim=0)
        #     batch["context"]["extrinsics"] = torch.cat([context_extrinsics_left, context_extrinsics_right], dim=0)


        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        iters = self.test_cfg.test_iters
        # Render Gaussians.
        # with self.benchmarker.time("encoder"):
        # Run the model.

        if n_cv > 2:
            gaussians_list, depths_list, _, _, visualization_dump_list, _ = self.encoder.multiview_forward(batch["context"]["image"],
                                                                                                           batch["context"]["intrinsics"],
                                                                                                           batch["context"]["extrinsics"],
                                                                                                           batch["context"]["near"],
                                                                                                           batch["context"]["far"],
                                                                                                           iters=iters)
        else:
            gaussians_list, depths_list, _, _, visualization_dump_list, _ = self.encoder(batch["context"]["image"],
                                                              batch["context"]["intrinsics"],
                                                              batch["context"]["extrinsics"],
                                                              batch["context"]["near"], batch["context"]["far"], iters=iters)

        # if self.test_cfg.save_image:
        #     intrinsics_save_path = Path(path / scene / "intrinsics.npz")
        #     extrinsics_save_path = Path(path / scene / "extrinsics.npz")
        #
        #     intrinsics_dict = {}
        #     extrinsics_dict = {}
        #
        #     for index, param in zip(batch["target"]["index"][0], batch["target"]["intrinsics"][0]):
        #         intrinsics_dict[f"{index:0>6}"] = param.cpu().numpy()
        #     for index, param in zip(batch["context"]["index"][0], batch["context"]["intrinsics"][0]):
        #         intrinsics_dict[f"{index:0>6}"] = param.cpu().numpy()
        #     for index, param in zip(batch["target"]["index"][0], batch["target"]["extrinsics"][0]):
        #         extrinsics_dict[f"{index:0>6}"] = param.cpu().numpy()
        #     for index, param in zip(batch["context"]["index"][0], batch["context"]["extrinsics"][0]):
        #         extrinsics_dict[f"{index:0>6}"] = param.cpu().numpy()
        #     np.savez(intrinsics_save_path, **intrinsics_dict)
        #     np.savez(extrinsics_save_path, **extrinsics_dict)

        psnr_list = []
        ssim_list = []
        lpips_list = []
        mse_list = []

        self.psnr_outputs[scene] = {}
        self.ssim_outputs[scene] = {}
        self.lpips_outputs[scene] = {}
        self.mse_outputs[scene] = {}

        for index in batch["target"]["index"][0]:
            self.psnr_outputs[scene][f"{index:0>6}"] = []
            self.ssim_outputs[scene][f"{index:0>6}"] = []
            self.lpips_outputs[scene][f"{index:0>6}"] = []
            self.mse_outputs[scene][f"{index:0>6}"] = []

        if self.test_cfg.save_image:
            for itr in range(iters):
                gaussians = gaussians_list[itr]
                visualization_dump = visualization_dump_list[itr]
                depth_mvs = depths_list[itr]

                # Upscale the depth maps to full resolution
                depth_mvs = F.interpolate(depth_mvs, size=(h, w), align_corners=True, mode="bilinear")
                depth_mvs = rearrange(depth_mvs, "(v b) c h w -> b v c h w", b=b)

                # with self.benchmarker.time("decoder", num_calls=v):
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode="depth",
                )

                output_context = self.decoder.forward(
                    gaussians,
                    batch["context"]["extrinsics"],
                    batch["context"]["intrinsics"],
                    batch["context"]["near"],
                    batch["context"]["far"],
                    (h, w),
                    depth_mode="depth",
                )


                images_prob = output.color[0]
                target_depth_rendered = output.depth[0]
                rgb_gt = batch["target"]["image"][0]

                for index, color, color_gt in zip(batch["target"]["index"][0], images_prob, rgb_gt):
                    psnr_itr = compute_psnr(color_gt.unsqueeze(dim=0), color.unsqueeze(dim=0)).mean().item()
                    ssim_itr = compute_ssim(color_gt.unsqueeze(dim=0), color.unsqueeze(dim=0)).mean().item()
                    lpips_itr = compute_lpips(color_gt.unsqueeze(dim=0),color.unsqueeze(dim=0)).mean().item()
                    mse_itr = torch.sum((color - color_gt)**2, dim=0, keepdim=False).mean().item()

                    self.psnr_outputs[scene][f"{index:0>6}"].append(psnr_itr)
                    self.ssim_outputs[scene][f"{index:0>6}"].append(ssim_itr)
                    self.lpips_outputs[scene][f"{index:0>6}"].append(lpips_itr)
                    self.mse_outputs[scene][f"{index:0>6}"].append(mse_itr)

                # Save images.
                if self.test_cfg.save_image:
                    for index, color in zip(batch["target"]["index"][0], images_prob):
                        save_image(color, path / scene / "color" / "target_rendered" / f"{index:0>6}_{itr:0>2}.png")

                    for index, color in zip(batch["target"]["index"][0], batch["target"]["image"][0]):
                        save_image(color, path / scene / "color" / "target" / f"{index:0>6}.png")

                    for index, color, color_gt in zip(batch["target"]["index"][0], images_prob, rgb_gt):
                        error_img = torch.sum((color - color_gt)**2, dim=0)
                        vis_error = viz_error_tensor(error_img.cpu(), return_numpy=False).to(dtype=error_img.dtype)
                        save_image(vis_error / 255, path / scene / "color" / "target_error" / f"{index:0>6}_{itr:0>2}.png")

                    for index, color in zip(batch["context"]["index"][0], output_context.color[0]):
                        save_image(color, path / scene / "color" / "context_rendered" / f"{index:0>6}_{itr:0>2}.png")

                    for index, color in zip(batch["context"]["index"][0], batch["context"]["image"][0]):
                        save_image(color, path / scene / "color" / "context" / f"{index:0>6}.png")

                    for index, color, color_gt in zip(batch["context"]["index"][0], output_context.color[0], batch["context"]["image"][0]):
                        error_img = torch.sum((color - color_gt)**2, dim=0)
                        vis_error = viz_error_tensor(error_img.cpu(), return_numpy=False).to(dtype=error_img.dtype)
                        save_image(vis_error / 255, path / scene / "color" / "context_error" / f"{index:0>6}_{itr:0>2}.png")

                    for index, depth in zip(batch["context"]["index"][0], depth_mvs[0]):
                        vis_depth = viz_depth_tensor(
                            1.0 / depth[0].cpu(), return_numpy=False
                        ).to(dtype=depth.dtype)
                        save_image(vis_depth / 255,
                                   path / scene / "depth_encoder" / "context" / f"{index:0>6}_{itr:0>2}.png")

                    for index, depth in zip(batch["target"]["index"][0], target_depth_rendered):
                        vis_depth = viz_depth_tensor(
                            1.0 / depth.cpu(), return_numpy=False
                        ).to(dtype=depth.dtype)
                        save_image(vis_depth / 255,
                                   path / scene / "depth_rendered" / "target" / f"{index:0>6}_{itr:0>2}.png")

                    for index, depth in zip(batch["context"]["index"][0], output_context.depth[0]):
                        vis_depth = viz_depth_tensor(
                            1.0 / depth.cpu(), return_numpy=False
                        ).to(dtype=depth.dtype)
                        save_image(vis_depth / 255,
                                   path / scene / "depth_rendered" / "context" / f"{index:0>6}_{itr:0>2}.png")

                    export_ply(batch["context"]["extrinsics"][0, 0],
                            gaussians.means[0],
                            visualization_dump["scales"][0],
                            visualization_dump["rotations"][0],
                            gaussians.harmonics[0],
                            gaussians.opacities[0],
                            path / scene / "point_clouds" / f"{itr:0>2}.ply")
        else:
            itr = self.test_cfg.test_iters - 1

            if self.test_cfg.test_wo_gaussian_upsampler:
                gaussians = gaussians_list[itr - 1]
            else:
                gaussians = gaussians_list[itr]
            # gaussians = gaussians_list[itr - 1]

            # if n_cv == 4:
            #     # means_01, means_12, means_23 = torch.split(gaussians.means, [b,b,b], dim=0)
            #     # covariances_01, covariances_12, covariances_23 = torch.split(gaussians.covariances, [b,b,b], dim=0)
            #     # harmonics_01, harmonics_12, harmonics_23 = torch.split(gaussians.harmonics, [b,b,b], dim=0)
            #     # opacities_01, opacities_12, opacities_23 = torch.split(gaussians.opacities, [b,b,b], dim=0)
            #     #
            #     # means = torch.cat([means_01, means_12, means_23], dim=1)
            #     # covariances = torch.cat([covariances_01, covariances_12, covariances_23], dim=1)
            #     # harmonics = torch.cat([harmonics_01, harmonics_12, harmonics_23], dim=1)
            #     # opacities = torch.cat([opacities_01, opacities_12, opacities_23], dim=1)
            #
            #     means_01, means_23 = torch.split(gaussians.means, [b,b], dim=0)
            #     covariances_01, covariances_23 = torch.split(gaussians.covariances, [b,b], dim=0)
            #     harmonics_01, harmonics_23 = torch.split(gaussians.harmonics, [b,b], dim=0)
            #     opacities_01, opacities_23 = torch.split(gaussians.opacities, [b,b], dim=0)
            #
            #     means = torch.cat([means_01, means_23], dim=1)
            #     covariances = torch.cat([covariances_01, covariances_23], dim=1)
            #     harmonics = torch.cat([harmonics_01, harmonics_23], dim=1)
            #     opacities = torch.cat([opacities_01, opacities_23], dim=1)
            #
            #     gaussians = Gaussians(means, covariances, harmonics, opacities)


            # with self.benchmarker.time("decoder", num_calls=v):
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode="depth",
            )

            images_prob = output.color[0]
            # images_prob = F.interpolate(images_prob, size=(256, 448), mode="bilinear", align_corners=True)
            rgb_gt = batch["target"]["image"][0]

        # save video
        # if self.test_cfg.save_video:
        #     frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
        #     save_video(
        #         [a for a in images_prob],
        #         path / "video" / f"{scene}_frame_{frame_str}.mp4",
        #     )
        # vis_gaussian_output = visualize_gaussians(batch["context"]["image"],
        #                                           gaussians.opacities,
        #                                           gaussians.covariances,
        #                                           gaussians.harmonics[..., 0],
        #                                           self.encoder.cfg.gaussian_sampling_resolution)
        # print("Vis Gaussians: ", vis_gaussian_output.shape, vis_gaussian_output.min(), vis_gaussian_output.max())


        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v
            rgb = images_prob


            if f"psnr" not in self.test_step_outputs:
                self.test_step_outputs[f"psnr"] = []
            if f"ssim" not in self.test_step_outputs:
                self.test_step_outputs[f"ssim"] = []
            if f"lpips" not in self.test_step_outputs:
                self.test_step_outputs[f"lpips"] = []

            self.test_step_outputs[f"psnr"].append(
                compute_psnr(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"ssim"].append(
                compute_ssim(rgb_gt, rgb).mean().item()
            )
            self.test_step_outputs[f"lpips"].append(
                compute_lpips(rgb_gt, rgb).mean().item()
            )

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        out_dir = self.test_cfg.output_path / name
        saved_scores = {}

        if self.test_cfg.save_image and self.test_cfg.compute_scores:
            with (self.test_cfg.output_path / f"psnr_outputs.json").open("w") as f:
                json.dump(self.psnr_outputs, f)
            with (self.test_cfg.output_path / f"ssim_outputs.json").open("w") as f:
                json.dump(self.ssim_outputs, f)
            with (self.test_cfg.output_path / f"lpips_outputs.json").open("w") as f:
                json.dump(self.lpips_outputs, f)
            with (self.test_cfg.output_path / f"mse_outputs.json").open("w") as f:
                json.dump(self.mse_outputs, f)

        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)

            with (out_dir / f"psnr_outputs.json").open("w") as f:
                json.dump(self.psnr_outputs, f)

            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
            self.benchmarker.dump_memory(
                self.test_cfg.output_path / name / "peak_memory.json"
            )
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1

        # Run the model.
        gaussians_softmax_list, _, _, _, _, _ = self.encoder(batch["context"]["image"],
                                         batch["context"]["intrinsics"],
                                         batch["context"]["extrinsics"],
                                         batch["context"]["near"], batch["context"]["far"], iters=self.iters)
        gaussians_softmax = gaussians_softmax_list[-1]

        output_softmax = self.decoder.forward(
            gaussians_softmax,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )
        rgb_softmax = output_softmax.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        for tag, rgb in zip(
            ("val",), (rgb_softmax,)
        ):
            psnr = compute_psnr(rgb_gt, rgb).mean()
            self.log(f"val/psnr_{tag}", psnr)
            lpips = compute_lpips(rgb_gt, rgb).mean()
            self.log(f"val/lpips_{tag}", lpips)
            ssim = compute_ssim(rgb_gt, rgb).mean()
            self.log(f"val/ssim_{tag}", ssim)

        # Construct comparison image.
        comparison = hcat(
            add_label(vcat(*batch["context"]["image"][0]), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_softmax), "Target (Softmax)"),
        )
        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

        # Render projections and construct projection image.
        # projections = hcat(*render_projections(
        #                         gaussians_softmax,
        #                         256,
        #                         extra_label="(Softmax)",
        #                     )[0])
        # self.logger.log_image(
        #     "projection",
        #     [prep_image(add_border(projections))],
        #     step=self.global_step,
        # )
        #
        # # Draw cameras.
        # cameras = hcat(*render_cameras(batch, 256))
        # self.logger.log_image(
        #     "cameras", [prep_image(add_border(cameras))], step=self.global_step
        # )
        #
        # if self.encoder_visualizer is not None:
        #     for k, image in self.encoder_visualizer.visualize(
        #         batch["context"], self.global_step
        #     ).items():
        #         self.logger.log_image(k, [prep_image(image)], step=self.global_step)
        #
        # # Run video validation step.
        # self.render_video_interpolation(batch)
        # self.render_video_wobble(batch)
        # if self.train_cfg.extended_visualization:
        #     self.render_video_interpolation_exaggerated(batch)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob_list, _, _, _, _, _ = self.encoder(batch["context"]["image"],
                                         batch["context"]["intrinsics"],
                                         batch["context"]["extrinsics"],
                                         batch["context"]["near"], batch["context"]["far"], iters=self.iters)
        gaussians_prob = gaussians_prob_list[-1]
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        # output_det = self.decoder.forward(
        #     gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth"
        # )
        # images_det = [
        #     vcat(rgb, depth)
        #     for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        # ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Softmax"),
                    # add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr,
                            self.trainer.max_steps + 10,
                            pct_start=0.01,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }
