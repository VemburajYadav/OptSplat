import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
from einops import rearrange
from .encoder.backbone.unimatch.geometry import coords_grid

# Get the path to the other project relative to current project
mast3r_path = Path(__file__).parent.parent.parent.parent / "../mast3r"
print("mast3r path: ", mast3r_path)
sys.path.append(str(mast3r_path))

from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs

import mast3r.utils.path_to_dust3r
from dust3r.inference import inference
from dust3r.utils.image import load_images


class PointMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
        # you can put the path to a local checkpoint in model_name if needed
        self.mast3r_model = AsymmetricMASt3R.from_pretrained(model_name)


    def process_inputs(self, img1, img2):
        b, _, h, w = img1.shape

        img_scaled = F.interpolate(torch.cat([img1, img2], dim=0), size=(384, 512), mode="bilinear", align_corners=True)
        img_scaled = 2 * img_scaled - 1

        img1_scaled, img2_scaled = torch.split(img_scaled, [b, b], dim=0)

        img1_dict = {"img": img1_scaled, "true_shape": np.array([[384, 512]], dtype=np.int32), "idx": 0, "instance": 0}
        img2_dict = {"img": img2_scaled, "true_shape": np.array([[384, 512]], dtype=np.int32), "idx": 1, "instance": 1}

        return [img1_dict, img2_dict]


    def run_inference(self, img1, img2):
        b,  _, h, w = img1.shape
        images = self.process_inputs(img1, img2)
        output = inference([tuple(images)], self.mast3r_model, img1.device, batch_size=b, verbose=False)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']

        desc1 = pred1["desc"]
        desc2 = pred2["desc"]

        return desc1, desc2

    def inference_and_matching(self, img1, img2):
        b, n_v, _, h, w = img1.shape

        img1 = rearrange(img1, "b v c h w -> (v b) c h w")
        img2 = rearrange(img2, "b v c h w -> (v b) c h w")

        matches_list = []

        with torch.no_grad():
            desc1, desc2 = self.run_inference(img1, img2)

            desc1 = desc1.detach()
            desc2 = desc2.detach()

            for i in range(b * n_v):
                with torch.no_grad():
                    # find 2D-2D matches between the two images
                    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1[i], desc2[i], subsample_or_initxy1=8,
                                                                   device=desc1.device, dist='dot', block_size=2 ** 13,
                                                                   )

                    x1_s, y1_s, x2_s, y2_s = matches_im0[:, 0], matches_im0[:, 1], matches_im1[:, 0], matches_im1[:, 1]
                    x1_s, y1_s, x2_s, y2_s = [torch.tensor(x, dtype=torch.float32, device=desc1.device) for x in
                                              (x1_s, y1_s, x2_s, y2_s)]

                    x1 = x1_s * w / 512
                    x2 = x2_s * w / 512
                    y1 = y1_s * h / 384
                    y2 = y2_s * h / 384

                    matches_list.append([x1, y1, x2, y2])

        return matches_list

    def compute_loss_from_depths_and_matches(self, depths, matches_list, k1, k2, poses1, poses2):
        b, n_v, _, h, w = depths.shape

        depths = rearrange(depths, "b v c h w -> (v b) c h w")
        k1 = rearrange(k1, "b v h w -> (v b) h w")
        k2 = rearrange(k2, "b v h w -> (v b) h w")
        poses1 = rearrange(poses1, "b v h w -> (v b) h w")
        poses2 = rearrange(poses2, "b v h w -> (v b) h w")

        with torch.no_grad():
            k1[:, 0, :] = k1[:, 0, :] * float(w)
            k1[:, 1, :] = k1[:, 1, :] * float(h)

            k2[:, 0, :] = k2[:, 0, :] * float(w)
            k2[:, 1, :] = k2[:, 1, :] * float(h)

            grid = coords_grid(
                1, h, w, homogeneous=True, device=depths.device).squeeze(0)  # [B, 3, H, W]

        total_loss = 0.0

        for i in range(b * n_v):
            x1, y1, x2, y2 = matches_list[i]

            if (x1.shape[0] > 0):
                with torch.no_grad():
                    x1 = x1.to(torch.int32)
                    y1 = y1.to(torch.int32)

                    k1i, k2i = k1[i], k2[i]
                    poses1i, poses2i = poses1[i], poses2[i]
                    rel_pose_i = poses2i.inverse() @ poses1i
                    points_i = grid[:, y1, x1]

                depths_i = depths[i][:, y1, x1]

                points_proj =  k2i @ (rel_pose_i[:3, :3] @ ((torch.inverse(k1i) @ points_i) * depths_i) + rel_pose_i[:3, -1:])
                x2_proj = points_proj[0] / points_proj[2]
                y2_proj = points_proj[1] / points_proj[2]

                l1_loss = (x2 - x2_proj).abs() + (y2 - y2_proj).abs()
                # print("element loss:", l1_loss.min(), l1_loss.max(), l1_loss.mean())
                total_loss = total_loss + l1_loss.mean()

        total_loss = total_loss / (b * n_v)
        # print("total loss: ", total_loss)
        return total_loss


    def compute_matching_loss(self, imgs_c, imgs_t, k_c, k_t, poses_c, poses_t, depths_c):
        b, n_t, _, h, w = imgs_t.shape

        imgs_c = rearrange(imgs_c, "b v c h w -> (v b) c h w")
        imgs_t = rearrange(imgs_t, "b v c h w -> (v b) c h w")
        depths_c = rearrange(depths_c, "b v c h w -> (v b) c h w")
        k_c = rearrange(k_c, "b v h w -> (v b) h w")
        k_t = rearrange(k_t, "b v h w -> (v b) h w")
        poses_c = rearrange(poses_c, "b v h w -> (v b) h w")
        poses_t = rearrange(poses_t, "b v h w -> (v b) h w")

        k_c[:, 0, :] = k_c[:, 0, :] * float(w)
        k_c[:, 1, :] = k_c[:, 1, :] * float(h)

        k_t[:, 0, :] = k_t[:, 0, :] * float(w)
        k_t[:, 1, :] = k_t[:, 1, :] * float(h)

        with torch.no_grad():
            desc1, desc2 = self.run_inference(imgs_c, imgs_t)

            desc1 = desc1.detach()
            desc2 = desc2.detach()

            grid = coords_grid(
                1, h, w, homogeneous=True, device=desc1.device
            ).squeeze(0)  # [B, 3, H, W]

        total_loss = 0.0

        for i in range(b * n_t):
            with torch.no_grad():
                # find 2D-2D matches between the two images
                matches_im0, matches_im1 = fast_reciprocal_NNs(desc1[i], desc2[i], subsample_or_initxy1=8,
                                                               device=desc1.device, dist='dot', block_size=2 ** 13,
                                                               )

                x1_s, y1_s, x2_s, y2_s = matches_im0[:, 0], matches_im0[:, 1], matches_im1[:, 0], matches_im1[:, 1]
                x1_s, y1_s, x2_s, y2_s = [torch.tensor(x, dtype=torch.float32, device=desc1.device) for x in (x1_s, y1_s, x2_s, y2_s)]

                x1 = x1_s * w / 512
                x2 = x2_s * w / 512
                y1 = y1_s * h / 384
                y2 = y2_s * h / 384

                # print(x1.shape, x1.max(), x2.max(), y1.max(), y2.max())

            if (x1.shape[0] > 0):
                with torch.no_grad():
                    x1 = x1.to(torch.int32)
                    y1 = y1.to(torch.int32)

                    k_ci, k_ti = k_c[i], k_t[i]
                    poses_ci, poses_ti = poses_c[i], poses_t[i]
                    rel_pose_i = poses_ti.inverse() @ poses_ci
                    points_i = grid[:, y1, x1]

                depths_i = depths_c[i][:, y1, x1]
                # print(points_i.shape, depths_i.shape)

                points_proj =  k_ti @ (rel_pose_i[:3, :3] @ ((torch.inverse(k_ci) @ points_i) * depths_i) + rel_pose_i[:3, -1:])
                x2_proj = points_proj[0] / points_proj[2]
                y2_proj = points_proj[1] / points_proj[2]
                # print(points_proj.shape, x2_proj.shape, x2_proj.min(), x2_proj.max(), y2_proj.min(), y2_proj.max())

                l1_loss = (x2 - x2_proj).abs() + (y2 - y2_proj).abs()
                # print("element loss:", l1_loss.min(), l1_loss.max(), l1_loss.mean())

                total_loss = total_loss + l1_loss.mean()

        total_loss = total_loss / (b * n_t)
        # print("total_loss: ", total_loss)

        return total_loss






