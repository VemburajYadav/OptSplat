import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class DisparityHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(DisparityHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class GaussianHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=82):
        super(GaussianHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class BasicOutputHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=82):
        super(BasicOutputHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))


    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, cfg):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = cfg.corr_levels * (2*cfg.corr_radius + 1)
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicDisparityEncoder(nn.Module):
    def __init__(self, cfg):
        super(BasicDisparityEncoder, self).__init__()
        cor_planes = cfg.corr_levels * (2 * cfg.corr_radius + 1)
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, cfg, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(cfg)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class BasicUpdateBlock(nn.Module):
    def __init__(self, cfg, hidden_dim=128, input_dim=128, gru_res=8):
        super(BasicUpdateBlock, self).__init__()

        self.gru_res = gru_res
        self.encoder = BasicDisparityEncoder(cfg)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.disparity_head = DisparityHead(hidden_dim, hidden_dim=256)
        self.gaussian_head = GaussianHead(hidden_dim, hidden_dim=256, output_dim=82)

        self.mask_disparity = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, ((gru_res**2) // (cfg.gaussian_sampling_resolution**2))*9, 1, padding=0))

        self.mask_density = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, ((gru_res**2) // (cfg.gaussian_sampling_resolution**2))*9, 1, padding=0))

        self.mask_gaussian = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, ((gru_res**2) // (cfg.gaussian_sampling_resolution**2))*9, 1, padding=0))


    def forward(self, net, inp, corr, disparity, density):
        disp_inp = torch.cat([disparity, density], dim=1)
        depth_features = self.encoder(disp_inp, corr)
        inp = torch.cat([inp, depth_features], dim=1)

        # gru update
        net = self.gru(net, inp)

        # disparity head
        disp_out = self.disparity_head(net)

        delta_disp, opacity = torch.split(disp_out, [1, 1], dim=1)
        opacity = F.sigmoid(opacity)

        # gaussian head
        raw_gausians = self.gaussian_head(net)

        # scale mask to balence gradients
        mask_disparity = .25 * self.mask_disparity(net)
        mask_density = .25 * self.mask_density(net)
        mask_gaussians = .25 * self.mask_gaussian(net)

        return net, mask_disparity, mask_density, mask_gaussians, delta_disp, opacity, raw_gausians


class BasicUpdateBlockWoMask(nn.Module):
    def __init__(self, cfg, hidden_dim=128, input_dim=128, gru_res=8, output_dim=82):
        super(BasicUpdateBlockWoMask, self).__init__()

        self.gru_res = gru_res
        self.encoder = BasicDisparityEncoder(cfg)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.disparity_head = DisparityHead(hidden_dim, hidden_dim=256)
        self.gaussian_head = GaussianHead(hidden_dim, hidden_dim=256, output_dim=output_dim)

    def forward(self, net, inp, corr, disparity, density):
        disp_inp = torch.cat([disparity, density], dim=1)
        depth_features = self.encoder(disp_inp, corr)
        inp = torch.cat([inp, depth_features], dim=1)

        # gru update
        net = self.gru(net, inp)

        # disparity head
        disp_out = self.disparity_head(net)

        delta_disp, opacity = torch.split(disp_out, [1, 1], dim=1)
        opacity = F.sigmoid(opacity)

        # gaussian head
        raw_gausians = self.gaussian_head(net)

        return net, delta_disp, opacity, raw_gausians


class BasicUpdateBlockSingleHead(nn.Module):
    def __init__(self, cfg, hidden_dim=128, input_dim=128, gru_res=8, output_dim=82, upsample_res=1):
        super(BasicUpdateBlockSingleHead, self).__init__()

        self.gru_res = gru_res
        self.encoder = BasicDisparityEncoder(cfg)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.output_head = BasicOutputHead(hidden_dim, hidden_dim=256, output_dim=output_dim)

        self.return_up_mask = True if ((cfg.context_based_upsampling_iters == "last") and (gru_res != upsample_res)) else False

        up_scale = gru_res // upsample_res
        if self.return_up_mask:
            self.mask = nn.Sequential(
                nn.Conv2d(hidden_dim, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, (up_scale ** 2) * 9, 1, padding=0))

    def forward(self, net, inp, corr, disparity, density, scale=1.0):
        disp_inp = torch.cat([disparity, density], dim=1)
        depth_features = self.encoder(disp_inp, corr)
        inp = torch.cat([inp, depth_features], dim=1)

        # gru update
        net = self.gru(net, inp)

        # output head
        out = self.output_head(net)

        if self.return_up_mask:
            up_mask = scale * self.mask(net)
            return net, out, up_mask

        return net, out, None
