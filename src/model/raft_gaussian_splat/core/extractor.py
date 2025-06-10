import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, norm_channels=8, num_layers=1,
                 downsample=True, activation="silu", normalization=None):
        super(ResidualDownBlock, self).__init__()

        self.down_sample = downsample
        self.num_layers = num_layers
        self.activation_fn = activation
        if activation == "silu":
            self.activation = nn.SiLU
        elif activation == "gelu":
            self.activation = nn.GELU
        else:
            self.activation = nn.Identity

        self.resnet_conv_first = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(norm_channels, out_channels) if normalization is not None else nn.Sequential(),
                    self.activation(),
                ) for i in range(self.num_layers)])

        self.resnet_conv_second = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(norm_channels, out_channels) if normalization is not None else nn.Sequential(),
                    self.activation(),
                ) for _ in range(self.num_layers)])


        self.residual_input_conv = nn.ModuleList([nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1) for i in range(self.num_layers)])

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          3, 2, 1) if self.down_sample else nn.Identity()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

        # Downsample
        out = self.down_sample_conv(out)
        return out


class ResidualUpBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, norm_channels=8, num_layers=1,
                 upsample=True, activation="silu", normalization=None):
        super(ResidualUpBlock, self).__init__()

        self.up_sample = upsample
        self.num_layers = num_layers
        self.activation_fn = activation
        if activation == "silu":
            self.activation = nn.SiLU
        elif activation == "gelu":
            self.activation = nn.GELU
        else:
            self.activation = nn.Identity

        self.resnet_conv_first = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(norm_channels, out_channels) if normalization is not None else nn.Sequential(),
                    self.activation(),
                ) for i in range(self.num_layers)])

        self.resnet_conv_second = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1, padding=1),
                    nn.GroupNorm(norm_channels, out_channels) if normalization is not None else nn.Sequential(),
                    self.activation(),
                ) for _ in range(self.num_layers)])


        self.residual_input_conv = nn.ModuleList([nn.Conv2d(in_channels if i == 0 else out_channels,
                                                            out_channels, kernel_size=1) for i in range(self.num_layers)])
        self.up_sample_nn = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) if self.up_sample else nn.Identity()


    def forward(self, x, x_down):
        # Upsample
        x = self.up_sample_nn(x)
        x = torch.cat([x, x_down], dim=1)

        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv_first[i](out)
            out = self.resnet_conv_second[i](out)
            out = out + self.residual_input_conv[i](resnet_input)

        return out


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, channel_dims=(32, 64, 128), activation="silu", normalization=None):
        super(ResNetEncoder, self).__init__()

        self.in_channels = in_channels
        self.channel_dims = channel_dims
        self.activation_fn = activation
        if activation == "silu":
            self.activation = nn.SiLU
        elif activation == "gelu":
            self.activation = nn.GELU
        else:
            self.activation = nn.Identity


        self.conv_in = nn.Conv2d(3, self.channel_dims[0], kernel_size=7, stride=1, padding=3)
        self.norm_in = nn.GroupNorm(8, self.channel_dims[0]) if normalization is not None else nn.Sequential()
        self.activation_in = self.activation()

        self.encoder_blocks = nn.ModuleList([
            ResidualDownBlock(in_channels=self.channel_dims[i], out_channels=self.channel_dims[i+1],
                              num_layers=2, downsample=True, activation=activation, normalization=normalization)
            for i in range(len(self.channel_dims) - 1)])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.activation_in(self.norm_in(self.conv_in(x)))

        x_list = [x]
        for blk in self.encoder_blocks:
            x = blk(x)
            x_list.append(x)

        return x_list


class GaussianUpsampler(nn.Module):
    def __init__(self, input_dim=128, output_dim=84, context_dims_up=(32, 64, 128),
                 scale_factor=4, normalization=None):
        super(GaussianUpsampler, self).__init__()

        self.input_dim = input_dim
        self.context_dims_up = context_dims_up
        self.scale_factor = scale_factor
        self.num_levels = int(math.log2(scale_factor))

        self.input_block = ResidualDownBlock(in_channels=input_dim + self.context_dims_up[-1],
                                             out_channels=self.context_dims_up[-1], num_layers=2,
                                             downsample=False, activation="gelu", normalization=normalization)

        self.up_blocks = nn.ModuleList([
            ResidualUpBlock(in_channels=self.context_dims_up[-1 - i] + self.context_dims_up[-2 - i],
                            out_channels=self.context_dims_up[-2 - i] if i < (len(self.context_dims_up) - 2) else output_dim, num_layers=2,
                            upsample=True, activation="gelu", normalization=normalization)
        for i in range(len(self.context_dims_up) - 1)])

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
        #         if m.weight is not None:
        #             nn.init.constant_(m.weight, 1)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

    def forward(self, x, contexts):
        # print("Gaussian Upsampler")
        # print("Pre In: ", x.shape, contexts[-1].shape)
        x = torch.cat([x, contexts[-1]], dim=1)

        x = self.input_block(x)
        # print("Pre Out: ", x.shape)

        for i in range(self.num_levels):
            # print(f"Pre In {i}: {x.shape}, {contexts[-2 - i].shape}")
            x = self.up_blocks[i](x, contexts[-2 - i])
            # print(f"Pre Out {i}: {x.shape}")

        return x



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0, fmap_res=8):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)

        assert fmap_res == 4 or fmap_res == 8
        stride_layer_3 = fmap_res // 4
        dim_layer_2 = 128 if fmap_res == 4 else 96

        self.layer2 = self._make_layer(dim_layer_2, stride=2)
        self.layer3 = self._make_layer(128, stride=stride_layer_3)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x
