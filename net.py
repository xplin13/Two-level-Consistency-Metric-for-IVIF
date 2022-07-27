import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_nobn(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv_nobn = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv_nobn(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, n_feat=64, kernel_size=3, stride=2, bias=False, n_classes=2,
                 device=0):
        super(net, self).__init__()

        self.unet_dense = UNet_dense_softmax(n_channels=2, n_classes=1, bilinear=True)

        self.conv_in_vis = nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                     bias=bias)
        self.conv_in_ir = nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                    bias=bias)
        self.conv_in_fusion = nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                                    bias=bias)

        self.vis_degration1 = DoubleConv_nobn(out_channels, 32)
        self.vis_degration2 = OutConv(32, out_channels)
        self.ir_degration2 = OutConv(32, out_channels)

    def forward(self, images, images_ir):
        Input = torch.cat((images, images_ir), 1)
        output = self.unet_dense.forward(Input)

        vis_degration = self.vis_degration1(output)
        fusion_out = self.conv_in_fusion(output)
        ir_degration = fusion_out - vis_degration
        vis_degration = self.vis_degration2(vis_degration)
        ir_degration = self.ir_degration2(ir_degration)

        return output, ir_degration, vis_degration



#********************************************************UNET**********************************************************#
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

############################################ms_attention#################################################################

class Up_dense_1_softmax(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels_0, in_channels_1, in_channels_2, in_channels_3, out_channels, bilinear=True):
        super(Up_dense_1_softmax, self).__init__()
        self.maxpool_conv_0 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels_0, 512)
        )
        self.maxpool_conv_1 = nn.Sequential(
            nn.MaxPool2d(4),
            DoubleConv(in_channels_1, 512)
        )
        self.maxpool_conv_2 = nn.Sequential(
            nn.MaxPool2d(8),
            DoubleConv(in_channels_2, 512)
        )
        if bilinear:
            self.up_0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels_3, out_channels, in_channels_3 // 2)
        else:
            self.up_0 = nn.ConvTranspose2d(in_channels_3 , in_channels_3 // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels_3, out_channels)


    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.up_0(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x3 = self.maxpool_conv_0(x3)
        x4 = self.maxpool_conv_1(x4)
        x5 = self.maxpool_conv_2(x5)

        x_add = spatial_fusion(x2, x3, x4, x5, spatial_type='mean')
        x = torch.cat([x_add, x1], dim=1)
        return self.conv(x)

class Up_dense_2_softmax(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels_0, in_channels_1, in_channels_2, out_channels, bilinear=True):
        super(Up_dense_2_softmax, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels_0, out_channels, in_channels_0 // 2)
            self.conv_1 = DoubleConv(in_channels_0, out_channels, in_channels_0 // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels_0 , in_channels_0 // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels_0, out_channels)

        self.maxpool_conv_0 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels_1, out_channels)
        )
        self.maxpool_conv_1 = nn.Sequential(
            nn.MaxPool2d(4),
            DoubleConv(in_channels_2, out_channels)
        )

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.up(x1)
        x2 = self.up_1(x2)
        diffY = x3.size()[2] - x1.size()[2]
        diffX = x3.size()[3] - x1.size()[3]
        diffM = x3.size()[2] - x2.size()[2]
        diffN = x3.size()[3] - x2.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x2 = F.pad(x2, [diffN // 2, diffN - diffN // 2,
                        diffM // 2, diffM - diffM // 2])

        x1 = self.conv(x1)
        x2 = self.conv_1(x2)

        x4 = self.maxpool_conv_0(x4)
        x5 = self.maxpool_conv_1(x5)

        x_add = spatial_fusion(x2, x3, x4, x5, spatial_type='mean')
        x = torch.cat([x_add, x1], dim=1)
        return self.conv(x)

class Up_dense_3_softmax(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels_0, in_channels_1,  in_channels_2, out_channels, bilinear=True):
        super(Up_dense_3_softmax, self).__init__()
        if bilinear:
            self.up_0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up_1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels_0, out_channels, in_channels_0 // 2)
            self.conv_1 = DoubleConv(in_channels_1, out_channels, in_channels_1 // 2)
            self.conv_2 = DoubleConv(in_channels_0, out_channels, in_channels_0 // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels_0 , in_channels_0 // 2, kernel_size=2, stride=2)
            self.conv_0 = DoubleConv(in_channels_0, out_channels)

        self.maxpool_conv_0 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels_2, out_channels)
        )


    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.up_0(x1)
        x2 = self.up_1(x2)
        x3 = self.up_2(x3)
        diffY = x4.size()[2] - x1.size()[2]
        diffX = x4.size()[3] - x1.size()[3]
        diffM = x4.size()[2] - x2.size()[2]
        diffN = x4.size()[3] - x2.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x2 = F.pad(x2, [diffN // 2, diffN - diffN // 2,
                        diffM // 2, diffM - diffM // 2])
        x3 = F.pad(x3, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])


        x1 = self.conv(x1)
        x2 = self.conv_1(x2)
        x3 = self.conv_2(x3)

        x5 = self.maxpool_conv_0(x5)

        x_add = spatial_fusion(x2, x3, x4, x5, spatial_type='mean')
        x = torch.cat([x_add, x1], dim=1)
        return self.conv(x)

class Up_dense_4_softmax(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels_0, in_channels_1, in_channels_2, out_channels, bilinear=True):
        super(Up_dense_4_softmax, self).__init__()
        if bilinear:
            self.up_0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up_1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            self.up_2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.up_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels_0, out_channels, in_channels_0 // 2)
            self.conv_1 = DoubleConv(in_channels_1, out_channels, in_channels_1 // 2)
            self.conv_2 = DoubleConv(in_channels_2, out_channels, in_channels_1 // 2)
            self.conv_3 = DoubleConv(in_channels_0, out_channels, in_channels_0 // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels_0 , in_channels_0 // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels_0, out_channels)


    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.up_0(x1)
        x2 = self.up_1(x2)
        x3 = self.up_2(x3)
        x4 = self.up_3(x4)
        diffY = x5.size()[2] - x1.size()[2]
        diffX = x5.size()[3] - x1.size()[3]
        diffM = x5.size()[2] - x2.size()[2]
        diffN = x5.size()[3] - x2.size()[3]
        diffA = x5.size()[2] - x3.size()[2]
        diffB = x5.size()[3] - x3.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x2 = F.pad(x2, [diffN // 2, diffN - diffN // 2,
                        diffM // 2, diffM - diffM // 2])
        x3 = F.pad(x3, [diffB // 2, diffB - diffB // 2,
                        diffA // 2, diffA - diffA // 2])
        x4 = F.pad(x4, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x1 = self.conv(x1)
        x2 = self.conv_1(x2)
        x3 = self.conv_2(x3)
        x4 = self.conv_3(x4)

        x_add = spatial_fusion(x2, x3, x4, x5, spatial_type='mean')
        x = torch.cat([x_add, x1], dim=1)
        return self.conv(x)


class UNet_dense_softmax(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_dense_softmax, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up_dense_1_softmax(256, 128, 64, 1024,  1024 // factor, bilinear)
        self.up2 = Up_dense_2_softmax(512, 128, 64, 512 // factor, bilinear)
        self.up3 = Up_dense_3_softmax(256, 512, 64, 256 // factor, bilinear)
        self.up4 = Up_dense_4_softmax(128, 512, 256, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4, x3, x2, x1)
        x = self.up2(x, x4, x3, x2, x1)
        x = self.up3(x, x4, x3, x2, x1)
        x = self.up4(x, x4, x3, x2, x1)
        logits = self.outc(x)
        return logits

###############################ms_add#################################################

EPSILON = 1e-5


def spatial_fusion(tensor1, tensor2, tensor3, tensor4, spatial_type='mean'):
    shape = tensor1.size()
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)
    spatial3 = spatial_attention(tensor3, spatial_type)
    spatial4 = spatial_attention(tensor4, spatial_type)

    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + torch.exp(spatial3) + torch.exp(spatial4) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + torch.exp(spatial3) + torch.exp(spatial4) + EPSILON)
    spatial_w3 = torch.exp(spatial3) / (torch.exp(spatial1) + torch.exp(spatial2) + torch.exp(spatial3) + torch.exp(spatial4) + EPSILON)
    spatial_w4 = torch.exp(spatial4) / (torch.exp(spatial1) + torch.exp(spatial2) + torch.exp(spatial3) + torch.exp(spatial4) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)
    spatial_w3 = spatial_w3.repeat(1, shape[1], 1, 1)
    spatial_w4 = spatial_w4.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2 + spatial_w3 * tensor3 + spatial_w4 * tensor4

    return tensor_f

def spatial_attention(tensor, spatial_type='mean'):
    spatial = []
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial
