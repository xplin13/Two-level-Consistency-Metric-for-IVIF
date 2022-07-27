import torch
import torch.nn.functional as F
from math import exp
import numpy as np
import torch.nn as nn

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


class TV_Loss(nn.Module):
	def __init__(self):
		super(TV_Loss,self).__init__()

	def forward(self,IA,IF):
		r=IA-IF
		batch_size=r.shape[0]
		h=r.shape[2]
		w=r.shape[3]
		tv1=torch.pow((r[:,:,1:,:]-r[:,:,:h-1,:]),2).mean()
		tv2=torch.pow((r[:,:,:,1:]-r[:,:,:,:w-1]),2).mean()
		return tv1+tv2

class SSIM_Loss(nn.Module):
	def __init__(self,lam=1000,size=11,stride=11,C=9e-4,use_ssim=True,use_tv=True):
		super().__init__()
		self.lam=lam
		self.size=size
		self.use_ssim=use_ssim
		self.use_tv=use_tv
		self.stride=stride
		self.C=C
		self.TV=TV_Loss()

	def forward(self,IA,IB,IF):

		if self.use_tv:
			TV1=self.TV(IA,IF)
			TV2=self.TV(IB,IF)
		if self.use_ssim:

			window=torch.ones((1,1,self.size,self.size))/(self.size*self.size)
			window=window.cuda()

			mean_IA=F.conv2d(IA,window,stride=self.stride)
			mean_IB=F.conv2d(IB,window,stride=self.stride)
			mean_IF=F.conv2d(IF,window,stride=self.stride)

			mean_IA_2=F.conv2d(torch.pow(IA,2),window,stride=self.stride)
			mean_IB_2=F.conv2d(torch.pow(IB,2),window,stride=self.stride)
			mean_IF_2=F.conv2d(torch.pow(IF,2),window,stride=self.stride)

			var_IA=mean_IA_2-torch.pow(mean_IA,2)
			var_IB=mean_IB_2-torch.pow(mean_IB,2)
			var_IF=mean_IF_2-torch.pow(mean_IF,2)

			mean_IAIF=F.conv2d(IA*IF,window,stride=self.stride)
			mean_IBIF=F.conv2d(IB*IF,window,stride=self.stride)

			sigma_IAIF=mean_IAIF-mean_IA*mean_IF
			sigma_IBIF=mean_IBIF-mean_IB*mean_IF

			C=torch.ones(sigma_IAIF.shape)*self.C
			C=C.cuda()

			ssim_IAIF=(2*sigma_IAIF+C)/(var_IA+var_IF+C)
			ssim_IBIF=(2*sigma_IBIF+C)/(var_IB+var_IF+C)

			score=ssim_IAIF*(mean_IA>mean_IB)+ssim_IBIF*(mean_IA<=mean_IB)
			ssim=1-torch.mean(score)

		if self.use_ssim and self.use_tv:
			return self.lam*ssim+TV1
		elif self.use_ssim and not self.use_tv:
			return self.lam*ssim
		elif not self.use_ssim and self.use_tv:
			return (TV1+TV2)/2
		else:
			return 0

