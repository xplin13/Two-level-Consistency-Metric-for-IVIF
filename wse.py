from torch.nn.modules.module import Module
from torch.nn import _reduction as _Reduction
import warnings
import torch
import numpy as np

def low_freq_mutate_np( amp_src):
    a_src = np.fft.fftshift(amp_src, axes=(-2, -1))
    batch, _, h, w = a_src.shape
    b = 3
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    mask = np.ones((batch, 1, h, w), np.uint8)
    mask[:, :, h1:h2, w1:w2] = 0

    mask_img_1 = a_src * mask + a_src
    a_src = np.fft.ifftshift(mask_img_1, axes=(-2, -1))
    return a_src


def FA(src_img):
    src_img_np = src_img.cuda().data.cpu().numpy()
    fft_src_np = np.fft.fft2(src_img_np, axes=(-2, -1))
    amp_src, pha_src = np.abs(fft_src_np), np.angle(fft_src_np)

    amp_src_ = low_freq_mutate_np(amp_src)
    fft_src_ = amp_src_ * np.exp(1j * pha_src)

    src_in_trg = np.fft.ifft2(fft_src_, axes=(-2, -1))
    src_in_trg = np.real(src_in_trg)
    src_in_trg = src_in_trg.astype('float32')
    src_in_trg = torch.from_numpy(src_in_trg)

    return src_in_trg

class _Loss(Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class WSELoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input_0, target_0, input_1, target_1):
        ret = self.wse_loss(input_0, target_0, input_1, target_1)
        return ret



    def wse_loss(self, input, target, input_1, target_1, size_average=None, reduce=None, reduction='mean'):
        if not (target.size() == input.size()):
            warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                          "This will likely lead to incorrect results due to broadcasting. "
                          "Please ensure they have the same size.".format(target.size(), input.size()),
                          stacklevel=2)
        enhance_target = FA(target)
        enhance_target = enhance_target.cuda()
        ret = (input - enhance_target) ** 2

        enhance_target_1 = FA(target_1)
        enhance_target_1 = enhance_target_1.cuda()
        ret_1 = (input_1 - enhance_target_1) ** 2

        batch, c, h, w = enhance_target.shape
        mask_ret = enhance_target.view(batch, c, -1)
        mask_ret = torch.softmax(mask_ret, dim=2)
        mask_ret = mask_ret.view(batch, c, h, w)

        batch1, c1, h1, w1 = enhance_target_1.shape
        mask_ret1 = enhance_target_1.view(batch1, c1, -1)
        mask_ret1 = torch.softmax(mask_ret1, dim=2)
        mask_ret1 = mask_ret1.view(batch1, c1, h1, w1)

        ret = mask_ret * ret
        ret_1 = mask_ret1 * ret_1
        ret_out = torch.sum(ret) + torch.sum(ret_1)

        return ret_out