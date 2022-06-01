
import warnings
from collections import Iterable

import torch.nn as nn
import torch
import copy
from functools  import partial
import math
from utils.frame import OneStageModelFrame,OneStageClassifier,PACK,IDetection

# <editor-fold desc='激活函数'>
# Swish
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.as_tensor(1.0))

    def forward(self, x):
        x = x * torch.sigmoid(x * self.beta)
        return x


# SiLU
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


# Mish
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


# Relu6
class ReLU6(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.clamp(x, min=0, max=6)
        return x


class HSiLU(nn.Module):
    def forward(self, x):
        out = x * (torch.clamp(x, min=-3, max=3) / 6 + 0.5)
        return out


class HSigmoid(nn.Module):
    def forward(self, x):
        out = torch.clamp(x, min=-3, max=3) / 6 + 0.5
        return out


class ACT:
    LK = 'lk'
    RELU = 'relu'
    SIG = 'sig'
    RELU6 = 'relu6'
    MISH = 'mish'
    SILU = 'silu'
    HSILU = 'hsilu'
    HSIG = 'hsig'
    SWISH = 'swish'
    TANH = 'tanh'
    NONE = None

    @staticmethod
    def build(act_name=None):
        if isinstance(act_name, nn.Module):
            act = act_name
        elif act_name is None or act_name == '':
            act = None
        elif act_name == ACT.LK:
            act = nn.LeakyReLU(0.1)
        elif act_name == ACT.RELU:
            act = nn.ReLU(inplace=True)
        elif act_name == ACT.SIG:
            act = nn.Sigmoid()
        elif act_name == ACT.SWISH:
            act = Swish()
        elif act_name == ACT.RELU6:
            act = ReLU6()
        elif act_name == ACT.MISH:
            act = Mish()
        elif act_name == ACT.SILU:
            act = SiLU()
        elif act_name == ACT.HSILU:
            act = HSiLU()
        elif act_name == ACT.HSIG:
            act = HSigmoid()
        elif act_name == ACT.TANH:
            act = nn.Tanh()
        else:
            raise Exception('err act name' + str(act_name))
        return act


# </editor-fold>


# <editor-fold desc='卷积子模块'>
# Conv+BN+Act
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        super(Conv, self).__init__()
        # kernel_size = kernel_size if isinstance(kernel_size, Iterable) else (kernel_size, kernel_size)
        # stride = stride if isinstance(stride, Iterable) else (stride, stride)
        # dilation = dilation if isinstance(dilation, Iterable) else (dilation, dilation)
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride), padding=(padding, padding), dilation=(dilation, dilation),
            bias=not bn, groups=groups, **kwargs)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03) if bn else None
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = ACT.build(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.act(x) if self.act else x
        return x


# Conv+BN+Act padding=0
class ConvNP(Conv):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        padding = 0
        super(ConvNP, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=padding, bn=bn, act=act, **kwargs)


# Conv+BN+Act padding=auto
class ConvAP(Conv):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        padding = (kernel_size - 1) * dilation // 2
        super(ConvAP, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=padding, bn=bn, act=act)


class Conv1(Conv):
    def __init__(self, in_channels, out_channels, bn=True, act=None, **kwargs):
        super(Conv1, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
            groups=1, bn=bn, act=act, **kwargs)


class Conv1Pure(Conv):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv1Pure, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, dilation=1,
            bn=False, act=None, **kwargs)


class Conv3(Conv):
    def __init__(self, in_channels, out_channels, bn=True, act=None, **kwargs):
        super(Conv3, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
            groups=1, bn=bn, act=act, **kwargs)


class Conv3Pure(Conv):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv3Pure, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
            bn=False, act=None, **kwargs)


class ConvAPDW(ConvAP):
    def __init__(self, channels, kernel_size, stride, bn=True, act=None, **kwargs):
        super(ConvAPDW, self).__init__(
            in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=stride, dilation=1,
            groups=channels, bn=bn, act=act, **kwargs)


# ConvTrans+BN+Act
class ConvT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 output_padding=0, bn=True, act=None, **kwargs):
        super(ConvT, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=(kernel_size, kernel_size),
                                       stride=(stride, stride), groups=groups, padding=(padding, padding),
                                       output_padding=(output_padding, output_padding),
                                       dilation=dilation, bias=not bn, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = ACT.build(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# Conv+BN+Act padding=auto
class ConvTAP(ConvT):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1,
                 bn=True, act=None, **kwargs):
        padding = (kernel_size - 1) // 2
        output_padding = kernel_size % 2
        super(ConvTAP, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
            dilation=dilation, groups=groups, padding=padding, output_padding=output_padding, bn=bn, act=act, **kwargs)


# </editor-fold>

class AnchorLayer(nn.Module):
    def __init__(self, anchors, stride, feat_size=(0, 0)):
        super(AnchorLayer, self).__init__()
        self.anchors = anchors
        self.stride = stride
        self.feat_size = feat_size

    @property
    def anchors(self):
        return self._anchors

    @anchors.setter
    def anchors(self, anchors):
        anchors = torch.Tensor(anchors)
        if len(anchors.size()) == 1:
            anchors = torch.unsqueeze(anchors, dim=0)
        self._anchors = anchors
        self.Na = anchors.size(0)

    @staticmethod
    def generate_offset(Wf, Hf, anchors):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Na = anchors.size(0)  # (Na,2)
            x = torch.arange(Wf)[None, :].repeat(Hf, 1)
            y = torch.arange(Hf)[:, None].repeat(1, Wf)
            xy_offset = torch.stack([x, y], dim=2)  # (Hf,Wf,2)
            if Na == 1:
                wh_offset = anchors[None, :, :].expand(Hf, Wf, 2).contiguous()  # (Hf,Wf,2)
                return xy_offset, wh_offset
            else:
                xy_offset = xy_offset.unsqueeze(dim=2).expand(Hf, Wf, Na, 2).contiguous()  # (Hf, Wf, Na, 2)
                wh_offset = anchors[None, None, :, :].expand(Hf, Wf, Na, 2).contiguous()  # (Hf, Wf, Na, 2)
                return xy_offset, wh_offset

    @staticmethod
    def generate_pboxes(xy_offset, wh_offset, stride):
        pboxes = torch.cat([(xy_offset + 0.5) * stride, wh_offset], dim=-1)
        xc, yc, w, h = pboxes[..., 0], pboxes[..., 1], pboxes[..., 2], pboxes[..., 3]
        pboxes[..., 0], pboxes[..., 2] = xc - w / 2, xc + w / 2
        pboxes[..., 1], pboxes[..., 3] = yc - h / 2, yc + h / 2
        pboxes = pboxes.reshape(-1, 4)
        return pboxes

    @property
    def feat_size(self):
        return (self.Wf, self.Hf)

    @feat_size.setter
    def feat_size(self, feat_size):
        (self.Wf, self.Hf) = feat_size if isinstance(feat_size, Iterable) else (feat_size, feat_size)
        self.xy_offset, self.wh_offset = AnchorLayer.generate_offset(self.Wf, self.Hf, self.anchors)
        self.pboxes = AnchorLayer.generate_pboxes(self.xy_offset, self.wh_offset, self.stride)
        self.num_pbx = self.Na * self.Wf * self.Hf

class AnchorLayerImg(AnchorLayer):
    def __init__(self, anchors, stride, img_size=(0, 0)):
        super().__init__(anchors, stride, AnchorLayerImg.calc_feat_size(img_size, stride))

    @staticmethod
    def calc_feat_size(img_size, stride):
        (W, H) = img_size if isinstance(img_size, Iterable) else (img_size, img_size)
        return (int(math.ceil(W / stride)), int(math.ceil(H / stride)))

    @property
    def img_size(self):
        return (self.Wf * self.stride, self.Hf * self.stride)

    @img_size.setter
    def img_size(self, img_size):
        self.feat_size = AnchorLayerImg.calc_feat_size(img_size, self.stride)

def init_sig(bias, prior_prob=0.1):
    nn.init.constant_(bias.data, -math.log((1 - prior_prob) / prior_prob))
    return None

def Focalloss(pred, target, alpha=0.5, gamma=2, reduction='sum'):
    loss_ce = -target * torch.log(pred + 1e-16) - (1 - target) * torch.log(1 - pred + 1e-16)
    prop = target * pred + (1 - target) * (1 - pred)
    alpha_full = target * alpha + (1 - target) * (1 - alpha)
    loss = loss_ce * alpha_full * (1 - prop) ** gamma
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('err reduction')
