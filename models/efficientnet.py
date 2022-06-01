from .modules import *


class SeModule(nn.Module):
    def __init__(self, channels, ratio=0.25, act=ACT.SILU):
        super(SeModule, self).__init__()
        inner_channels = int(ratio * channels)
        self.se_pth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv1(in_channels=channels, out_channels=inner_channels, bn=False, act=act),
            Conv1(in_channels=inner_channels, out_channels=channels, bn=False, act=ACT.SIG)
        )

    def forward(self, x):
        return x * self.se_pth(x)


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, ratio_exp, ratio_se, kernel_size, stride, act=ACT.RELU):
        super(MBConv, self).__init__()
        inner_channels = int(ratio_exp * in_channels)
        self.ratio_exp = ratio_exp
        self.expend = None if ratio_exp == 1 else Conv1(in_channels=in_channels, out_channels=inner_channels, act=act)
        self.conv2 = ConvAPDW(channels=inner_channels, stride=stride, kernel_size=kernel_size, act=act)
        self.se = SeModule(channels=inner_channels, ratio=ratio_se, act=act)
        self.conv3 = Conv1(in_channels=inner_channels, out_channels=out_channels, act=None)
        self.has_shortcut = stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = self.expend(x) if self.expend is not None else x
        out = self.conv2(out)
        out = self.se(out)
        out = self.conv3(out)
        out = out + x if self.has_shortcut else out
        return out


class EfficientNetBkbn(nn.Module):
    def __init__(self, ratio_depth, ratio_width, act=ACT.SILU):
        super(EfficientNetBkbn, self).__init__()
        repeat_nums = [EfficientNetBkbn.expend_repeat(repeat_num, ratio_depth)
                       for repeat_num in [1, 2, 2, 3, 3, 4, 1]]
        channelss = [EfficientNetBkbn.expend_channels(channels, ratio_width)
                     for channels in [16, 24, 40, 80, 112, 192, 320]]

        self.pre = ConvAP(in_channels=3, out_channels=32, kernel_size=3, stride=2, act=act)
        self.stage1 = EfficientNetBkbn.MBConvRepeat(
            in_channels=32, out_channels=channelss[0], ratio_exp=1, ratio_se=0.25, kernel_size=3, stride=1,
            repeat_num=repeat_nums[0], act=act)
        self.stage2 = EfficientNetBkbn.MBConvRepeat(
            in_channels=channelss[0], out_channels=channelss[1], ratio_exp=6, ratio_se=0.25, kernel_size=3, stride=2,
            repeat_num=repeat_nums[1], act=act)
        self.stage3 = EfficientNetBkbn.MBConvRepeat(
            in_channels=channelss[1], out_channels=channelss[2], ratio_exp=6, ratio_se=0.25, kernel_size=5, stride=2,
            repeat_num=repeat_nums[2], act=act)
        self.stage4 = EfficientNetBkbn.MBConvRepeat(
            in_channels=channelss[2], out_channels=channelss[3], ratio_exp=6, ratio_se=0.25, kernel_size=3, stride=2,
            repeat_num=repeat_nums[3], act=act)
        self.stage5 = EfficientNetBkbn.MBConvRepeat(
            in_channels=channelss[3], out_channels=channelss[4], ratio_exp=6, ratio_se=0.25, kernel_size=5, stride=1,
            repeat_num=repeat_nums[4], act=act)
        self.stage6 = EfficientNetBkbn.MBConvRepeat(
            in_channels=channelss[4], out_channels=channelss[5], ratio_exp=6, ratio_se=0.25, kernel_size=5, stride=2,
            repeat_num=repeat_nums[5], act=act)
        self.stage7 = EfficientNetBkbn.MBConvRepeat(
            in_channels=channelss[5], out_channels=channelss[6], ratio_exp=6, ratio_se=0.25, kernel_size=3, stride=1,
            repeat_num=repeat_nums[6], act=act)

    @staticmethod
    def expend_channels(channels, ratio_width):
        divisor = 8
        channels = channels * ratio_width
        channels_new = max(divisor, int(channels + divisor / 2) // divisor * divisor)
        channels_new = int(channels_new + divisor) if channels_new < 0.9 * channels else int(channels_new)
        return channels_new

    @staticmethod
    def expend_repeat(repeat_num, ratio_depth):
        return int(math.ceil(ratio_depth * repeat_num))

    @staticmethod
    def MBConvRepeat(in_channels, out_channels, ratio_exp, ratio_se, kernel_size, stride, repeat_num=1, act=ACT.SILU):
        backbone = []
        backbone.append(
            MBConv(in_channels=in_channels, out_channels=out_channels, ratio_exp=ratio_exp, ratio_se=ratio_se,
                   kernel_size=kernel_size, stride=stride, act=act))
        for i in range(1, repeat_num):
            backbone.append(MBConv(in_channels=out_channels, out_channels=out_channels, ratio_exp=ratio_exp,
                                   ratio_se=ratio_se, kernel_size=kernel_size, stride=1, act=act))
        backbone = nn.Sequential(*backbone)
        return backbone

    def forward(self, imgs):
        feat0 = self.pre(imgs)
        feat1 = self.stage1(feat0)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        feat5 = self.stage5(feat4)
        feat6 = self.stage6(feat5)
        feat7 = self.stage7(feat6)
        return feat7

    B0_PARA = dict(ratio_depth=1.0, ratio_width=1.0)
    B1_PARA = dict(ratio_depth=1.0, ratio_width=1.1)
    B2_PARA = dict(ratio_depth=1.1, ratio_width=1.2)
    B3_PARA = dict(ratio_depth=1.2, ratio_width=1.4)
    B4_PARA = dict(ratio_depth=1.4, ratio_width=1.8)
    B5_PARA = dict(ratio_depth=1.6, ratio_width=2.2)
    B6_PARA = dict(ratio_depth=1.8, ratio_width=2.6)
    B7_PARA = dict(ratio_depth=2.0, ratio_width=3.1)
    B8_PARA = dict(ratio_depth=2.2, ratio_width=3.6)

    @staticmethod
    def B0(act=ACT.SILU):
        return EfficientNetBkbn(**EfficientNetBkbn.B0_PARA, act=act)

    @staticmethod
    def B1(act=ACT.SILU):
        return EfficientNetBkbn(**EfficientNetBkbn.B1_PARA, act=act)

    @staticmethod
    def B2(act=ACT.SILU):
        return EfficientNetBkbn(**EfficientNetBkbn.B2_PARA, act=act)

    @staticmethod
    def B3(act=ACT.SILU):
        return EfficientNetBkbn(**EfficientNetBkbn.B3_PARA, act=act)

    @staticmethod
    def B4(act=ACT.SILU):
        return EfficientNetBkbn(**EfficientNetBkbn.B4_PARA, act=act)

    @staticmethod
    def B5(act=ACT.SILU):
        return EfficientNetBkbn(**EfficientNetBkbn.B5_PARA, act=act)

    @staticmethod
    def B6(act=ACT.SILU):
        return EfficientNetBkbn(**EfficientNetBkbn.B6_PARA, act=act)

    @staticmethod
    def B7(act=ACT.SILU):
        return EfficientNetBkbn(**EfficientNetBkbn.B7_PARA, act=act)

    @staticmethod
    def B8(act=ACT.SILU):
        return EfficientNetBkbn(**EfficientNetBkbn.B8_PARA, act=act)


class EfficientNetMain(EfficientNetBkbn):
    def __init__(self, ratio_depth, ratio_width, act=ACT.SILU, num_cls=0):
        super(EfficientNetMain, self).__init__(ratio_depth=ratio_depth, ratio_width=ratio_width, act=act)
        self.num_cls = num_cls
        in_channels = EfficientNetBkbn.expend_channels(320, ratio_width)
        inner_channels = EfficientNetBkbn.expend_channels(1280, ratio_width)
        self.conv8 = Conv1(in_channels=in_channels, out_channels=inner_channels, act=act)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features=inner_channels, out_features=num_cls)

    def forward(self, imgs):
        feat = super(EfficientNetMain, self).forward(imgs)
        feat = self.conv8(feat)
        feat = self.pool(feat)
        # feat = feat.squeeze(dim=3).squeeze(dim=2)
        feat = feat.view(list(feat.size())[:2])
        feat = self.dropout(feat)
        feat = self.linear(feat)
        return feat

    @staticmethod
    def B0(act=ACT.SILU, num_cls=10):
        return EfficientNetMain(**EfficientNetBkbn.B0_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def B1(act=ACT.SILU, num_cls=10):
        return EfficientNetMain(**EfficientNetBkbn.B1_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def B2(act=ACT.SILU, num_cls=10):
        return EfficientNetMain(**EfficientNetBkbn.B2_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def B3(act=ACT.SILU, num_cls=10):
        return EfficientNetMain(**EfficientNetBkbn.B3_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def B4(act=ACT.SILU, num_cls=10):
        return EfficientNetMain(**EfficientNetBkbn.B4_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def B5(act=ACT.SILU, num_cls=10):
        return EfficientNetMain(**EfficientNetBkbn.B5_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def B6(act=ACT.SILU, num_cls=10):
        return EfficientNetMain(**EfficientNetBkbn.B6_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def B7(act=ACT.SILU, num_cls=10):
        return EfficientNetMain(**EfficientNetBkbn.B7_PARA, act=act, num_cls=num_cls)


class EfficientNet(OneStageClassifier):
    def __init__(self, backbone, device=None, pack=None, img_size=(224, 224)):
        num_cls = backbone.num_cls
        super().__init__(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def B0(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = EfficientNetMain.B0(act=ACT.SILU, num_cls=num_cls)
        return EfficientNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def B1(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = EfficientNetMain.B1(act=ACT.SILU, num_cls=num_cls)
        return EfficientNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def B2(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = EfficientNetMain.B2(act=ACT.SILU, num_cls=num_cls)
        return EfficientNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def B3(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = EfficientNetMain.B3(act=ACT.SILU, num_cls=num_cls)
        return EfficientNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def B4(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = EfficientNetMain.B4(act=ACT.SILU, num_cls=num_cls)
        return EfficientNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def B5(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = EfficientNetMain.B5(act=ACT.SILU, num_cls=num_cls)
        return EfficientNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def B6(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = EfficientNetMain.B6(act=ACT.SILU, num_cls=num_cls)
        return EfficientNet(backbone=backbone, device=device, pack=pack, img_size=img_size)


if __name__ == '__main__':
    # model = EfficientNetBkbn.B3()
    model = EfficientNetMain(**EfficientNetBkbn.B0_PARA, num_cls=20)
    testx = torch.zeros(1, 3, 224, 224)
    y = model(testx)
