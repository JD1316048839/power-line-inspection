from .modules import *


class Conv3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn=True, act=None):
        super(Conv3, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
            stride=(stride, stride), padding=1, bias=False, groups=1)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = ACT.build(act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn else x
        x = self.act(x) if self.act else x
        return x


class Conv3DWAP(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bn=True, act=None):
        super(Conv3DWAP, self).__init__()
        print(in_channels, out_channels)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
            stride=(stride, stride), padding=1, bias=False, groups=out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=(1, 1),
            stride=(1, 1), padding=0, bias=False, groups=1)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.act = ACT.build(act)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x) if self.bn else x
        x = self.act(x) if self.act else x
        return x


class SeModule(nn.Module):
    def __init__(self, channels, ratio=0.25):
        super(SeModule, self).__init__()
        inner_channels = int(ratio * channels)
        self.se_pth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv1(in_channels=channels, out_channels=inner_channels, bn=False, act=ACT.HSIG)
        )

    def forward(self, x):
        return x * self.se_pth(x)


class OSABlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels, conv_num, se=False, depthwise=False,
                 identity=False, act=ACT.RELU):
        super(OSABlock, self).__init__()
        self.identity = identity
        trans = depthwise and in_channels != inner_channels
        self.trans_conv = Conv1(in_channels=in_channels, out_channels=inner_channels, act=act) if trans else None

        last_channels = inner_channels if trans else in_channels
        ConvType = Conv3DWAP if depthwise else Conv3
        convs = []
        for i in range(conv_num):
            convs.append(ConvType(in_channels=last_channels, out_channels=inner_channels, bn=True, act=act))
            last_channels = inner_channels
        self.convs = nn.ModuleList(convs)
        concat_channels = in_channels + conv_num * inner_channels
        self.concater = Conv1(in_channels=concat_channels, out_channels=out_channels, act=act)
        self.se = SeModule(channels=out_channels, ratio=1) if se else None

    def forward(self, x):
        residual = x
        outputs = [x]
        x = self.trans_conv(x) if self.trans_conv is not None else x
        for conv in self.convs:
            x = conv(x)
            outputs.append(x)
        x = torch.cat(outputs, dim=1)
        x = self.concater(x)
        x = self.se(x) if self.se is not None else x
        x = x + residual if self.identity else x
        return x


class VoVNetBkbn(nn.Module):
    def __init__(self, pre_channelss, channelss, inner_channelss, conv_num, repeat_nums, se, depthwise, act=ACT.RELU):
        super(VoVNetBkbn, self).__init__()
        ConvType = Conv3DWAP if depthwise else Conv3
        self.pre = nn.Sequential(
            Conv3(in_channels=3, out_channels=pre_channelss[0], stride=2, act=act),
            ConvType(in_channels=pre_channelss[0], out_channels=pre_channelss[1], stride=1, act=act),
            ConvType(in_channels=pre_channelss[1], out_channels=pre_channelss[2], stride=2, act=act),
        )
        self.stage1 = VoVNetBkbn.OSABlockRepeat(
            in_channels=pre_channelss[2], inner_channels=inner_channelss[0], out_channels=channelss[0],
            conv_num=conv_num, repeat_num=repeat_nums[0], pool=False, se=se, depthwise=depthwise, act=act)
        self.stage2 = VoVNetBkbn.OSABlockRepeat(
            in_channels=channelss[0], inner_channels=inner_channelss[1], out_channels=channelss[1],
            conv_num=conv_num, repeat_num=repeat_nums[1], pool=True, se=se, depthwise=depthwise, act=act)
        self.stage3 = VoVNetBkbn.OSABlockRepeat(
            in_channels=channelss[1], inner_channels=inner_channelss[2], out_channels=channelss[2],
            conv_num=conv_num, repeat_num=repeat_nums[2], pool=True, se=se, depthwise=depthwise, act=act)
        self.stage4 = VoVNetBkbn.OSABlockRepeat(
            in_channels=channelss[2], inner_channels=inner_channelss[3], out_channels=channelss[3],
            conv_num=conv_num, repeat_num=repeat_nums[3], pool=True, se=se, depthwise=depthwise, act=act)

    @staticmethod
    def OSABlockRepeat(in_channels, inner_channels, out_channels, conv_num=1, repeat_num=1, pool=True, se=False,
                       depthwise=False, act=ACT.RELU):
        backbone = [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)] if pool else []
        last_channels = in_channels
        for i in range(repeat_num):
            backbone.append(
                OSABlock(in_channels=last_channels, inner_channels=inner_channels, out_channels=out_channels,
                         conv_num=conv_num, se=se and i == (repeat_num - 1), depthwise=depthwise, act=act))
            last_channels = out_channels
        return nn.Sequential(*backbone)

    def forward(self, imgs):
        feat = self.pre(imgs)
        feat1 = self.stage1(feat)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        return feat4

    V19SDW_PARA = dict(pre_channelss=(64, 64, 64), channelss=(112, 256, 384, 512), inner_channelss=(64, 80, 96, 112),
                       conv_num=3, repeat_nums=(1, 1, 1, 1), se=True, depthwise=True)
    V19DW_PARA = dict(pre_channelss=(64, 64, 64), channelss=(256, 512, 768, 1024), inner_channelss=(128, 160, 192, 224),
                      conv_num=3, repeat_nums=(1, 1, 1, 1), se=True, depthwise=True)
    V19S_PARA = dict(pre_channelss=(64, 64, 128), channelss=(112, 256, 384, 512), inner_channelss=(64, 80, 96, 112),
                     conv_num=3, repeat_nums=(1, 1, 1, 1), se=True, depthwise=False)
    V19_PARA = dict(pre_channelss=(64, 64, 128), channelss=(256, 512, 768, 1024), inner_channelss=(128, 160, 192, 224),
                    conv_num=3, repeat_nums=(1, 1, 1, 1), se=True, depthwise=False)

    V39_PARA = dict(pre_channelss=(64, 64, 128), channelss=(256, 512, 768, 1024), inner_channelss=(128, 160, 192, 224),
                    conv_num=5, repeat_nums=(1, 1, 2, 2), se=True, depthwise=False)
    V57_PARA = dict(pre_channelss=(64, 64, 128), channelss=(256, 512, 768, 1024), inner_channelss=(128, 160, 192, 224),
                    conv_num=5, repeat_nums=(1, 1, 4, 3), se=True, depthwise=False)
    V99_PARA = dict(pre_channelss=(64, 64, 128), channelss=(256, 512, 768, 1024), inner_channelss=(128, 160, 192, 224),
                    conv_num=5, repeat_nums=(1, 1, 9, 3), se=True, depthwise=False)

    @staticmethod
    def V19SDW(act=ACT.RELU):
        return VoVNetBkbn(**VoVNetBkbn.V19SDW_PARA, act=act)

    @staticmethod
    def V19DW(act=ACT.RELU):
        return VoVNetBkbn(**VoVNetBkbn.V19DW_PARA, act=act)

    @staticmethod
    def V19S(act=ACT.RELU):
        return VoVNetBkbn(**VoVNetBkbn.V19S_PARA, act=act)

    @staticmethod
    def V19(act=ACT.RELU):
        return VoVNetBkbn(**VoVNetBkbn.V19_PARA, act=act)

    @staticmethod
    def V39(act=ACT.RELU):
        return VoVNetBkbn(**VoVNetBkbn.V39_PARA, act=act)

    @staticmethod
    def V57(act=ACT.RELU):
        return VoVNetBkbn(**VoVNetBkbn.V57_PARA, act=act)

    @staticmethod
    def V99(act=ACT.RELU):
        return VoVNetBkbn(**VoVNetBkbn.V99_PARA, act=act)


class VoVNetMain(VoVNetBkbn):
    def __init__(self, pre_channelss, channelss, inner_channelss, conv_num, repeat_nums, se, depthwise, num_cls=10,
                 act=ACT.RELU):
        super(VoVNetMain, self).__init__(pre_channelss=pre_channelss, channelss=channelss,
                                         inner_channelss=inner_channelss, conv_num=conv_num, repeat_nums=repeat_nums,
                                         se=se, depthwise=depthwise, act=act)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.num_cls = num_cls
        self.linear = nn.Linear(in_features=channelss[3], out_features=num_cls)

    def forward(self, imgs):
        feat = self.pre(imgs)
        feat1 = self.stage1(feat)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        feat4 = self.pool(feat4)
        feat4 = feat4.squeeze(dim=-1).squeeze(dim=-1)
        cls = self.linear(feat4)
        return cls

    @staticmethod
    def V19SDW(num_cls=10, act=ACT.RELU):
        return VoVNetMain(**VoVNetBkbn.V19SDW_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def V19DW(num_cls=10, act=ACT.RELU):
        return VoVNetMain(**VoVNetBkbn.V19DW_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def V19S(num_cls=10, act=ACT.RELU):
        return VoVNetMain(**VoVNetBkbn.V19S_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def V19(num_cls=10, act=ACT.RELU):
        return VoVNetMain(**VoVNetBkbn.V19_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def V39(num_cls=10, act=ACT.RELU):
        return VoVNetMain(**VoVNetBkbn.V39_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def V57(num_cls=10, act=ACT.RELU):
        return VoVNetMain(**VoVNetBkbn.V57_PARA, act=act, num_cls=num_cls)

    @staticmethod
    def V99(num_cls=10, act=ACT.RELU):
        return VoVNetMain(**VoVNetBkbn.V99_PARA, act=act, num_cls=num_cls)


class VoVNet(OneStageClassifier):
    def __init__(self, backbone, device=None, pack=None, img_size=(512, 512)):
        num_cls = backbone.num_cls
        super(VoVNet, self).__init__(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    @staticmethod
    def V19SDW(num_cls=10, device=None, pack=None, img_size=(512, 512)):
        backbone = VoVNetMain.V19SDW(num_cls=num_cls, act=ACT.RELU)
        return VoVNet(backbone=backbone, img_size=img_size, device=device, pack=pack)

    @staticmethod
    def V19DW(num_cls=10, device=None, pack=None, img_size=(512, 512)):
        backbone = VoVNetMain.V19DW(num_cls=num_cls, act=ACT.RELU)
        return VoVNet(backbone=backbone, img_size=img_size, device=device, pack=pack)

    @staticmethod
    def V19S(num_cls=10, device=None, pack=None, img_size=(512, 512)):
        backbone = VoVNetMain.V19S(num_cls=num_cls, act=ACT.RELU)
        return VoVNet(backbone=backbone, img_size=img_size, device=device, pack=pack)

    @staticmethod
    def V19(num_cls=10, device=None, pack=None, img_size=(512, 512)):
        backbone = VoVNetMain.V19(num_cls=num_cls, act=ACT.RELU)
        return VoVNet(backbone=backbone, img_size=img_size, device=device, pack=pack)

    @staticmethod
    def V39(num_cls=10, device=None, pack=None, img_size=(512, 512)):
        backbone = VoVNetMain.V39(num_cls=num_cls, act=ACT.RELU)
        return VoVNet(backbone=backbone, img_size=img_size, device=device, pack=pack)

    @staticmethod
    def V57(num_cls=10, device=None, pack=None, img_size=(512, 512)):
        backbone = VoVNetMain.V57(num_cls=num_cls, act=ACT.RELU)
        return VoVNet(backbone=backbone, img_size=img_size, device=device, pack=pack)

    @staticmethod
    def V99(num_cls=10, device=None, pack=None, img_size=(512, 512)):
        backbone = VoVNetMain.V99(num_cls=num_cls, act=ACT.RELU)
        return VoVNet(backbone=backbone, img_size=img_size, device=device, pack=pack)


if __name__ == '__main__':
    model = VoVNet.V19(device=0)
    from deploy import *
    # x=torch.zeros(1,3,224,224)
    # y=model(x)
    # model2onnx(model, './v39cus.onnx', 224)
