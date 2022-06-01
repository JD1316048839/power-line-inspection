from .modules import *


class InvertedBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, inner_channels, kernel_size, stride, act=ACT.RELU6):
        super(InvertedBottleNeck, self).__init__()
        self.stride = stride
        self.conv1 = Conv1(in_channels=in_channels, out_channels=inner_channels, act=act)
        self.conv2 = ConvAPDW(channels=inner_channels, stride=stride, kernel_size=kernel_size, act=act)
        self.conv3 = Conv1(in_channels=inner_channels, out_channels=out_channels, act=None)
        self.stride = stride
        if stride == 1:
            if in_channels != out_channels:
                self.shortcut = ConvAP(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=1, stride=stride, bn=True, act=None)
            else:
                self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class SeModule(nn.Module):
    def __init__(self, channels, ratio=0.25):
        super(SeModule, self).__init__()
        inner_channels = int(ratio * channels)
        self.se_pth = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv1(in_channels=channels, out_channels=inner_channels, bn=False, act=ACT.RELU),
            Conv1(in_channels=inner_channels, out_channels=channels, bn=False, act=ACT.SIG)
        )

    def forward(self, x):
        return x * self.se_pth(x)


class InvertedBottleNeckSE(InvertedBottleNeck):
    def __init__(self, in_channels, inner_channels, out_channels, kernel_size, stride, with_se=True, act=ACT.RELU6):
        super(InvertedBottleNeckSE, self).__init__(
            in_channels=in_channels, out_channels=out_channels, inner_channels=inner_channels,
            kernel_size=kernel_size, stride=stride, act=act)
        self.se = SeModule(channels=out_channels, ratio=0.25) if with_se else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.se(out) if self.se is not None else out
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV1Bkbn(nn.Module):
    def __init__(self, act=ACT.RELU, repeat_nums=(1, 1, 2, 2, 2)):
        super(MobileNetV1Bkbn, self).__init__()
        self.pre = ConvAP(in_channels=3, out_channels=32, kernel_size=3, stride=2, act=act)
        self.stage1 = MobileNetV1Bkbn.ConvDWRepeat(in_channels=32, out_channels=64, stride=1,
                                                   repeat_num=repeat_nums[0], act=act)
        self.stage2 = MobileNetV1Bkbn.ConvDWRepeat(in_channels=64, out_channels=128, stride=2,
                                                   repeat_num=repeat_nums[1], act=act)
        self.stage3 = MobileNetV1Bkbn.ConvDWRepeat(in_channels=128, out_channels=256, stride=2,
                                                   repeat_num=repeat_nums[2], act=act)
        self.stage4 = MobileNetV1Bkbn.ConvDWRepeat(in_channels=256, out_channels=512, stride=2,
                                                   repeat_num=repeat_nums[3], act=act)
        self.stage5 = MobileNetV1Bkbn.ConvDWRepeat(in_channels=512, out_channels=1024, stride=2,
                                                   repeat_num=repeat_nums[4], act=act)

    @staticmethod
    def ConvDWRepeat(in_channels, out_channels, stride, repeat_num=1, act=ACT.RELU):
        backbone = []
        last_channels = in_channels
        for i in range(repeat_num):
            backbone.append(ConvAPDW(channels=last_channels, kernel_size=3, stride=stride, act=act))
            backbone.append(Conv1(in_channels=last_channels, out_channels=out_channels, act=act))
            last_channels = out_channels
            stride = 1
        return nn.Sequential(*backbone)

    def forward(self, imgs):
        feat0 = self.pre(imgs)
        feat1 = self.stage1(feat0)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        feat5 = self.stage5(feat4)
        return feat5


class MobileNetV1Main(MobileNetV1Bkbn):
    def __init__(self, num_cls=10, act=ACT.RELU):
        super(MobileNetV1Main, self).__init__(act=act)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=1024, out_features=num_cls)

    def forward(self, imgs):
        feat = super(MobileNetV1Main, self).forward(imgs)
        feat = self.pool(feat)
        feat = feat.squeeze(dim=-1).squeeze(dim=-1)
        cls = self.linear(feat)
        return cls


class MobileNetV2Bkbn(nn.Module):
    def __init__(self, act=ACT.RELU6, repeat_nums=(1, 2, 3, 4, 3, 3, 1)):
        super(MobileNetV2Bkbn, self).__init__()
        self.pre = ConvAP(in_channels=3, out_channels=32, kernel_size=3, stride=2, act=act)
        self.stage1 = MobileNetV2Bkbn.BottleNeckRepeat(in_channels=32, out_channels=16, stride=1, expend=1,
                                                       repeat_num=repeat_nums[0], act=act)
        self.stage2 = MobileNetV2Bkbn.BottleNeckRepeat(in_channels=16, out_channels=24, stride=2, expend=6,
                                                       repeat_num=repeat_nums[1], act=act)
        self.stage3 = MobileNetV2Bkbn.BottleNeckRepeat(in_channels=24, out_channels=32, stride=2, expend=6,
                                                       repeat_num=repeat_nums[2], act=act)
        self.stage4 = MobileNetV2Bkbn.BottleNeckRepeat(in_channels=32, out_channels=64, stride=1, expend=6,
                                                       repeat_num=repeat_nums[3], act=act)
        self.stage5 = MobileNetV2Bkbn.BottleNeckRepeat(in_channels=64, out_channels=96, stride=2, expend=6,
                                                       repeat_num=repeat_nums[4], act=act)
        self.stage6 = MobileNetV2Bkbn.BottleNeckRepeat(in_channels=96, out_channels=160, stride=2, expend=6,
                                                       repeat_num=repeat_nums[5], act=act)
        self.stage7 = MobileNetV2Bkbn.BottleNeckRepeat(in_channels=160, out_channels=320, stride=1, expend=6,
                                                       repeat_num=repeat_nums[6], act=act)
        self.stage8 = Conv1(in_channels=320, out_channels=1280, act=act)

    @staticmethod
    def BottleNeckRepeat(in_channels, out_channels, stride, expend=6, repeat_num=1, act=ACT.RELU):
        backbone = []
        last_channels = in_channels
        for i in range(repeat_num):
            backbone.append(InvertedBottleNeck(in_channels=last_channels, out_channels=out_channels,
                                               inner_channels=last_channels * expend, kernel_size=3,
                                               stride=stride, act=act))
            last_channels = out_channels
            stride = 1
        return nn.Sequential(*backbone)

    def forward(self, imgs):
        feat0 = self.pre(imgs)
        feat1 = self.stage1(feat0)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        feat5 = self.stage5(feat4)
        feat6 = self.stage6(feat5)
        feat7 = self.stage7(feat6)
        feat8 = self.stage8(feat7)
        return feat8


class MobileNetV2Main(MobileNetV2Bkbn):
    def __init__(self, num_cls=10, act=ACT.RELU):
        super(MobileNetV2Main, self).__init__(act=act)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(in_features=1280, out_features=num_cls)

    def forward(self, imgs):
        feat = super(MobileNetV2Main, self).forward(imgs)
        feat = self.pool(feat)
        feat = feat.squeeze(dim=-1).squeeze(dim=-1)
        cls = self.linear(feat)
        return cls


class MobileNetV3SmallBkbn(nn.Module):
    def __init__(self, act1=ACT.RELU, act2=ACT.HSILU):
        super(MobileNetV3SmallBkbn, self).__init__()
        self.pre = ConvAP(in_channels=3, out_channels=16, kernel_size=3, stride=2, act=act1)
        self.stage1 = InvertedBottleNeckSE(16, 16, 16, kernel_size=3, stride=2, with_se=True, act=act1)
        self.stage2 = nn.Sequential(
            InvertedBottleNeckSE(16, 72, 24, kernel_size=3, stride=2, with_se=False, act=act1),
            InvertedBottleNeckSE(24, 88, 24, kernel_size=3, stride=1, with_se=False, act=act1)
        )
        self.stage3 = nn.Sequential(
            InvertedBottleNeckSE(24, 96, 40, kernel_size=5, stride=2, with_se=True, act=act2),
            InvertedBottleNeckSE(40, 240, 40, kernel_size=5, stride=1, with_se=True, act=act2),
            InvertedBottleNeckSE(40, 240, 40, kernel_size=5, stride=1, with_se=True, act=act2),
            InvertedBottleNeckSE(40, 120, 48, kernel_size=5, stride=1, with_se=True, act=act2),
            InvertedBottleNeckSE(48, 144, 48, kernel_size=5, stride=1, with_se=True, act=act2)
        )
        self.stage4 = nn.Sequential(
            InvertedBottleNeckSE(48, 288, 96, kernel_size=5, stride=2, with_se=True, act=act2),
            InvertedBottleNeckSE(96, 576, 96, kernel_size=5, stride=1, with_se=True, act=act2),
            InvertedBottleNeckSE(96, 576, 96, kernel_size=5, stride=1, with_se=True, act=act2),
            Conv1(in_channels=96, out_channels=576, bn=True, act=act2),
        )

    def forward(self, imgs):
        feat0 = self.pre(imgs)
        feat1 = self.stage1(feat0)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        return feat4


class MobileNetV3SmallMain(MobileNetV3SmallBkbn):
    def __init__(self, num_cls=10, act1=ACT.RELU, act2=ACT.HSILU):
        super(MobileNetV3SmallMain, self).__init__(act1=act1, act2=act2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024),
            ACT.build(act_name=act2),
            nn.Linear(in_features=1024, out_features=num_cls)
        )
        self.num_cls = num_cls

    def forward(self, imgs):
        feat = super(MobileNetV3SmallMain, self).forward(imgs)
        feat = self.pool(feat)
        feat = feat.squeeze(dim=-1).squeeze(dim=-1)
        cls = self.head(feat)
        return cls


class MobileNetV3LargeBkbn(nn.Module):
    def __init__(self, act1=ACT.RELU, act2=ACT.HSILU):
        super(MobileNetV3LargeBkbn, self).__init__()
        self.pre = nn.Sequential(
            ConvAP(in_channels=3, out_channels=16, kernel_size=3, stride=2, act=act1),
            InvertedBottleNeckSE(16, 16, 16, kernel_size=3, stride=1, with_se=False, act=act1),
        )
        self.stage1 = nn.Sequential(
            InvertedBottleNeckSE(16, 64, 24, kernel_size=3, stride=2, with_se=False, act=act1),
            InvertedBottleNeckSE(24, 72, 24, kernel_size=3, stride=1, with_se=False, act=act1)
        )
        self.stage2 = nn.Sequential(
            InvertedBottleNeckSE(24, 72, 40, kernel_size=5, stride=2, with_se=True, act=act1),
            InvertedBottleNeckSE(40, 120, 40, kernel_size=5, stride=1, with_se=True, act=act1),
            InvertedBottleNeckSE(40, 120, 40, kernel_size=5, stride=1, with_se=True, act=act1)
        )
        self.stage3 = nn.Sequential(
            InvertedBottleNeckSE(40, 240, 80, kernel_size=3, stride=2, with_se=False, act=act2),
            InvertedBottleNeckSE(80, 200, 80, kernel_size=3, stride=1, with_se=False, act=act2),
            InvertedBottleNeckSE(80, 184, 80, kernel_size=3, stride=1, with_se=False, act=act2),
            InvertedBottleNeckSE(80, 184, 80, kernel_size=3, stride=1, with_se=False, act=act2),
            InvertedBottleNeckSE(80, 480, 112, kernel_size=3, stride=1, with_se=True, act=act2),
            InvertedBottleNeckSE(112, 672, 112, kernel_size=3, stride=1, with_se=True, act=act2),
        )
        self.stage4 = nn.Sequential(
            InvertedBottleNeckSE(112, 672, 160, kernel_size=5, stride=2, with_se=True, act=act2),
            InvertedBottleNeckSE(160, 672, 160, kernel_size=5, stride=1, with_se=True, act=act2),
            InvertedBottleNeckSE(160, 960, 160, kernel_size=5, stride=1, with_se=True, act=act2),
            Conv1(in_channels=160, out_channels=960, bn=True, act=act2)
        )

    def forward(self, imgs):
        feat0 = self.pre(imgs)
        feat1 = self.stage1(feat0)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        return feat4


class MobileNetV3LargeMain(MobileNetV3LargeBkbn):
    def __init__(self, num_cls=10, act1=ACT.RELU, act2=ACT.HSILU):
        super(MobileNetV3LargeMain, self).__init__(act1=act1, act2=act2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280),
            ACT.build(act_name=act2),
            nn.Linear(in_features=1280, out_features=num_cls)
        )
        self.num_cls = num_cls

    def forward(self, imgs):
        feat = super(MobileNetV3LargeMain, self).forward(imgs)
        feat = self.pool(feat)
        feat = feat.squeeze(dim=-1).squeeze(dim=-1)
        cls = self.head(feat)
        return cls


class MobileNet(OneStageClassifier):
    def __init__(self, backbone, device=None, pack=None, img_size=(512, 512)):
        num_cls = backbone.num_cls
        super(MobileNet, self).__init__(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)
        self.img_size = img_size

    @staticmethod
    def V1(device=None, pack=None, num_cls=10, img_size=(512, 512)):
        backbone = MobileNetV1Main(act=ACT.RELU, num_cls=num_cls)
        return MobileNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def V2(device=None, pack=None, num_cls=10, img_size=(512, 512)):
        backbone = MobileNetV2Main(act=ACT.RELU6, num_cls=num_cls)
        return MobileNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def V3Small(device=None, pack=None, num_cls=10, img_size=(512, 512)):
        backbone = MobileNetV3SmallMain(act1=ACT.RELU, act2=ACT.HSILU, num_cls=num_cls)
        return MobileNet(backbone=backbone, device=device, pack=pack, img_size=img_size)

    @staticmethod
    def V3Large(device=None, pack=None, num_cls=10, img_size=(512, 512)):
        backbone = MobileNetV3LargeMain(act1=ACT.RELU, act2=ACT.HSILU, num_cls=num_cls)
        return MobileNet(backbone=backbone, device=device, pack=pack, img_size=img_size)


if __name__ == '__main__':
    model = MobileNetV3LargeBkbn()
    # x = torch.zeros(1, 3, 224, 224)
    # y = model(x)
