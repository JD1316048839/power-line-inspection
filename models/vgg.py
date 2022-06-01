from .modules import *


class VGGBkbn(nn.Module):
    def __init__(self, repeat_nums=(1, 1, 2, 2, 2), act=ACT.RELU):
        super(VGGBkbn, self).__init__()
        self.stage1 = VGGBkbn.Conv3Repeat(in_channels=3, out_channels=64,
                                          repeat_num=repeat_nums[0], act=act, with_pool=False)
        self.stage2 = VGGBkbn.Conv3Repeat(in_channels=64, out_channels=128,
                                          repeat_num=repeat_nums[1], act=act, with_pool=True)
        self.stage3 = VGGBkbn.Conv3Repeat(in_channels=128, out_channels=256,
                                          repeat_num=repeat_nums[2], act=act, with_pool=True)
        self.stage4 = VGGBkbn.Conv3Repeat(in_channels=256, out_channels=512,
                                          repeat_num=repeat_nums[3], act=act, with_pool=True)
        self.stage5 = VGGBkbn.Conv3Repeat(in_channels=512, out_channels=512,
                                          repeat_num=repeat_nums[4], act=act, with_pool=True)

    @staticmethod
    def Conv3Repeat(in_channels, out_channels, repeat_num=1, with_pool=False, act=ACT.RELU):
        backbone = nn.Sequential()
        if with_pool:
            backbone.add_module('pool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False))
        last_channels = in_channels
        for i in range(repeat_num):
            backbone.add_module(str(i), Conv3(in_channels=last_channels, out_channels=out_channels, act=act))
            last_channels = out_channels
        return backbone

    def forward(self, imgs):
        feat1 = self.stage1(imgs)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        feat5 = self.stage5(feat4)
        return feat5

    A_PARA = dict(repeat_nums=(1, 1, 2, 2, 2))
    B_PARA = dict(repeat_nums=(2, 2, 2, 2, 2))
    D_PARA = dict(repeat_nums=(2, 2, 3, 3, 3))
    E_PARA = dict(repeat_nums=(2, 2, 4, 3, 4))

    @staticmethod
    def A(act=ACT.RELU):
        return VGGBkbn(**VGGBkbn.A_PARA, act=act)

    @staticmethod
    def B(act=ACT.RELU):
        return VGGBkbn(**VGGBkbn.B_PARA, act=act)

    @staticmethod
    def D(act=ACT.RELU):
        return VGGBkbn(**VGGBkbn.D_PARA, act=act)

    @staticmethod
    def E(act=ACT.RELU):
        return VGGBkbn(**VGGBkbn.E_PARA, act=act)


class VGGMain(VGGBkbn):
    def __init__(self, repeat_nums=(1, 1, 2, 2, 2), act=ACT.RELU, num_cls=20, head_channel=4096, img_size=(224, 224)):
        super(VGGMain, self).__init__(repeat_nums=repeat_nums, act=act)
        self.num_cls = num_cls
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.head = nn.Sequential(
            nn.Linear(7 * 7 * 512, head_channel),
            ACT.build(act),
            nn.Dropout(),
            nn.Linear(head_channel, head_channel),
            ACT.build(act),
            nn.Dropout(),
            nn.Linear(head_channel, num_cls),
        )
        self._img_size = img_size

    @property
    def img_size(self):
        return self._img_size

    def forward(self, imgs):
        feat1 = self.stage1(imgs)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        feat4 = self.stage4(feat3)
        feat5 = self.stage5(feat4)
        feat = self.pool(feat5)
        feat = feat.reshape(feat.size(0), -1)
        feat = self.head(feat)
        return feat

    @staticmethod
    def A(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return VGGMain(**VGGBkbn.A_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=4096)

    @staticmethod
    def B(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return VGGMain(**VGGBkbn.B_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=4096)

    @staticmethod
    def D(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return VGGMain(**VGGBkbn.D_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=4096)

    @staticmethod
    def E(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return VGGMain(**VGGBkbn.E_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=4096)

    @staticmethod
    def AC(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return VGGMain(**VGGBkbn.A_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=512)

    @staticmethod
    def BC(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return VGGMain(**VGGBkbn.B_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=512)

    @staticmethod
    def DC(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return VGGMain(**VGGBkbn.D_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=512)

    @staticmethod
    def EC(act=ACT.RELU, num_cls=20, img_size=(224, 224)):
        return VGGMain(**VGGBkbn.E_PARA, act=act, num_cls=num_cls, img_size=img_size, head_channel=512)


class VGG(OneStageClassifier):
    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        img_size = backbone.img_size
        num_cls = backbone.num_cls
        super(VGG, self).__init__(backbone=backbone, device=device, pack=pack, img_size=img_size, num_cls=num_cls)

    def freeze_para(self):
        for name, para in self.named_parameters():
            if 'stage' in name:
                print('Freeze ' + name)
                para.requires_grad = False
            else:
                print('Activate ' + name)
                para.requires_grad = True
        return True

    @staticmethod
    def A(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = VGGMain.A(act=ACT.RELU, num_cls=num_cls, img_size=img_size)
        return VGG(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def B(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = VGGMain.B(act=ACT.RELU, num_cls=num_cls, img_size=img_size)
        return VGG(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def D(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = VGGMain.D(act=ACT.RELU, num_cls=num_cls, img_size=img_size)
        return VGG(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def E(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = VGGMain.E(act=ACT.RELU, num_cls=num_cls, img_size=img_size)
        return VGG(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def AC(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = VGGMain.AC(act=ACT.RELU, num_cls=num_cls, img_size=img_size)
        return VGG(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def BC(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = VGGMain.BC(act=ACT.RELU, num_cls=num_cls, img_size=img_size)
        return VGG(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def DC(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = VGGMain.DC(act=ACT.RELU, num_cls=num_cls, img_size=img_size)
        return VGG(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def EC(device=None, pack=None, num_cls=20, img_size=(224, 224)):
        backbone = VGGMain.EC(act=ACT.RELU, num_cls=num_cls, img_size=img_size)
        return VGG(backbone=backbone, device=device, pack=pack)


if __name__ == '__main__':
    model = VGGMain.A(img_size=(512, 512))
    x = torch.rand(1, 3, 512, 512)
    y = model(x)
