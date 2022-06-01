import imgaug as ia
import imgaug.augmenters as iaa
from .cus import *
from .pkgs import *

PAD_CVAL = 127


class AugSeq(list):

    def __init__(self, *item):
        super().__init__(item)

    def __call__(self, imgs, labels=None):
        for seq in self:
            if isinstance(seq, AugSeq):
                imgs, labels = seq.__call__(imgs, labels)
            elif isinstance(seq, iaa.Sequential):
                imgs, labels = iaa_transform(imgs, labels, seq)
            elif isinstance(seq, CusTransform):
                imgs, labels = seq(imgs, labels)
            elif isinstance(seq, torch.nn.Sequential) or isinstance(seq, transforms.Compose):
                imgs, labels = tor_transform(imgs, labels, seq)
            else:
                raise Exception('aug err')
        return imgs, labels

    @staticmethod
    def DET_NORM(img_size=(416, 352)):
        W, H = img_size
        seq = AugSeq(
            iaa.Sequential([
                iaa.PadToAspectRatio(aspect_ratio=W / H, position='center-center'),
                iaa.Resize({"width": W, "height": H})
            ]),
            CusToTensor(),
        )
        return seq

    @staticmethod
    def CLS_NORM(img_size=(32, 32)):
        W, H = img_size
        seq = AugSeq(
            iaa.Sequential([
                iaa.PadToAspectRatio(aspect_ratio=W / H, position='center-center', pad_cval=PAD_CVAL),
                iaa.Resize({"width": W, "height": H})
            ]),
            CusToTensor(),
        )
        return seq

    @staticmethod
    def DET_AUG_V3(img_size=(416, 352)):
        W, H = img_size
        seq = AugSeq(
            iaa.Sequential([
                iaa.PadToAspectRatio(aspect_ratio=W / H),
                iaa.Affine(rotate=(-5, 5), translate_percent=(-0.2, 0.2), scale=(0.8, 1.5), shear=(-5, 5),
                           mode='constant', cval=0),
                iaa.Resize({"width": W, "height": H}),
                iaa.Sharpen((0.0, 0.1)),
                iaa.AddToBrightness((-60, 40)),
                iaa.AddToHue((-10, 10)),
                iaa.Fliplr(0.5),
            ]),
            CusToTensor(),
        )
        return seq

    @staticmethod
    def CLS_AUG_V3(img_size=(416, 352)):
        W, H = img_size
        seq = AugSeq(
            iaa.Sequential([
                iaa.PadToAspectRatio(aspect_ratio=W / H, pad_cval=PAD_CVAL),
                iaa.Affine(rotate=(-90, 90), translate_percent=(-0.1, 0.1), scale=(0.6, 1.2), shear=(-5, 5),
                           mode='constant', cval=PAD_CVAL),
                iaa.Resize({"width": W, "height": H}),
                # iaa.GaussianBlur(sigma=(0, 1)),
                iaa.Sharpen((0.0, 0.2)),
                iaa.AddToBrightness((-60, 60)),
                iaa.AddToHue((-20, 20)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ]),
            CusToTensor(),
        )
        return seq

    @staticmethod
    def DET_AUG_V4(img_size=(416, 352)):
        W, H = img_size
        seq = AugSeq(
            iaa.Sequential([
                iaa.Affine(rotate=(-5, 5), translate_percent=(-0.1, 0.1), scale={"x": (0.8, 1.5), "y": (0.8, 1.5)},
                           shear=(-5, 5), mode='constant', cval=0),
                iaa.Fliplr(0.5),
            ]),
            Mosaic(repeat=0.2, img_size=img_size, add_type=ADD_TYPE.COVER),
            CutMix(repeat=0.1, num_patch=0.5, add_type=ADD_TYPE.COVER),
            iaa.Sequential([
                iaa.PadToAspectRatio(aspect_ratio=W / H),
                iaa.Resize({"width": W, "height": H}),
                iaa.Sharpen((0.0, 0.1)),
                iaa.AddToBrightness((-60, 40)),
                iaa.AddToHue((-10, 10)),
                iaa.Sometimes(p=0.7, then_list=iaa.CoarseDropout(0.1, size_percent=0.02)),
            ]),
            CusToTensor(),
        )
        return seq

