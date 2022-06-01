import numpy as np
import PIL
from PIL import Image, ImageDraw
import copy
import torch
import math


class BOX_FMT:
    XYXY = 'xyxy'
    XYWH = 'xywh'
    XYWHA = 'xywha'
    XYWHAB = 'xywhab'
    XLYL = 'xlyl'
    AUTO = 'auto'


BOX_DIM = {
    BOX_FMT.XYXY: [4],
    BOX_FMT.XYWH: [4],
    BOX_FMT.XYWHA: [5],
    BOX_FMT.XYWHAB: [6],
    BOX_FMT.XLYL: [4, 2],
}


class BoxList(list):
    def __init__(self, **meta):
        super().__init__()
        self.meta = meta

    def empty(self):
        return BoxList(**self.meta)


# <editor-fold desc='图像格式转换'>
def pil2ten(img: PIL.Image.Image) -> torch.Tensor:
    img = np.array(img)
    img = img / 255
    img = np.transpose(img, (2, 0, 1))  # HWC转CHW
    img = torch.from_numpy(img).float()[None, :]
    return img


def ten2pil(img: torch.Tensor) -> PIL.Image.Image:
    img = img.detach().cpu().numpy()
    if len(img.shape) == 4 and img.shape[0] == 1:
        img = img.squeeze(axis=0)
    if len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW转为HWC
    img = img * 255
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img


def arr2pil(img: np.ndarray) -> PIL.Image.Image:
    img = Image.fromarray(img.astype('uint8')).convert('RGB')
    return img


def pil2arr(img: PIL.Image.Image) -> np.ndarray:
    img = np.array(img)
    return img


def arrs2pils(imgs: list) -> list:
    for i in range(len(imgs)):
        imgs[i] = Image.fromarray(imgs[i].astype('uint8')).convert('RGB')
    return imgs


def pils2arrs(imgs: list) -> list:
    for i in range(len(imgs)):
        imgs[i] = np.array(imgs[i])
    return imgs


def pil_mask_xlylN(img: PIL.Image.Image, xlylN: np.ndarray) -> PIL.Image.Image:
    draw = ImageDraw.Draw(img)
    draw.polygon(list(xlylN.reshape(-1)), outline=0, fill=0)
    return img


def xlylN2pil_mask(xlylN: np.ndarray, w: int, h: int) -> np.ndarray:
    mask = Image.new('L', (w, h), 0)
    PIL.ImageDraw.Draw(mask).polygon(list(xlylN.reshape(-1)), outline=255, fill=255)
    return mask


def xywhaN2pil_mask(xywhaN: np.ndarray, w: int, h: int) -> np.ndarray:
    xlylN = xywhaN2xlylN(xywhaN)
    mask = xlylN2pil_mask(xlylN=xlylN, w=w, h=h)
    return mask


def xyxyN2pil_mask(xyxyN: np.ndarray, w: int, h: int) -> np.ndarray:
    xlylN = xyxyN2xlylN(xyxyN)
    mask = xlylN2pil_mask(xlylN=xlylN, w=w, h=h)
    return mask


def pil_imask_xlylN(img: PIL.Image.Image, xlylN: np.ndarray) -> PIL.Image.Image:
    w, h = img.size
    mask = Image.new('L', (w, h), 0)
    PIL.ImageDraw.Draw(mask).polygon(list(xlylN.reshape(-1)), outline=1, fill=1)
    mask = np.array(mask)
    img_new = np.array(img)
    img_new *= mask[..., None]
    img_new = Image.fromarray(img_new.astype('uint8')).convert('RGB')
    return img_new


def pil_imask_xywhaN(img: PIL.Image.Image, xywhaN: np.ndarray) -> PIL.Image.Image:
    xlylN = xywhaN2xlylN(xywhaN)
    img = pil_imask_xlylN(img=img, xlylN=xlylN)
    return img


def pil_mask_xywhaN(img: PIL.Image.Image, xywhaN: np.ndarray) -> PIL.Image.Image:
    xlylN = xywhaN2xlylN(xywhaN)
    pil_mask_xlylN(img=img, xlylN=xlylN)
    return img


def pil_crop_xlylN(img: PIL.Image.Image, xlylN: np.ndarray) -> PIL.Image.Image:
    xyxy = xlylN2xyxyN(xlylN)
    patch = img.crop(xyxy)
    pw, ph = patch.size
    mask = Image.new('L', (pw, ph), 0)
    xlyl_ptch = xlylN - xyxy[:2]
    PIL.ImageDraw.Draw(mask).polygon(list(xlyl_ptch.reshape(-1)), outline=255, fill=255)
    r, g, b = patch.split()
    patch = Image.merge('RGBA', (r, g, b, mask))
    return patch


def pil_flip(patch: PIL.Image.Image, alpha: float, flip: bool = False, vflip: bool = False) -> PIL.Image.Image:
    if not flip and not vflip:
        return patch
    elif flip and vflip:
        pc = np.array(patch.size) / 2
        data = np.array([-1, 0, pc[0] * 2, 0, -1, pc[1] * 2])
        patch = patch.transform(size=patch.size, method=Image.AFFINE, data=data, resample=Image.BILINEAR)
        return patch
    cos, sin = np.cos(alpha), np.sin(alpha)
    cos2, sin2, cossin = cos ** 2, sin ** 2, cos * sin
    pc = np.array(patch.size) / 2
    mat = np.array([[cos2 - sin2, 2 * cossin], [2 * cossin, sin2 - cos2]])
    mat = mat if flip else -mat
    bias = pc - pc @ mat
    data = np.concatenate([mat[0], [bias[0]], mat[1], [bias[1]]])
    patch = patch.transform(size=patch.size, method=Image.AFFINE, data=data, resample=Image.BILINEAR)
    return patch


def pil_crop_xywhaN_with_flip(img: PIL.Image.Image, xywhaN: np.ndarray, wflip: bool = False,
                              hflip: bool = False) -> PIL.Image.Image:
    xlyl = xywhaN2xlylN(xywhaN)
    patch = pil_crop_xlylN(img, xlyl)
    patch = pil_flip(patch, alpha=xywhaN[4], flip=wflip, vflip=hflip)
    return patch


def pil_flip_with_xywhaN(img: PIL.Image.Image, xywhaN: np.ndarray, wflip: bool = False,
                         hflip: bool = False) -> PIL.Image.Image:
    if not wflip and not hflip:
        return img
    xlyl = xywhaN2xlylN(xywhaN)
    patch = pil_crop_xlylN(img, xlyl)
    patch = pil_flip(patch, alpha=xywhaN[4], flip=wflip, vflip=hflip)
    pwh = np.array(patch.size)
    pmin = xywhaN[:2] - pwh / 2
    r, g, b, a = patch.split()
    img.paste(patch, box=(int(pmin[0]), int(pmin[1])), mask=a)
    return img


# </editor-fold>


# <editor-fold desc='numpy格式转换'>
CORNERSN = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])


def clsN2oclsN(clsN: np.ndarray, num_cls: int) -> np.ndarray:
    clso = np.zeros(shape=num_cls)
    clso[clsN] = 1
    return clso


def clsesN2oclsesN(clsesN: np.ndarray, num_cls: int) -> np.ndarray:
    num = len(clsesN)
    clsos = np.zeros(shape=(num, num_cls))
    clsos[range(num), clsesN] = 1
    return clsos


def xywhN2xyxyN(xywhN: np.ndarray) -> np.ndarray:
    xcyc, wh_2 = xywhN[:2], xywhN[2:4] / 2
    return np.concatenate([xcyc - wh_2, xcyc + wh_2], axis=0)


def xywhsN2xyxysN(xywhsN) -> np.ndarray:
    xcyc, wh_2 = xywhsN[..., :2], xywhsN[..., 2:4] / 2
    return np.concatenate([xcyc - wh_2, xcyc + wh_2], axis=-1)


def xyxyN2xywhN(xyxyN: np.ndarray) -> np.ndarray:
    x1y1, x2y2 = xyxyN[:2], xyxyN[2:4]
    return np.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1], axis=0)


def xyxysN2xywhsN(xyxysN: np.ndarray) -> np.ndarray:
    x1y1, x2y2 = xyxysN[..., :2], xyxysN[..., 2:4]
    return np.concatenate([(x1y1 + x2y2) / 2, x2y2 - x1y1], axis=-1)


def xyxyN2xlylN(xyxyN: np.ndarray) -> np.ndarray:
    xlyl = np.stack([xyxyN[[0, 2, 2, 0]], xyxyN[[1, 1, 3, 3]]], axis=1)
    return xlyl


def xyxysN2xlylsN(xyxysN: np.ndarray) -> np.ndarray:
    xlyls = np.stack([xyxysN[..., [0, 2, 2, 0]], xyxysN[..., [1, 1, 3, 3]]], axis=-1)
    return xlyls


def xywhN2xlylN(xywhN: np.ndarray) -> np.ndarray:
    return xywhN[:2] + CORNERSN * xywhN[2:3] / 2


def xywhsN2xlylsN(xywhN: np.ndarray) -> np.ndarray:
    return xywhN[..., None, :2] + CORNERSN * xywhN[..., None, 2:3] / 2


def xywhaN2x1y1whaN(xywhaN: np.ndarray) -> np.ndarray:
    cos, sin = math.cos(xywhaN[4]), math.sin(xywhaN[4])
    mat = np.array([[cos, -sin], [sin, cos]])
    return np.concatenate([xywhaN[:2] - xywhaN[2:4] @ mat / 2, xywhaN[2:5]], axis=0)


def xywhasN2x1y1whasN(xywhasN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(xywhasN[..., 4]), np.sin(xywhasN[..., 4])
    wh_2 = xywhasN[..., 2:4, None] / 2
    mat = np.stack([np.stack([cos, -sin], axis=-1), np.stack([sin, cos], axis=-1)], axis=-2)
    return np.concatenate([xywhasN[..., :2] - wh_2 @ mat, xywhasN[..., 2:5]], axis=-1)


def xywhaN2xlylN(xywhaN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(xywhaN[4]), np.sin(xywhaN[4])
    mat = np.array([[cos, sin], [-sin, cos]])
    xlyl = xywhaN[:2] + (CORNERSN * xywhaN[2:4] / 2) @ mat
    return xlyl


def xywhasN2xlylsN(xywhasN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(xywhasN[..., 4]), np.sin(xywhasN[..., 4])
    mat = np.stack([np.stack([cos, sin], axis=-1), np.stack([-sin, cos], axis=-1)], axis=-2)
    xlyls = xywhasN[..., None, :2] + (CORNERSN * xywhasN[..., None, 2:4] / 2) @ mat
    return xlyls


def xywhasN2xyxysN(xywhasN: np.ndarray) -> np.ndarray:
    xlylsN = xywhasN2xlylsN(xywhasN)
    x1y1 = np.min(xlylsN, axis=-2)
    x2y2 = np.max(xlylsN, axis=-2)
    xyxysN = np.concatenate([x1y1, x2y2], axis=-1)
    return xyxysN


def xywhabN2xlylN(xywhabN: np.ndarray) -> np.ndarray:
    cosa, sina = np.cos(xywhabN[4]), np.sin(xywhabN[4])
    cosb, sinb = np.cos(xywhabN[5]), np.sin(xywhabN[5])
    mat = np.array([[cosa, sina], [cosb, sinb]])
    xlyl = xywhabN[:2] + (CORNERSN * xywhabN[2:4] / 2) @ mat
    return xlyl


def xywhabsN2xlylsN(xywhabsN: np.ndarray) -> np.ndarray:
    cosa, sina = np.cos(xywhabsN[..., 4]), np.sin(xywhabsN[..., 4])
    cosb, sinb = np.cos(xywhabsN[..., 5]), np.sin(xywhabsN[..., 5])
    mat = np.stack([np.stack([cosa, sina], axis=-1), np.stack([cosb, sinb], axis=-1)], axis=-2)
    xlyls = xywhabsN[..., None, :2] + (CORNERSN * xywhabsN[..., None, 2:4] / 2) @ mat
    return xlyls


def xlylN2xyxyN(xlylN: np.ndarray) -> np.ndarray:
    x1y1 = np.min(xlylN, axis=0)
    x2y2 = np.max(xlylN, axis=0)
    return np.concatenate([x1y1, x2y2], axis=0)


def xywhaN2xyxyN(xywhaN: np.ndarray) -> np.ndarray:
    xlylN = xywhaN2xlylN(xywhaN)
    return xlylN2xyxyN(xlylN)


def xywhabN2xyxyN(xywhabN: np.ndarray) -> np.ndarray:
    xlylN = xywhabN2xlylN(xywhabN)
    return xlylN2xyxyN(xlylN)


def boxN2xyxyN(boxN: np.ndarray, box_fmt: str) -> np.ndarray:
    if box_fmt == BOX_FMT.XYXY:
        return boxN
    elif box_fmt == BOX_FMT.XYWH:
        return xywhN2xyxyN(boxN)
    elif box_fmt == BOX_FMT.XYWHA:
        return xywhaN2xyxyN(boxN)
    elif box_fmt == BOX_FMT.XYWHAB:
        return xywhabN2xyxyN(boxN)
    elif box_fmt == BOX_FMT.XLYL:
        return xlylN2xyxyN(boxN)
    else:
        raise Exception('err box fmt' + str(box_fmt))


def xywhaN2xywhN(xywhaN: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(xywhaN[4]), np.sin(xywhaN[4])
    wh = np.abs(cos * xywhaN[2:4]) + np.abs(sin * xywhaN[3:1:-1])
    return np.concatenate([xywhaN[:2], wh], axis=0)


def xywhabN2xywhN(xywhabN: np.ndarray) -> np.ndarray:
    cosa, sina = np.cos(xywhabN[4]), np.sin(xywhabN[4])
    cosb, sinb = np.cos(xywhabN[5]), np.sin(xywhabN[5])
    w = np.abs(cosa * xywhabN[2]) + np.abs(cosb * xywhabN[3])
    h = np.abs(sinb * xywhabN[3]) + np.abs(sina * xywhabN[2])
    return np.concatenate([xywhabN[:2], w[None], h[None]], axis=0)


def xlylN2xywhN(xlylN: np.ndarray) -> np.ndarray:
    xyxy = xlylN2xyxyN(xlylN)
    return xyxyN2xywhN(xyxy)


def boxN2xywhN(boxN: np.ndarray, box_fmt: str) -> np.ndarray:
    if box_fmt == BOX_FMT.XYXY:
        return xyxyN2xywhN(boxN)
    elif box_fmt == BOX_FMT.XYWH:
        return boxN
    elif box_fmt == BOX_FMT.XYWHA:
        return xywhaN2xywhN(boxN)
    elif box_fmt == BOX_FMT.XYWHAB:
        return xywhabN2xywhN(boxN)
    elif box_fmt == BOX_FMT.XLYL:
        return xlylN2xywhN(boxN)
    else:
        raise Exception('err box fmt' + str(box_fmt))


def xyxyN2xywhaN(xyxyN: np.ndarray) -> np.ndarray:
    xywh = xyxyN2xywhN(xyxyN)
    return np.concatenate([xywh, [0]], axis=0)


def xlylN2xywhaN(xlylN: np.ndarray) -> np.ndarray:
    vw = xlylN[0] - xlylN[1]
    vh = xlylN[1] - xlylN[2]
    xy = np.mean(xlylN, axis=0)
    w = np.sqrt(np.sum(vw ** 2))
    h = np.sqrt(np.sum(vh ** 2))
    a = np.arctan2(vw[1], vw[0]) % np.pi
    return np.concatenate([xy, [w, h, a]], axis=0)


def xlylN2xywhabN(xlylN: np.ndarray) -> np.ndarray:
    vw = xlylN[0] - xlylN[1]
    vh = xlylN[1] - xlylN[2]
    xy = np.mean(xlylN, axis=0)
    w = np.sqrt(np.sum(vw ** 2))
    h = np.sqrt(np.sum(vh ** 2))
    a = np.arctan2(vw[1], vw[0]) % np.pi
    b = np.arctan2(vh[0], vh[1]) % np.pi + np.pi / 2
    return np.concatenate([xy, [w, h, a, b]], axis=0)


def boxN2xywhaN(boxN: np.ndarray, box_fmt: str) -> np.ndarray:
    if box_fmt == BOX_FMT.XYXY:
        xywh = xyxyN2xywhN(boxN)
        return np.concatenate([xywh, [0]], axis=0)
    elif box_fmt == BOX_FMT.XYWH:
        return np.concatenate([boxN, [0]], axis=0)
    elif box_fmt == BOX_FMT.XYWHA:
        return boxN
    elif box_fmt == BOX_FMT.XYWHAB:
        return boxN[:5]
    elif box_fmt == BOX_FMT.XLYL:
        return xlylN2xywhaN(boxN)
    else:
        raise Exception('err box fmt' + str(box_fmt))


def boxN2xywhabN(boxN: np.ndarray, box_fmt: str) -> np.ndarray:
    if box_fmt == BOX_FMT.XYXY:
        xywh = xyxyN2xywhN(boxN)
        return np.concatenate([xywh, [0, np.pi / 2]], axis=0)
    elif box_fmt == BOX_FMT.XYWH:
        return np.concatenate([boxN, [0, np.pi / 2]], axis=0)
    elif box_fmt == BOX_FMT.XYWHA:
        return np.concatenate([boxN, [np.pi / 2]], axis=0)
    elif box_fmt == BOX_FMT.XYWHAB:
        return boxN
    elif box_fmt == BOX_FMT.XLYL:
        return xlylN2xywhabN(boxN)
    else:
        raise Exception('err box fmt' + str(box_fmt))


def boxN2xlylN(boxN: np.ndarray, box_fmt: str) -> np.ndarray:
    if box_fmt == BOX_FMT.XYXY:
        return xyxyN2xlylN(boxN)
    elif box_fmt == BOX_FMT.XYWH:
        return xywhN2xlylN(boxN)
    elif box_fmt == BOX_FMT.XYWHA:
        return xywhaN2xlylN(boxN)
    elif box_fmt == BOX_FMT.XYWHAB:
        return xywhabN2xlylN(boxN)
    elif box_fmt == BOX_FMT.XLYL:
        return boxN
    else:
        raise Exception('err box fmt' + str(box_fmt))


def xlylN2boxN(xlylN: np.ndarray, box_fmt: str) -> np.ndarray:
    if box_fmt == BOX_FMT.XYXY:
        return xlylN2xyxyN(xlylN)
    elif box_fmt == BOX_FMT.XYWH:
        return xlylN2xywhN(xlylN)
    elif box_fmt == BOX_FMT.XYWHA:
        return xlylN2xywhaN(xlylN)
    elif box_fmt == BOX_FMT.XYWHAB:
        return xlylN2xywhabN(xlylN)
    elif box_fmt == BOX_FMT.XLYL:
        return xlylN
    else:
        raise Exception('err box fmt' + str(box_fmt))


# if __name__ == '__main__':
#     from visual import *
#     img = Image.open('/home/user/JD/Datasets/RawC/unknown/003000_5.jpg')
#     pil_mask_xywhaN(img, xywhaN=np.array([100, 100, 50, 50, np.pi / 4]))
#     img.show()


# if __name__ == '__main__':
#     a = np.array([10, 10, 4, 6, np.pi / 2, np.pi / 4])
#     b = xywhabN2xywhN(a)

# if __name__ == '__main__':
#     a = np.array([[10, 10, 6, 2, 0, np.pi / 2], [10, 10, 6, 2, np.pi / 2, np.pi]])
#     b = xywhabsN2xlylsN(a)

def xlylN_clip(xlylN: np.ndarray, xy_min: list, xy_max: list, thres: int = -1) -> (np.ndarray, bool):
    xlylN[:, 0] = np.clip(xlylN[:, 0], a_min=xy_min[0], a_max=xy_max[0])
    xlylN[:, 1] = np.clip(xlylN[:, 1], a_min=xy_min[1], a_max=xy_max[1])
    return xlylN, thres > 0


def xyxyN_clip(xyxyN: np.ndarray, xy_min: list, xy_max: list, thres: int = -1) -> (np.ndarray, bool):
    xyxyN[0:4:2] = np.clip(xyxyN[0:4:2], a_min=xy_min[0], a_max=xy_max[0])
    xyxyN[1:4:2] = np.clip(xyxyN[1:4:2], a_min=xy_min[1], a_max=xy_max[1])
    return xyxyN, (thres > 0) and np.any(xyxyN[2:4] - xyxyN[:2] < thres)


def xywhN_clip(xywhN: np.ndarray, xy_min: list, xy_max: list, thres: int = -1) -> (np.ndarray, bool):
    xyxy = xywhN2xyxyN(xywhN)
    xyxy, cliped = xyxyN_clip(xyxy, xy_min=xy_min, xy_max=xy_max, thres=thres)
    xywhN = xyxyN2xywhN(xyxy)
    return xywhN, cliped


def xywhaN_clip(xywhaN: np.ndarray, xy_min: list, xy_max: list, thres: int = -1) -> (np.ndarray, bool):
    cos, sin = np.cos(xywhaN[4]), np.sin(xywhaN[4])
    pns = np.stack([xy_min, xy_max, [xy_min[0], xy_max[1]], [xy_max[0], xy_min[1]]], axis=0)
    pns_dt = (pns - xywhaN[:2])

    vecw = np.array([cos, sin])
    vecw[vecw == 0] = 1e-16
    # w_pos_cast = pns_dt @ vecw
    # w_max1 = np.max(w_pos_cast)
    # w_min1 = np.min(w_pos_cast)
    w_neg_cast = pns_dt[:2] / vecw
    w_min2 = np.max(np.min(w_neg_cast, axis=0))
    w_max2 = np.min(np.max(w_neg_cast, axis=0))
    ws = np.clip(np.array([-xywhaN[2], xywhaN[2]]) / 2, a_min=w_min2, a_max=w_max2)

    vech = np.array([-sin, cos])
    vech[vech == 0] = 1e-16
    # h_pos_cast = pns_dt @ vech
    # h_max1 = np.max(h_pos_cast)
    # h_min1 = np.min(h_pos_cast)
    h_neg_cast = pns_dt[:2] / vech
    h_min2 = np.max(np.min(h_neg_cast, axis=0))
    h_max2 = np.min(np.max(h_neg_cast, axis=0))
    hs = np.clip(np.array([-xywhaN[3], xywhaN[3]]) / 2, a_min=h_min2, a_max=h_max2)

    whs = np.stack([ws, hs], axis=0)
    wh = whs[:, 1] - whs[:, 0]
    mat = np.array([[cos, sin], [-sin, cos]])
    xy = xywhaN[:2] + np.average(whs, axis=-1) @ mat

    croped = (w_min2 > 0 or w_max2 < 0) and (h_min2 > 0 or h_max2 < 0)

    return np.concatenate([xy, wh, xywhaN[4:5]], axis=0), (thres > 0) and (np.any(wh < thres) or croped)


def xywhabN_clip(xywhabN: np.ndarray, xy_min: list, xy_max: list, thres: int = -1) -> (np.ndarray, bool):
    cosa, sina = np.cos(xywhabN[4]), np.sin(xywhabN[4])
    cosb, sinb = np.cos(xywhabN[5]), np.sin(xywhabN[5])
    border = np.stack([xy_min, xy_max], axis=1)
    xy_dt = border - xywhabN[:2, None]

    vecw = np.array([cosa, sina])
    vecw[vecw == 0] = 1e-16
    border_w = xy_dt / vecw[:, None]
    w_min = np.max(np.min(border_w, axis=-1))
    w_max = np.min(np.max(border_w, axis=-1))
    ws = np.clip(np.array([-xywhabN[2], xywhabN[2]]) / 2, a_min=w_min, a_max=w_max)

    vech = np.array([cosb, sinb])
    vech[vech == 0] = 1e-16
    border_h = xy_dt / vech[:, None]
    h_min = np.max(np.min(border_h, axis=-1))
    h_max = np.min(np.max(border_h, axis=-1))
    hs = np.clip(np.array([-xywhabN[3], xywhabN[3]]) / 2, a_min=h_min, a_max=h_max)

    whs = np.stack([ws, hs], axis=0)
    wh = whs[:, 1] - whs[:, 0]
    mat = np.array([[cosa, sina], [cosb, sinb]])
    xy = xywhabN[:2] + np.average(whs, axis=-1) @ mat

    return np.concatenate([xy, wh, xywhabN[4:6]], axis=0), (thres > 0) and np.any(wh < thres)


def boxN_clip(boxN: np.ndarray, box_fmt: str, xy_min: list, xy_max: list, thres: int = -1) -> (np.ndarray, bool):
    if box_fmt == BOX_FMT.XYXY:
        return xyxyN_clip(boxN, xy_min=xy_min, xy_max=xy_max, thres=thres)
    elif box_fmt == BOX_FMT.XYWH:
        return xywhN_clip(boxN, xy_min=xy_min, xy_max=xy_max, thres=thres)
    elif box_fmt == BOX_FMT.XYWHA:
        return xywhaN_clip(boxN, xy_min=xy_min, xy_max=xy_max, thres=thres)
    elif box_fmt == BOX_FMT.XYWHAB:
        return xywhabN_clip(boxN, xy_min=xy_min, xy_max=xy_max, thres=thres)
    elif box_fmt == BOX_FMT.XLYL:
        return xyxyN_clip(boxN, xy_min=xy_min, xy_max=xy_max, thres=thres)
    else:
        raise Exception('err box fmt' + str(box_fmt))


# if __name__ == '__main__':
#     a = np.array([10, 20, 8, 6, 0])
#     b = xywhaN_clip(a, [5, 5], [12, 22])

# </editor-fold>


# <editor-fold desc='torch格式转换'>
CORNERST = torch.Tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]])


def clsT2oclsT(clsT: torch.Tensor, num_cls: int) -> torch.Tensor:
    clso = torch.zeros(num_cls)
    clso[clsT] = 1
    return clso


def clsesT2oclsesT(clsesT: torch.Tensor, num_cls: int) -> torch.Tensor:
    num = len(clsesT)
    clsos = torch.zeros(num, num_cls)
    clsos[range(num), clsesT] = 1
    return clsos


def xyxyT2xywhT(xyxyT: torch.Tensor) -> torch.Tensor:
    x1y1, x2y2 = xyxyT[:2], xyxyT[2:4]
    return torch.cat([(x1y1 + x2y2) / 2, x2y2 - x1y1], dim=0)


def xyxysT2xywhsT(xyxysT: torch.Tensor) -> torch.Tensor:
    x1y1, x2y2 = xyxysT[..., :2], xyxysT[..., 2:4]
    return torch.cat([(x1y1 + x2y2) / 2, x2y2 - x1y1], dim=-1)


def xywhT2xyxyT(xywhT: torch.Tensor) -> torch.Tensor:
    xcyc, wh_2 = xywhT[:2], xywhT[2:4] / 2
    return torch.cat([xcyc - wh_2, xcyc + wh_2], dim=0)


def xywhsT2xyxysT(xywhsT: torch.Tensor) -> torch.Tensor:
    xcyc, wh_2 = xywhsT[..., :2], xywhsT[..., 2:4] / 2
    return torch.cat([xcyc - wh_2, xcyc + wh_2], dim=-1)


def xyxyT2xlylT(xyxyT: torch.Tensor) -> torch.Tensor:
    xlyl = torch.stack([xyxyT[[0, 2, 2, 0]], xyxyT[[1, 1, 3, 3]]], dim=1)
    return xlyl


def xyxysT2xlylsT(xyxysT: torch.Tensor) -> torch.Tensor:
    xlyls = torch.stack([xyxysT[..., [0, 2, 2, 0]], xyxysT[..., [1, 1, 3, 3]]], dim=-1)
    return xlyls


def xywhaT2xlylT(xywhaT: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(xywhaT[4]), torch.sin(xywhaT[4])
    mat = torch.Tensor([[cos, sin], [-sin, cos]])
    xlyl = xywhaT[:2] + (CORNERST * xywhaT[2:4] / 2) @ mat
    return xlyl


def xywhasT2xlylsT(xywhasT: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(xywhasT[..., 4]), torch.sin(xywhasT[..., 4])
    mat = torch.stack([torch.stack([cos, sin], dim=-1), torch.stack([-sin, cos], dim=-1)], dim=-2)
    xlyls = xywhasT[..., None, :2] + (CORNERST.to(xywhasT.device) * xywhasT[..., None, 2:4] / 2) @ mat
    return xlyls


def xywhabsT2xlylsT(xywhabsT: torch.Tensor) -> torch.Tensor:
    cosa, sina = torch.cos(xywhabsT[..., 4]), torch.sin(xywhabsT[..., 4])
    cosb, sinb = torch.cos(xywhabsT[..., 5]), torch.sin(xywhabsT[..., 5])
    mat = torch.stack([torch.stack([cosa, sina], dim=-1), torch.stack([cosb, sinb], dim=-1)], dim=-2)
    xlyls = xywhabsT[..., None, :2] + (CORNERST.to(xywhabsT.device) * xywhabsT[..., None, 2:4] / 2) @ mat
    return xlyls


def xlylsT2xyxysT(xlylsT: torch.Tensor) -> torch.Tensor:
    x1y1 = torch.min(xlylsT, dim=-2)[0]
    x2y2 = torch.max(xlylsT, dim=-2)[0]
    xyxys = torch.cat([x1y1, x2y2], dim=-1)
    return xyxys


def xywhasT2xyxysT(xywhasT: torch.Tensor) -> torch.Tensor:
    xlyls = xywhasT2xlylsT(xywhasT)
    xyxys = xlylsT2xyxysT(xlyls)
    return xyxys


# </editor-fold>

# <editor-fold desc='list格式转换'>


def xyxy2xywh(xyxy) -> list:
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def xywh2xyxy(xywh) -> list:
    xc, yc, w_2, h_2 = xywh[0], xywh[1], xywh[2], xywh[3]
    return [xc - w_2, yc - h_2, xc + w_2, yc + h_2]


def xyxy2xywhN(xyxy) -> np.ndarray:
    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])


def xywh2xyxyN(xywh) -> np.ndarray:
    xc, yc, w_2, h_2 = xywh[0], xywh[1], xywh[2], xywh[3]
    return np.array([xc - w_2, yc - h_2, xc + w_2, yc + h_2])


# </editor-fold>


# <editor-fold desc='box格式转换'>
def box2fmt(box: dict) -> str:
    for fmt in [BOX_FMT.XYXY, BOX_FMT.XYWH, BOX_FMT.XYWHA, BOX_FMT.XYWHAB, BOX_FMT.XLYL]:
        if fmt in box.keys():
            return fmt
    raise Exception('no fmt')


def box2xyxyN(box: dict) -> np.ndarray:
    fmt = box2fmt(box)
    return boxN2xyxyN(box[fmt], box_fmt=fmt)


def box2xywhN(box: dict) -> np.ndarray:
    fmt = box2fmt(box)
    return boxN2xywhN(box[fmt], box_fmt=fmt)


def box2xywhaN(box: dict) -> np.ndarray:
    fmt = box2fmt(box)
    return boxN2xywhaN(box[fmt], box_fmt=fmt)


def box2xywhabN(box: dict) -> np.ndarray:
    fmt = box2fmt(box)
    return boxN2xywhabN(box[fmt], box_fmt=fmt)


def box2xlylN(box: dict) -> np.ndarray:
    fmt = box2fmt(box)
    return boxN2xlylN(box[fmt], box_fmt=fmt)


def clsN_confN2box(clsN: np.ndarray, confN: np.ndarray, cls2name=None) -> dict:
    box = {'conf': confN, 'cls': int(clsN)}
    if cls2name is not None:
        box['name'] = cls2name(box['cls'])
    return box


# </editor-fold>

# <editor-fold desc='clses格式转换'>
def oclsesT2clses(oclsesT: torch.Tensor, cls2name=None) -> list:
    clses = []
    oclsesN = oclsesT.detach().cpu().numpy()
    clsesN = np.argmax(oclsesN, axis=-1)
    for i in range(oclsesT.size(0)):
        cls = {'cls': clsesN[i], 'ocls': oclsesN[i]}
        if cls2name is not None:
            cls['name'] = cls2name(cls['cls'])
        clses.append(cls)
    return clses


def clses2oclsesN(clses: list) -> np.ndarray:
    oclsesN = []
    for cls in clses:
        oclsesN.append(cls['ocls'])
    oclsesN = np.stack(oclsesN, axis=0)
    return oclsesN


def clses2clsesN(clses: list) -> np.ndarray:
    clsesN = []
    for cls in clses:
        clsesN.append(cls['cls'])
    clsesN = np.array(clsesN)
    return clsesN


def clses2confsN(clses: list) -> np.ndarray:
    confsN = []
    for cls in clses:
        confsN.append(cls['ocls'][cls['cls']])
    confsN = np.array(confsN)
    return confsN


# </editor-fold>
# <editor-fold desc='boxes格式转换'>

def boxes2clsesN(boxes: list) -> np.ndarray:
    clses = []
    for i, box in enumerate(boxes):
        clses.append(box['cls'])
    clses = np.array(clses, axis=0)
    return clses


def boxes2oclsesN(boxes: list, num_cls: int = 20) -> np.ndarray:
    oclses = [np.zeros(shape=(0, num_cls))]
    for i, box in enumerate(boxes):
        cls_i = np.zeros(shape=(1, num_cls))
        cls_i[:, box['cls']] = box['conf']
        oclses.append(cls_i)
    oclses = np.concatenate(oclses, axis=0)
    return oclses


def boxes2clsesN_diffsN(boxes: list) -> (np.ndarray, np.ndarray):
    clsesN = []
    diffsN = []
    for i, box in enumerate(boxes):
        clsesN.append(box['cls'])
        diffsN.append('difficult' in box.keys() and box['difficult'])
    clsesN = np.array(clsesN)
    diffsN = np.array(diffsN, dtype=np.bool_)
    return clsesN, diffsN


def boxes2clsesN_confsN(boxes: list) -> (np.ndarray, np.ndarray):
    clsesN = []
    confsN = []
    for i, box in enumerate(boxes):
        clsesN.append(box['cls'])
        confsN.append(box['conf'])
    clsesN = np.array(clsesN)
    confsN = np.array(confsN)
    return clsesN, confsN


def boxesN_confsN_clsesN2boxes(boxesN: np.ndarray, confsN: np.ndarray, clsesN: np.ndarray, box_fmt: str,
                               cls2name=None) -> list:
    boxes = []
    for i in range(boxesN.shape[0]):
        box = {'conf': confsN[i], 'cls': int(clsesN[i]), box_fmt: boxesN[i]}
        if cls2name is not None:
            box['name'] = cls2name(box['cls'])
        boxes.append(box)
    return boxes


def boxesT_confsT_clsesT2boxes(boxesT: torch.Tensor, confsT: torch.Tensor, clsesT: torch.Tensor, box_fmt: str,
                               cls2name=None) -> list:
    boxesN = boxesT.detach().cpu().numpy()
    confsN = confsT.detach().cpu().numpy()
    clsesN = clsesT.detach().cpu().numpy()
    return boxesN_confsN_clsesN2boxes(boxesN=boxesN, confsN=confsN, clsesN=clsesN, box_fmt=box_fmt, cls2name=cls2name)


def boxes2fmt(boxes: list) -> str:
    return box2fmt(boxes[0])


def boxes2boxesN(boxes: list, box_fmt: str) -> np.ndarray:
    boxesN = [np.zeros(shape=[0] + BOX_DIM[box_fmt])]
    for i, box in enumerate(boxes):
        boxesN.append(box[box_fmt][None, :])
    boxesN = np.concatenate(boxesN, axis=0)
    return boxesN

# </editor-fold>
