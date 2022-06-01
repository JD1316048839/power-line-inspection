import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import copy
import torchvision.transforms as transforms
import torchvision
from utils.label import xyxyN2xlylN, xlylN2xyxyN, box2fmt, xlylN2boxN, boxN2xlylN, boxN_clip


# torch定义联合变换
def tor_transform(imgs, labels, seq):
    imgs_aug = []
    for i in range(len(imgs)):
        img_aug = seq(imgs[i])
        imgs_aug.append(img_aug)
    return imgs_aug, labels


# Imgaug联合变换
def iaa_transform(imgs, labels, seq):
    # seq = seq.to_deterministic()
    if labels is None or isinstance(labels[0], dict):
        imgs = [seq(image=np.array(img)) for img in imgs]
        return imgs, labels
    elif isinstance(labels[0], list):
        imgs_aug = []
        boxess_aug = []
        for img, boxes in zip(imgs, labels):
            kps = []
            box_fmts = []
            for i, box in enumerate(boxes):
                box_fmt = box2fmt(box)
                xlyl = boxN2xlylN(box[box_fmt], box_fmt=box_fmt)
                kps += [ia.Keypoint(x=x, y=y) for x, y in xlyl]
                box_fmts.append(box_fmt)
            kps = ia.KeypointsOnImage(kps, shape=img.shape)
            img_aug, kps_aug = seq(image=img, keypoints=kps)
            h, w, _ = img_aug.shape
            boxes_aug = []
            for i, box in enumerate(boxes):
                xlyl = np.array([(kps_aug[i * 4 + j].x, kps_aug[i * 4 + j].y) for j in range(4)])
                # 范围检验
                box_fmt = box_fmts[i]
                boxN = xlylN2boxN(xlyl, box_fmt)
                boxN, cliped = boxN_clip(boxN, box_fmt=box_fmt, xy_min=[0, 0], xy_max=[w, h], thres=10)
                if cliped:
                    continue
                box_aug = copy.deepcopy(box)
                box_aug[box_fmt] = boxN
                boxes_aug.append(box_aug)
            imgs_aug.append(img_aug)
            boxess_aug.append(boxes_aug)
        return imgs_aug, boxess_aug
    else:
        raise Exception('aug err')
