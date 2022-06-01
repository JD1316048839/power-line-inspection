from utils.iou import *
import torch
import copy


# 变换boxes
def affine_boxes(boxes, bias=(0, 0), scale=(1, 1), rotate=0, pn_min=(0, 0), pn_max=(400, 400)):
    scale = np.squeeze(np.array(scale))
    scale = np.repeat(scale, repeats=2) if len(scale.shape) < 1 else scale
    cos, sin = np.cos(rotate), np.sin(rotate)
    weight = np.array([[cos, sin], [-sin, cos]]) * scale
    bias = np.squeeze(np.array(bias))
    bias = np.repeat(bias, repeats=2) if len(bias.shape) < 1 else bias
    boxes_new = boxes.empty() if isinstance(boxes, BoxList) else []
    for box in boxes:
        box_fmt = box2fmt(box)
        xlyl = boxN2xlylN(box[box_fmt], box_fmt=box_fmt)
        xlyl_trand = xlyl @ weight + bias[None, :]
        boxN = xlylN2boxN(xlyl_trand, box_fmt)
        boxN, cliped = boxN_clip(boxN, box_fmt=box_fmt, xy_min=list(pn_min), xy_max=list(pn_max), thres=10)
        if cliped:
            continue
        box_new = copy.deepcopy(box)
        box_new[box_fmt] = boxN
        boxes_new.append(box_new)
    return boxes_new


class CusTransform():
    def __init__(self):
        pass

    def __call__(self, imgs, labels):
        return imgs, labels


# 标准tnsor输出格式CHW PIL格式HWC
class CusToTensor(CusTransform):
    def __init__(self, concat=True):
        super().__init__()
        self.concat = concat

    def __call__(self, imgs, labels):
        for i in range(len(imgs)):
            img = np.transpose(imgs[i], (2, 0, 1))
            img = np.expand_dims(img, 0)
            imgs[i] = torch.Tensor(img.copy()).float() / 255
        if self.concat:
            imgs = torch.cat(imgs, dim=0)
        return imgs, labels


class ADD_TYPE:
    APPEND = 'append'
    REPLACE = 'replace'
    COVER = 'cover'


# 基类
class DetMutiMixer(CusTransform):
    INPUT_SAME_SIZE = False
    OUTPUT_SAME_SIZE = False

    def __init__(self, repeat=3, inupt_num=0, add_type=False):
        super().__init__()
        self.inupt_num = inupt_num
        self.repeat = repeat
        self.add_type = add_type

    def __call__(self, imgs, boxess):
        if len(imgs) < self.inupt_num:
            return imgs, boxess
        repeat = self.repeat
        if isinstance(repeat, float):
            repeat = int(np.ceil(repeat * len(imgs)))
        imgs_procd = []
        boxess_procd = []
        for n in range(repeat):
            inds = np.random.choice(a=len(imgs), replace=False, size=self.inupt_num)
            imgs_c = [imgs[int(ind)] for ind in inds]
            boxess_c = [boxess[int(ind)] for ind in inds]
            # 函数
            img_procd, boxes_procd = self.forward(arrs2pils(imgs_c), boxess_c)
            img_procd = pil2arr(img_procd)
            imgs_procd.append(img_procd)
            boxess_procd.append(boxes_procd)
        imgs_procd = pils2arrs(imgs_procd)
        if self.add_type == ADD_TYPE.REPLACE:
            return imgs_procd, boxess_procd
        elif self.add_type == ADD_TYPE.APPEND:
            imgs += imgs_procd
            boxess += boxess_procd
            return imgs, boxess
        elif self.add_type == ADD_TYPE.COVER:
            cover_num = min(len(imgs), repeat)
            inds_dist = np.random.choice(a=len(imgs), replace=False, size=cover_num)
            inds_src = np.random.choice(a=repeat, replace=False, size=cover_num)
            for ind_dist, ind_src in zip(inds_dist, inds_src):
                imgs[ind_dist] = imgs_procd[ind_src]
                boxess[ind_dist] = boxess_procd[ind_src]
            return imgs, boxess
        else:
            raise Exception('err type')

    def forward(self, imgs, boxess):
        raise NotImplementedError


# 变换序列
class MixSequence(DetMutiMixer):
    INPUT_SAME_SIZE = False
    OUTPUT_SAME_SIZE = False

    def __init__(self, mix_list, add_type=ADD_TYPE.COVER):
        self.mix_list = mix_list
        repeat = 0
        inupt_num = 0
        for mix in self.mix_list:
            repeat += mix.repeat
            inupt_num = max(inupt_num, mix.inupt_num)
        super().__init__(repeat=repeat, inupt_num=inupt_num, add_type=add_type)

    def __call__(self, imgs, boxess):
        assert len(imgs) >= self.inupt_num, 'num err'
        imgs_aug = []
        boxess_aug = []
        for mix in self.mix_list:
            imgs_a, boxess_a = mix(imgs, boxess)
            imgs_aug += imgs_a
            boxess_aug += boxess_a
        return imgs_aug, boxess_aug


# 按透明度混合
class MixAlpha(DetMutiMixer):
    INPUT_SAME_SIZE = True
    OUTPUT_SAME_SIZE = True

    def __init__(self, repeat=3, mix_rate=0.5, add_type=ADD_TYPE.COVER):
        super().__init__(repeat=repeat, inupt_num=2, add_type=add_type)
        self.mix_rate = mix_rate

    def forward(self, imgs, boxess):
        img = Image.blend(imgs[0], imgs[1], self.mix_rate)
        boxess = copy.deepcopy(boxess)  # 隔离
        boxes = []
        for box in boxess[0]:
            box['conf'] = box['conf'] * (1 - self.mix_rate)
            boxes.append(box)
        for box in boxess[1]:
            box['conf'] = box['conf'] * self.mix_rate
            boxes.append(box)
        return img, boxes


class Mosaic(DetMutiMixer):
    INPUT_SAME_SIZE = True
    OUTPUT_SAME_SIZE = True

    def __init__(self, repeat=3, img_size=(416, 416), add_type=ADD_TYPE.COVER):
        super().__init__(repeat=repeat, inupt_num=4, add_type=add_type)
        self.img_size = np.array(img_size)

    def forward(self, imgs, boxess):
        w, h = self.img_size
        boxess = copy.deepcopy(boxess)  # 隔离
        wp = int((np.random.rand() * 0.5 + 0.25) * w)
        hp = int((np.random.rand() * 0.5 + 0.25) * h)
        # 定义box偏移量
        box_xyxy = np.array([
            [0, 0, wp, hp],
            [0, hp, wp, h],
            [wp, 0, w, hp],
            [wp, hp, w, h]
        ])
        # 定义img偏移量
        img_wh = np.array([img.size for img in imgs], dtype=np.float64)
        scale = np.random.uniform(0.6, 1, size=4) * np.sqrt(w * h / img_wh[:, 0] / img_wh[:, 1])
        img_wh *= scale[:, None]
        img_xyxy = np.array([
            [wp - img_wh[0, 0], hp - img_wh[0, 1], wp, hp],
            [hp - img_wh[1, 1], hp, wp, wp + img_wh[1, 0]],
            [wp, hp - img_wh[2, 1], wp + img_wh[2, 0], hp],
            [wp, hp, wp + img_wh[3, 0], hp + img_wh[3, 1]]
        ])
        # 整合
        img_sum = Image.new('RGB', (w, h))
        boxes_sum = []
        for i in range(4):
            boxes = affine_boxes(boxess[i], scale=scale[i], bias=img_xyxy[i, :2],
                                 pn_min=box_xyxy[i, :2], pn_max=box_xyxy[i, 2:])
            img = imgs[i].resize(size=img_wh[i].astype(np.int32))
            img_sum.paste(img, box=tuple(img_xyxy[i, :2].astype(np.int32)))
            boxes_sum += boxes
        img_sum = pil2arr(img_sum)
        return img_sum, boxes_sum


# 目标区域的裁剪混合
class CutMix(DetMutiMixer):
    INPUT_SAME_SIZE = False
    OUTPUT_SAME_SIZE = False

    def __init__(self, repeat=3, num_patch=2, add_type=ADD_TYPE.COVER):
        super().__init__(repeat=repeat, inupt_num=2, add_type=add_type)
        self.num_patch = num_patch

    # 随机抠图并遮挡
    def forward(self, imgs, boxess):
        # num_patch
        num_box = len(boxess[0])
        num_patch = self.num_patch
        if isinstance(num_patch, float):
            num_patch = int(np.ceil(num_patch * len(boxess[1])))
        if num_patch == 0 or len(boxess[1]) == 0:
            return imgs[0], boxess[0]
        img_mixed = copy.deepcopy(imgs[0])
        w, h = img_mixed.size
        boxes_mixed = dict(zip(range(num_box), copy.deepcopy(boxess[0])))
        boxes_xyxy_dist = boxes2boxesN(boxess[0], box_fmt=BOX_FMT.XYXY)
        # 提取patch
        patches = []
        patches_xyxy = []
        boxes_patch = []
        for i in range(num_patch):
            ind = np.random.choice(a=len(boxess[1]))
            box_src = box2xyxyN(boxess[1][ind])
            patch = imgs[1].crop(box_src)
            patch_size = np.array(patch.size)
            patch_size = np.maximum((patch_size * np.random.uniform(0.5, 1.5)), 5).astype(np.int32)
            patch = patch.resize(tuple(patch_size), Image.ANTIALIAS)
            patches.append(patch)

            x1, y1 = np.random.uniform(0, w - patch_size[0]), np.random.uniform(0, h - patch_size[1])
            patch_xyxy = np.array([x1, y1, x1 + patch_size[0], y1 + patch_size[1]])
            patches_xyxy.append(patch_xyxy)

            box_patch = copy.deepcopy(boxess[1][ind])
            box_patch['xyxy'] = patch_xyxy
            boxes_patch.append(box_patch)

        patches_xyxy = np.array(patches_xyxy).astype(np.int32)
        boxes_xyxy_all = np.concatenate([boxes_xyxy_dist, patches_xyxy], axis=0)
        boxes_xywh_all = xyxysN2xywhsN(boxes_xyxy_all)
        iarea = clac_iou_matN(patches_xyxy, boxes_xyxy_all, iou_type=IOU_TYPE.IAREA)
        ratios = iarea / boxes_xywh_all[:, 2] / boxes_xywh_all[:, 3]
        # 放置patch
        for i in range(num_patch):
            if num_box > 0 and np.max(ratios[i, :(i + num_box)]) > 0.6:  # 防止新粘贴的图像影响原有目标
                continue
            if np.any(np.array(patches[i].size) < 10):  # 过滤掉太小的目标
                continue
            for j, box in boxes_mixed.items():
                box['conf'] = box['conf'] * (1 - ratios[i, j])
            # 放置patch
            img_mixed.paste(patches[i], box=tuple(patches_xyxy[i, :2]))
            boxes_mixed[i + num_box] = boxes_patch[i]
        boxes_mixed = list(boxes_mixed.values())
        return img_mixed, boxes_mixed
