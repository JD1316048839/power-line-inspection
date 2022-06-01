from utils.label import *
import math


class IOU_TYPE:
    IOU = 'iou'
    GIOU = 'giou'
    CIOU = 'ciou'
    DIOU = 'diou'
    IAREA = 'iarea'


# <editor-fold desc='numpy'>
def _align_shapeN(arr1, arr2, last_axis=1):
    assert arr1.shape[-last_axis:] == arr2.shape[-last_axis:], 'shape err'
    shape1_ = arr1.shape[:-last_axis]
    shape2_ = arr2.shape[:-last_axis]
    val_shape_len = len(shape1_) + len(shape2_)
    full_shape_len = val_shape_len + last_axis
    trans = list(range(len(shape2_), val_shape_len)) + \
            list(range(len(shape2_))) + \
            list(range(val_shape_len, full_shape_len))
    arr1 = np.broadcast_to(arr1, list(shape2_) + list(arr1.shape)).transpose(trans)
    arr2 = np.broadcast_to(arr2, list(shape1_) + list(arr2.shape))
    return arr1, arr2


def any_arr_xywhsN(xywhs1, xywhs2, iou_type=IOU_TYPE.IOU):
    xymin1, xymax1 = xywhs1[..., :2] - xywhs1[..., 2:4] / 2, xywhs1[..., :2] + xywhs1[..., 2:4] / 2
    xymin2, xymax2 = xywhs2[..., :2] - xywhs2[..., 2:4] / 2, xywhs2[..., :2] + xywhs2[..., 2:4] / 2
    xymax_min = np.minimum(xymax1, xymax2)
    xymin_max = np.maximum(xymin1, xymin2)
    whi = np.maximum(xymax_min - xymin_max, 0)
    areai = np.prod(whi, axis=-1)
    if iou_type == IOU_TYPE.IAREA:
        return areai
    area1 = np.prod(xywhs1[..., 2:4], axis=-1)
    area2 = np.prod(xywhs2[..., 2:4], axis=-1)
    areau = area1 + area2 - areai
    iou = areai / areau
    if iou_type is None or iou_type == IOU_TYPE.IOU:
        return iou
    xymax_max = np.maximum(xymax1, xymax2)
    xymin_min = np.minimum(xymin1, xymin2)
    whb = xymax_max - xymin_min
    areab = np.prod(whb, axis=-1)
    if iou_type == IOU_TYPE.GIOU:
        return iou - (areab - areau) / areab
    diagb = np.sum(whb ** 2, axis=-1)
    diagc = np.sum((xywhs1[..., :2] - xywhs2[..., :2]) ** 2, axis=-1)
    diou = iou - diagc / diagb
    if iou_type == IOU_TYPE.DIOU:
        return diou
    r1 = np.arctan(xywhs1[..., 2] / xywhs1[..., 3])
    r2 = np.arctan(xywhs2[..., 2] / xywhs2[..., 3])
    v = ((r1 - r2) * 2 / math.pi) ** 2
    alpha = v * 2 / (1 - areai / areau + v + 1e-7)
    if iou_type == IOU_TYPE.CIOU:
        return diou - v * alpha
    raise Exception('err iou type ' + str(iou_type))


def any_arr_xyxysN(xyxys1, xyxys2, iou_type=IOU_TYPE.IOU):
    xymax_min = np.minimum(xyxys1[..., 2:4], xyxys2[..., 2:4])
    xymin_max = np.maximum(xyxys1[..., :2], xyxys2[..., :2])
    whi = np.maximum(xymax_min - xymin_max, 0)
    areai = np.prod(whi, axis=-1)
    if iou_type == IOU_TYPE.IAREA:
        return areai
    area1 = np.prod(xyxys1[..., 2:4] - xyxys1[..., :2], axis=-1)
    area2 = np.prod(xyxys2[..., 2:4] - xyxys2[..., :2], axis=-1)
    areau = area1 + area2 - areai
    iou = areai / areau
    if iou_type is None or iou_type == IOU_TYPE.IOU:
        return iou
    xymax_max = np.maximum(xyxys1[..., 2:4], xyxys2[..., 2:4])
    xymin_min = np.minimum(xyxys1[..., :2], xyxys2[..., :2])
    whb = xymax_max - xymin_min
    areab = np.prod(whb, axis=-1)
    if iou_type == IOU_TYPE.GIOU:
        return iou - (areab - areau) / areab
    xyc1 = (xyxys1[..., 2:4] + xyxys1[..., :2]) / 2
    xyc2 = (xyxys2[..., 2:4] + xyxys2[..., :2]) / 2
    diagb = np.sum(whb ** 2, axis=-1)
    diagc = np.sum((xyc1 - xyc2) ** 2, axis=-1)
    diou = iou - diagc / diagb
    if iou_type == IOU_TYPE.DIOU:
        return diou
    wh1 = xyxys1[..., 2:4] - xyxys1[..., :2]
    wh2 = xyxys2[..., 2:4] - xyxys2[..., :2]
    r1 = np.arctan(wh1[..., 0] / wh1[..., 1])
    r2 = np.arctan(wh2[..., 0] / wh2[..., 0])
    v = ((r1 - r2) * 2 / math.pi) ** 2
    alpha = v * 2 / (1 - areai / areau + v + 1e-7)
    if iou_type == IOU_TYPE.CIOU:
        return diou - v * alpha
    raise Exception('err iou type ' + str(iou_type))


def clac_iou_arrN(boxesN1, boxesN2, iou_type=IOU_TYPE.IOU, box_fmt=BOX_FMT.XYXY):
    assert boxesN1.shape == boxesN2.shape, 'shape err'
    if box_fmt == BOX_FMT.XYXY:
        return any_arr_xyxysN(boxesN1, boxesN2, iou_type=iou_type)
    elif box_fmt == BOX_FMT.XYWH:
        return any_arr_xywhsN(boxesN1, boxesN2, iou_type=iou_type)
    else:
        raise Exception('err box fmt ' + str(box_fmt))


def clac_iou_matN(boxesN1, boxesN2, iou_type=IOU_TYPE.IOU, box_fmt=BOX_FMT.XYXY):
    boxesN1, boxesN2 = _align_shapeN(boxesN1, boxesN2, last_axis=1)
    return clac_iou_arrN(boxesN1, boxesN2, iou_type=iou_type, box_fmt=box_fmt)


def clac_iou_mat_box(boxesN1, boxesN2, box_fmt1=BOX_FMT.XYXY, box_fmt2=BOX_FMT.XYXY, iou_type=IOU_TYPE.IOU):
    if box_fmt1 == box_fmt2:
        return clac_iou_matN(boxesN1, boxesN2, iou_type=iou_type, box_fmt=box_fmt1)
    elif box_fmt1 == BOX_FMT.XYXY and box_fmt2 == BOX_FMT.XYWH:
        boxesN2 = xywhsN2xyxysN(boxesN2)
        return clac_iou_matN(boxesN1, boxesN2, iou_type=iou_type, box_fmt=BOX_FMT.XYXY)
    elif box_fmt1 == BOX_FMT.XYWH and box_fmt2 == BOX_FMT.XYXY:
        boxesN1 = xywhsN2xyxysN(boxesN1)
        return clac_iou_matN(boxesN1, boxesN2, iou_type=iou_type, box_fmt=BOX_FMT.XYXY)
    else:
        raise Exception('err box fmt ' + str(box_fmt1))

# <editor-fold desc='torch'>
def _align_shapeT(ten1, ten2, last_axis=1):
    assert list(ten1.size())[-last_axis:] == list(ten2.size())[-last_axis:], 'shape err'
    shape1_ = list(ten1.size())[:-last_axis]
    shape2_ = list(ten2.size())[:-last_axis]
    val_shape_len = len(shape1_) + len(shape2_)
    full_shape_len = val_shape_len + last_axis
    trans = list(range(len(shape2_), val_shape_len)) + \
            list(range(len(shape2_))) + \
            list(range(val_shape_len, full_shape_len))
    ten1 = torch.broadcast_to(ten1, list(shape2_) + list(ten1.shape)).permute(*trans)
    ten2 = torch.broadcast_to(ten2, list(shape1_) + list(ten2.shape))
    return ten1, ten2


def any_arr_xywhsT(xywhs1, xywhs2, iou_type=IOU_TYPE.IOU):
    xymin1, xymax1 = xywhs1[..., :2] - xywhs1[..., 2:4] / 2, xywhs1[..., :2] + xywhs1[..., 2:4] / 2
    xymin2, xymax2 = xywhs2[..., :2] - xywhs2[..., 2:4] / 2, xywhs2[..., :2] + xywhs2[..., 2:4] / 2
    xymax_min = torch.minimum(xymax1, xymax2)
    xymin_max = torch.maximum(xymin1, xymin2)
    whi = torch.clamp(xymax_min - xymin_max, min=0)
    areai = torch.prod(whi, dim=-1)
    if iou_type == IOU_TYPE.IAREA:
        return areai
    area1 = torch.prod(xywhs1[..., 2:4], dim=-1)
    area2 = torch.prod(xywhs2[..., 2:4], dim=-1)
    areau = area1 + area2 - areai
    iou = areai / areau
    if iou_type is None or iou_type == IOU_TYPE.IOU:
        return iou
    xymax_max = torch.maximum(xymax1, xymax2)
    xymin_min = torch.minimum(xymin1, xymin2)
    whb = xymax_max - xymin_min
    areab = torch.prod(whb, dim=-1)
    if iou_type == IOU_TYPE.GIOU:
        return iou - (areab - areau) / areab
    diagb = torch.sum(whb ** 2, dim=-1)
    diagc = torch.sum((xywhs1[..., :2] - xywhs2[..., :2]) ** 2, dim=-1)
    diou = iou - diagc / diagb
    if iou_type == IOU_TYPE.DIOU:
        return diou
    r1 = torch.arctan(xywhs1[..., 2] / xywhs1[..., 3])
    r2 = torch.arctan(xywhs2[..., 2] / xywhs2[..., 3])
    v = ((r1 - r2) * 2 / math.pi) ** 2
    alpha = v * 2 / (1 - areai / areau + v + 1e-7)
    if iou_type == IOU_TYPE.CIOU:
        return diou - v * alpha
    raise Exception('err iou type ' + str(iou_type))


def any_arr_xyxysT(xyxys1, xyxys2, iou_type=IOU_TYPE.IOU):
    xymax_min = torch.minimum(xyxys1[..., 2:4], xyxys2[..., 2:4])
    xymin_max = torch.maximum(xyxys1[..., :2], xyxys2[..., :2])
    whi = torch.clamp(xymax_min - xymin_max, min=0)
    areai = torch.prod(whi, dim=-1)
    if iou_type == IOU_TYPE.IAREA:
        return areai
    area1 = torch.prod(xyxys1[..., 2:4] - xyxys1[..., :2], dim=-1)
    area2 = torch.prod(xyxys2[..., 2:4] - xyxys2[..., :2], dim=-1)
    areau = area1 + area2 - areai
    iou = areai / areau
    if iou_type is None or iou_type == IOU_TYPE.IOU:
        return iou
    xymax_max = torch.maximum(xyxys1[..., 2:4], xyxys2[..., 2:4])
    xymin_min = torch.minimum(xyxys1[..., :2], xyxys2[..., :2])
    whb = xymax_max - xymin_min
    areab = torch.prod(whb, dim=-1)
    if iou_type == IOU_TYPE.GIOU:
        return iou - (areab - areau) / areab
    xyc1 = (xyxys1[..., 2:4] + xyxys1[..., :2]) / 2
    xyc2 = (xyxys2[..., 2:4] + xyxys2[..., :2]) / 2
    diagb = torch.sum(whb ** 2, dim=-1)
    diagc = torch.sum((xyc1 - xyc2) ** 2, dim=-1)
    diou = iou - diagc / diagb
    if iou_type == IOU_TYPE.DIOU:
        return diou
    wh1 = xyxys1[..., 2:4] - xyxys1[..., :2]
    wh2 = xyxys2[..., 2:4] - xyxys2[..., :2]
    r1 = torch.arctan(wh1[..., 0] / wh1[..., 1])
    r2 = torch.arctan(wh2[..., 0] / wh2[..., 0])
    v = ((r1 - r2) * 2 / math.pi) ** 2
    alpha = v * 2 / (1 - areai / areau + v + 1e-7)
    if iou_type == IOU_TYPE.CIOU:
        return diou - v * alpha
    raise Exception('err iou type ' + str(iou_type))


def clac_iou_arrT(boxes1, boxes2, iou_type=IOU_TYPE.IOU, box_fmt=BOX_FMT.XYXY):
    if box_fmt == BOX_FMT.XYXY:
        return any_arr_xyxysT(boxes1, boxes2, iou_type=iou_type)
    elif box_fmt == BOX_FMT.XYWH:
        return any_arr_xywhsT(boxes1, boxes2, iou_type=iou_type)
    else:
        raise Exception('err box fmt ' + str(box_fmt))


def clac_iou_matT(boxes1, boxes2, iou_type=IOU_TYPE.IOU, box_fmt=BOX_FMT.XYXY):
    boxes1, boxes2 = _align_shapeT(boxes1, boxes2, last_axis=1)
    return clac_iou_arrT(boxes1, boxes2, iou_type=iou_type, box_fmt=box_fmt)


if __name__ == '__main__':
    import time

    boxesT1 = torch.rand(9000 * 9000, 4).to(torch.device(0))
    boxesT2 = torch.rand(9000 * 9000, 4).to(torch.device(0))

    time1 = time.time()
    ious2 = clac_iou_arrT(boxesT1, boxesT2, iou_type=IOU_TYPE.IOU, box_fmt=BOX_FMT.XYWH)
    time2 = time.time()
    print('muti ', time2 - time1)
