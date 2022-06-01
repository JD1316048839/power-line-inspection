from utils.iou import *
import torchvision
import time


class NMS_TYPE:
    HARD = 'hard'
    SOFT = 'soft'
    NOOP = 'none'


def nms_softN(boxesN, confsN, iou_thres=0.45, iou_type=IOU_TYPE.IOU, box_fmt=BOX_FMT.XYXY):
    confsN = copy.deepcopy(confsN)  # 隔离
    conf_thres = np.min(confsN)  # 阈值
    iou_mat = clac_iou_matN(boxesN, boxesN, iou_type=iou_type, box_fmt=box_fmt)
    prsv_inds = []
    for i in range(len(boxesN)):
        ind = np.argmax(confsN)
        if confsN[ind] < conf_thres:
            break
        prsv_inds.append(ind)
        # 相关box抑制
        mask = (iou_mat[ind, :] > iou_thres)
        confsN[mask] *= (1 - iou_mat[ind, mask])
        # 移除ind相关
        iou_mat[:, ind] = 0
        confsN[ind] = 0

    prsv_inds = np.array(prsv_inds)
    return prsv_inds


def nms_hardN(boxesN, confsN, iou_thres=0.45, iou_type=IOU_TYPE.IOU, box_fmt=BOX_FMT.XYXY):
    confsN = copy.deepcopy(confsN)  # 隔离
    order = np.argsort(-confsN)
    boxesN, confsN = boxesN[order], confsN[order]
    prsv_inds = []
    for i in range(boxesN.shape[0]):
        if confsN[i] == 0:
            continue
        prsv_inds.append(order[i])
        # 相关box抑制
        res_inds = i + 1 + np.nonzero(confsN[i + 1:] > 0)[0]
        boxesN1 = np.repeat(boxesN[i:i + 1], repeats=len(res_inds), axis=0)
        ious = clac_iou_arrN(boxesN1, boxesN[res_inds], iou_type=iou_type, box_fmt=box_fmt)

        confsN[res_inds[ious > iou_thres]] = 0
    prsv_inds = np.array(prsv_inds)
    return prsv_inds


def nmsN(boxesN, confsN, clsesN=None, iou_thres=0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU,
         box_fmt=BOX_FMT.XYXY):
    if clsesN is None:
        if nms_type == NMS_TYPE.SOFT:
            return nms_softN(boxesN, confsN, iou_thres=iou_thres, iou_type=iou_type, box_fmt=box_fmt)
        elif nms_type == NMS_TYPE.HARD:
            return nms_hardN(boxesN, confsN, iou_thres=iou_thres, iou_type=iou_type, box_fmt=box_fmt)
        elif nms_type == NMS_TYPE.NOOP or nms_type is None:
            return np.arange(boxesN.shape[0])
        else:
            raise Exception('nms type err')
    else:
        prsv_inds = []
        num_cls = int(np.max(clsesN))
        for i in range(num_cls + 1):
            inds = clsesN == i
            if np.any(inds):
                boxesN_cls = boxesN[inds]
                confsN_cls = confsN[inds]
                prsv_inds_cls = nmsN(boxesN_cls, confsN_cls, clsesN=None, iou_thres=iou_thres,
                                     nms_type=nms_type, iou_type=iou_type, box_fmt=box_fmt)
                inds = np.nonzero(inds)[0]
                prsv_inds.append(inds[prsv_inds_cls])
        prsv_inds = np.concatenate(prsv_inds, axis=0)
    return prsv_inds


# SOFT NMS
def nms_softT(boxesT, confsT, iou_thres=0.45, iou_type=IOU_TYPE.IOU, box_fmt=BOX_FMT.XYXY):
    iou_mat = clac_iou_matT(boxesT, boxesT, iou_type=iou_type, box_fmt=box_fmt)
    confsT = copy.deepcopy(confsT)  # 隔离
    conf_thres = torch.min(confsT)  # 阈值
    prsv_inds = []
    for i in range(boxesT.size(0)):
        ind = torch.argmax(confsT)
        if confsT[ind] < conf_thres:
            break
        prsv_inds.append(ind)
        # 相关box抑制
        mask = (iou_mat[ind, :] > iou_thres)
        confsT[mask] *= (1 - iou_mat[ind, mask])
        # 移除ind相关
        iou_mat[:, ind] = 0
        confsT[ind] = 0

    prsv_inds = torch.Tensor(prsv_inds).long()
    return prsv_inds


# HARD NMS
def nms_hardT(boxesT, confsT, iou_thres=0.45, iou_type=IOU_TYPE.IOU, box_fmt=BOX_FMT.XYXY):
    confsT = copy.deepcopy(confsT)  # 隔离
    order = torch.argsort(confsT, descending=True)
    boxesT, confsT = boxesT[order], confsT[order]
    prsv_inds = []
    for i in range(boxesT.size(0)):
        if confsT[i] == 0:
            continue
        prsv_inds.append(order[i])
        # 相关box抑制
        res_inds = i + 1 + torch.nonzero(confsT[i + 1:] > 0, as_tuple=True)[0]
        boxesT1 = boxesT[i:i + 1].repeat(len(res_inds), 1)
        ious = clac_iou_arrT(boxesT1, boxesT[res_inds], iou_type=iou_type, box_fmt=box_fmt)
        confsT[res_inds[ious > iou_thres]] = 0

    prsv_inds = torch.Tensor(prsv_inds).long()
    return prsv_inds


# NMS
def nmsT(boxesT, confsT, clsesT=None, iou_thres=0.45, nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU,
         box_fmt=BOX_FMT.XYXY):
    if clsesT is None:
        if nms_type == NMS_TYPE.SOFT:
            return nms_softT(boxesT, confsT, iou_thres=iou_thres, iou_type=iou_type, box_fmt=box_fmt)
        elif nms_type == NMS_TYPE.HARD:
            if iou_type == IOU_TYPE.IOU and box_fmt == BOX_FMT.XYXY:
                return torchvision.ops.nms(boxesT, confsT, iou_threshold=iou_thres)
            else:
                return nms_hardT(boxesT, confsT, iou_thres=iou_thres, iou_type=iou_type, box_fmt=box_fmt)
        elif nms_type == NMS_TYPE.NOOP or nms_type is None:
            return torch.arange(boxesT.size(0))
        else:
            raise Exception('nms type err')
    else:
        prsv_inds = []
        num_cls = int(torch.max(clsesT).item())
        for i in range(num_cls + 1):
            inds = clsesT == i
            if torch.any(inds):
                boxes_cls = boxesT[inds]
                confs_cls = confsT[inds]
                prsv_inds_cls = nmsT(boxes_cls, confs_cls, clsesT=None, iou_thres=iou_thres,
                                     nms_type=nms_type, iou_type=iou_type)
                inds = torch.nonzero(inds, as_tuple=False).squeeze(dim=1)
                prsv_inds.append(inds[prsv_inds_cls])
        prsv_inds = torch.cat(prsv_inds, dim=0)
    return prsv_inds
