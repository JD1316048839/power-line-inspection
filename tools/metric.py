import pandas as pd
from utils.iou import *
import threading
from aug import AugSeq
from utils.device import PACK, select_device
from functools import partial
from utils.frame import IDetectionInfer, IClassificationInfer, ModelFrame, ModelInferFrame
from abc import ABCMeta, abstractmethod
import io
import time


class Prefetcher():
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device) if device.index is not None else None

    @property
    def sampler(self):
        return self.loader.sampler

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self.thread = threading.Thread(target=self.load, daemon=True)
        self.thread.start()
        return self

    def __next__(self):
        self.thread.join()
        if self.imgs is None:
            raise StopIteration
        else:
            imgs, gtss = self.imgs, self.gtss
            self.thread = threading.Thread(target=self.load)
            self.thread.start()
            return imgs, gtss

    def load(self):
        try:
            self.imgs, self.gtss = next(self.loader_iter)
            if self.stream is not None:
                with torch.cuda.stream(self.stream):
                    self.imgs = self.imgs.to(device=self.device, non_blocking=True)
        except StopIteration:
            self.imgs, self.gtss = None, None
        return None


def print_data(data):
    data_str = str(data)
    lines = data_str.split('\n')
    for line in lines:
        print(line)
    return None


def print_data_with_fliter(data, filters=('Conv2d', 'Linear')):
    select_inds = data['Class'] == filters[0]
    for i in range(1, len(filters)):
        select_inds = select_inds | (data['Class'] == filters[i])
    data_flited = data.loc[select_inds]
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', 500)
    # pd.set_option('display.max_colwidth', 500)
    # pd.set_option('display.max_rows', None)
    print_data(data_flited)
    return data


class Metric(metaclass=ABCMeta):

    @abstractmethod
    def __call__(self, model):
        pass

    @staticmethod
    def Accuracy(loader, top_nums=(1, 5), **kwargs):
        return MetricAccuracy(loader=loader, top_nums=top_nums, **kwargs)

    @staticmethod
    def PrecRcal(loader, total_epoch=1, **kwargs):
        return MetricPrecRcal(loader=loader, total_epoch=total_epoch, **kwargs)

    @staticmethod
    def AUC(loader, total_epoch=1, **kwargs):
        return MetricAUC(loader=loader, total_epoch=total_epoch, **kwargs)

    @staticmethod
    def MAP(loader, **kwargs):
        return MetricMAP(loader=loader, **kwargs)

class MetricMAP(Metric):
    def __init__(self, loader, **kwargs):
        self.kwargs = kwargs
        self.loader = loader

    @property
    def img_size(self):
        return self.loader.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.loader.img_size = img_size

    def __call__(self, model):
        data = calc_map(model=model, loader=self.loader, num_cls=self.loader.num_cls, cls2name=self.loader.cls2name,
                        **self.kwargs)

        print_data(data)
        return np.array(data['AP'])[-1]


class MetricAccuracy(Metric):
    def __init__(self, loader, top_nums=(1, 5), **kwargs):
        self.kwargs = kwargs
        self.loader = loader
        self.top_nums = top_nums

    @property
    def img_size(self):
        return self.loader.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.loader.img_size = img_size

    def __call__(self, model):
        accs = calc_acc(model=model, loader=self.loader, top_nums=self.top_nums, cls2name=self.loader.cls2name,
                        **self.kwargs)
        msg = 'Accuracy ' + ' '.join(['Top%d' % num + ' %5.5f' % a for num, a in zip(self.top_nums, accs)])
        print(msg)
        return accs[0]


class MetricPrecRcal(Metric):
    def __init__(self, loader, total_epoch=1, **kwargs):
        self.kwargs = kwargs
        self.loader = loader
        self.total_epoch = total_epoch

    @property
    def img_size(self):
        return self.loader.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.loader.img_size = img_size

    def __call__(self, model):
        data = calc_prec_recall(model=model, loader=self.loader, total_epoch=self.total_epoch,
                                num_cls=self.loader.num_cls, cls2name=self.loader.cls2name, **self.kwargs)
        print_data(data)
        return np.array(data['Accuracy'])[-1]


class MetricF1(Metric):
    def __init__(self, loader, total_epoch=1, **kwargs):
        self.kwargs = kwargs
        self.loader = loader
        self.total_epoch = total_epoch

    @property
    def img_size(self):
        return self.loader.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.loader.img_size = img_size

    def __call__(self, model):
        data = calc_prec_recall(model=model, loader=self.loader, total_epoch=self.total_epoch,
                                num_cls=self.loader.num_cls, cls2name=self.loader.cls2name, **self.kwargs)
        print_data(data)
        return np.array(data['F1'])[0]


class MetricAUC(Metric):
    def __init__(self, loader, total_epoch=1, **kwargs):
        self.kwargs = kwargs
        self.loader = loader
        self.total_epoch = total_epoch

    @property
    def img_size(self):
        return self.loader.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.loader.img_size = img_size

    def __call__(self, model):
        data = calc_auc(model=model, loader=self.loader, total_epoch=self.total_epoch,
                        num_cls=self.loader.num_cls, cls2name=self.loader.cls2name, **self.kwargs)
        print_data(data)
        return np.array(data['AUC'])[-1]


def calc_acc(model, loader, top_nums=(1, 5), cls2name=None, **kwargs):
    assert isinstance(model, IClassificationInfer), 'model err'
    if isinstance(model, ModelFrame):
        ploader = Prefetcher(loader, model.device) if model.device.index is not None else loader
        model.eval()
    else:
        ploader = loader
    if isinstance(top_nums, int):
        top_nums = [top_nums]
    total = 0
    correct = np.zeros(shape=len(top_nums))
    num_batch = len(loader)

    for n, (imgs, cts) in enumerate(ploader):
        if n % 100 == 0 or n == num_batch - 1:
            print('Testing %5d' % (n + 1) + ' / %5d' % num_batch)
        ctsN = clses2clsesN(cts)
        clses = model.imgs2clses(imgs, cls2name=cls2name, **kwargs)
        oclsesN = clses2oclsesN(clses)
        order = np.argsort(-oclsesN, axis=-1)
        for i, num in enumerate(top_nums):
            for j, ctN in enumerate(ctsN):
                if ctN in order[j, :num]:
                    correct[i] += 1
        total += len(imgs)

    accs = correct / total
    return accs


def calc_auc(model, loader, total_epoch=1, num_cls=1, cls2name=None, **kwargs):
    assert isinstance(model, IClassificationInfer), 'model err'
    if isinstance(model, ModelFrame):
        ploader = Prefetcher(loader, model.device) if model.device.index is not None else loader
        model.eval()
    else:
        ploader = loader

    ctsN = []
    oclsesN = []
    num_batch = len(loader)
    for n in range(total_epoch):
        for i, (imgs, cts) in enumerate(ploader):
            if i % 100 == 0 or i == num_batch - 1:
                print('Testing Epoch %4d' % (n + 1) + ' / %-4d' % total_epoch +
                      ' Iter %4d' % (i + 1) + ' / %-4d' % num_batch)
            ctsN_i = clses2clsesN(cts)
            clses_i = model.imgs2clses(imgs, cls2name=cls2name, **kwargs)
            oclsesN_i = clses2oclsesN(clses_i)
            ctsN.append(ctsN_i)
            oclsesN.append(oclsesN_i)

    ctsN = np.concatenate(ctsN, axis=0)
    oclsesN = np.concatenate(oclsesN, axis=0)
    clsesN = np.argmax(oclsesN, axis=1)
    aucs = auc_per_class(oclses=oclsesN, cts=ctsN)
    # 统计结果
    data = pd.DataFrame(columns=['Class', 'Target', 'Pred', 'Pos', 'AUC'])
    for i in range(num_cls):
        cts_mask = ctsN == i
        clses_mask = clsesN == i
        tp = np.sum(cts_mask * clses_mask)
        cls_name = cls2name(i) if cls2name is not None else i
        data = pd.concat([data, pd.DataFrame.from_dict({
            'Class': cls_name, 'Target': np.sum(cts_mask), 'Pred': np.sum(clses_mask),
            'Pos': tp, 'AUC': aucs[i]
        })])
    if num_cls > 2:
        data = data.append({
            'Class': 'Total', 'Target': np.sum(data['Target']), 'Pred': np.sum(data['Pred']),
            'Pos': np.sum(data['Pos']), 'AUC': np.average(data['AUC'])
        }, ignore_index=True)
    return data


def calc_prec_recall(model, loader, total_epoch=1, num_cls=1, cls2name=None, **kwargs):
    assert isinstance(model, IClassificationInfer), 'model err'
    if isinstance(model, ModelFrame):
        ploader = Prefetcher(loader, model.device) if model.device.index is not None else loader
        model.eval()
    else:
        ploader = loader

    ctsN = []
    clsesN = []
    num_batch = len(loader)
    for n in range(total_epoch):
        for i, (imgs, cts) in enumerate(ploader):
            if i % 100 == 0 or i == num_batch - 1:
                print('Testing Epoch %4d' % (n + 1) + ' / %-4d' % total_epoch +
                      ' Iter %4d' % (i + 1) + ' / %-4d' % num_batch)
            ctsN_i = clses2clsesN(cts)
            clses_i = model.imgs2clses(imgs, cls2name=cls2name, **kwargs)
            clsesN_i = clses2clsesN(clses_i)
            ctsN.append(ctsN_i)
            clsesN.append(clsesN_i)

    ctsN = np.concatenate(ctsN, axis=0)
    clsesN = np.concatenate(clsesN, axis=0)
    # 计算每一类的具体情况
    data = pd.DataFrame(columns=['Class', 'Target', 'Pred', 'Percison', 'Recall', 'F1', 'Accuracy'])
    for i in range(num_cls):
        cts_mask = ctsN == i
        clses_mask = clsesN == i
        tp = np.sum(cts_mask * clses_mask)
        tn = np.sum(~cts_mask * ~clses_mask)
        fp = np.sum(~cts_mask * clses_mask)
        fn = np.sum(cts_mask * ~clses_mask)
        cls_name = cls2name(i) if cls2name is not None else i
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        recl = tp / (tp + fn) if (tp + fn) > 0 else 0
        acc = (tp + tn) / len(cts_mask) if len(cts_mask) > 0 else 0
        f1 = 2 / (1 / prec + 1 / recl) if prec > 0 and recl > 0 else 0
        data = pd.concat([data, pd.DataFrame({
            'Class': cls_name, 'Target': np.sum(cts_mask), 'Pred': np.sum(clses_mask),
            'Percison': prec, 'Recall': recl, 'F1': f1, 'Accuracy': acc
        }, index=[0])])
    if num_cls > 1:
        tars = np.array(data['Target'])
        tar = np.sum(tars)
        preds = np.array(data['Pred'])
        pred = np.sum(preds)
        data = pd.concat([data, pd.DataFrame({
            'Class': 'Total', 'Target': tar, 'Pred': pred,
            'Percison': np.sum(np.array(data['Percison']) * tars) / tar,
            'Recall': np.sum(np.array(data['Recall']) * preds) / pred,
            'F1': np.average(data['F1']), 'Accuracy': np.average(data['Accuracy'])
        }, index=[0])])
    return data


def calc_map(model, loader, num_cls=80, cls2name=None, **kwargs):
    assert isinstance(model, IDetectionInfer), 'model err'
    if isinstance(model, ModelFrame):
        ploader = Prefetcher(loader, model.device) if model.device.index is not None else loader
        model.eval()
    else:
        ploader = loader
    # 准备训练
    confs = []
    pred_clses = []
    neg_mask = []
    pos_mask = []
    gt_clses = []
    num_batch = len(loader)

    for i, (imgs, gtss) in enumerate(ploader):
        if i % 10 == 0 or i == num_batch - 1:
            print('Testing %5d' % (i + 1) + ' / %5d' % num_batch)
        # 检测
        boxess = model.imgs2boxss(imgs, **kwargs)
        # 匹配
        for boxes, gts in zip(boxess, gtss):
            # 处理gt
            if len(gts) > 0:
                box_fmt_gts = boxes2fmt(gts)
                gtsN = boxes2boxesN(gts, box_fmt=box_fmt_gts)
                gts_clses_i, gts_diff_i = boxes2clsesN_diffsN(gts)
                gt_clses.append(gts_clses_i[~gts_diff_i])
            else:
                gtsN, box_fmt_gts, gts_clses_i, gts_diff_i = None, None, None, None

            # 处理检测结果
            if len(boxes) > 0:
                box_fmt_bxs = boxes2fmt(boxes)
                boxsN = boxes2boxesN(boxes, box_fmt=box_fmt_bxs)
                pred_clses_i, confs_i = boxes2clsesN_confsN(boxes)
                order = np.argsort(-confs_i)
                boxsN, pred_clses_i, confs_i = boxsN[order], pred_clses_i[order], confs_i[order]
                # 与gt匹配
                pos_mask_i = np.zeros(shape=len(boxes), dtype=np.int32)
                neg_mask_i = np.zeros(shape=len(boxes), dtype=np.int32)
                if len(gts) == 0:
                    neg_mask_i = neg_mask_i + 1
                else:
                    iou_mat = clac_iou_mat_box(boxesN1=boxsN, box_fmt1=box_fmt_bxs,
                                               boxesN2=gtsN, box_fmt2=box_fmt_gts, iou_type=IOU_TYPE.IOU)
                    for k in range(len(boxes)):
                        iou_mat[k, ~(pred_clses_i[k] == gts_clses_i)] = 0  # 不同分类之间不能匹配比较
                        ind_gt = np.argmax(iou_mat[k, :])
                        if iou_mat[k, ind_gt] > 0.5:
                            if not gts_diff_i[ind_gt]:
                                pos_mask_i[k] = 1
                            iou_mat[:, ind_gt] = 0  # 防止重复匹配
                        else:
                            neg_mask_i[k] = 1
                # 添加
                confs.append(confs_i)
                pos_mask.append(pos_mask_i)
                neg_mask.append(neg_mask_i)
                pred_clses.append(pred_clses_i)

    confs = np.concatenate(confs, axis=0)
    pos_mask = np.concatenate(pos_mask, axis=0)
    neg_mask = np.concatenate(neg_mask, axis=0)
    pred_clses = np.concatenate(pred_clses, axis=0)
    gt_clses = np.concatenate(gt_clses, axis=0)
    aps = ap_per_class(pos_mask=pos_mask, neg_mask=neg_mask, confs=confs, pred_cls=pred_clses, gt_cls=gt_clses,
                       num_cls=num_cls)
    # 统计结果
    data = pd.DataFrame(columns=['Class', 'Target', 'Pred', 'Pos', 'Neg', 'Ign', 'AP'])
    for i in range(num_cls):
        mask_pred_clses = pred_clses == i
        num_pred = np.sum(mask_pred_clses)
        num_pos = np.sum(pos_mask[mask_pred_clses] == 1)
        num_neg = np.sum(neg_mask[mask_pred_clses] == 1)
        num_ign = np.sum((neg_mask[mask_pred_clses] == 0) * (pos_mask[mask_pred_clses] == 0))
        num_gt = np.sum(gt_clses == i)
        cls_name = cls2name(i) if cls2name is not None else str(i)
        data = pd.concat([data, pd.DataFrame({
            'Class': cls_name, 'Target': num_gt, 'Pred': num_pred,
            'Pos': num_pos, 'Neg': num_neg, 'Ign': num_ign, 'AP': aps[i]
        }, index=[0])])
    if num_cls > 1:
        data = pd.concat([data, pd.DataFrame({
            'Class': 'Total', 'Target': np.sum(data['Target']), 'Pred': np.sum(data['Pred']),
            'Pos': np.sum(data['Pos']), 'Neg': np.sum(data['Neg']), 'Ign': np.sum(data['Ign']),
            'AP': np.average(data['AP'])
        }, index=[0])])
    return data


# 分类别计算AUC
def auc_per_class(oclses, cts):
    num_samp, num_cls = oclses.shape
    assert num_samp == len(cts), 'len err'
    aucs = np.zeros(num_cls)
    for cls_ind in range(num_cls):
        cls_mask = cts == cls_ind
        num_pos = (cls_mask).sum()
        num_neg = num_samp - num_pos
        if num_pos == 0 or num_neg == 0:
            aucs[cls_ind] = 0
        else:
            # 置信度降序
            confs_cls = oclses[:, cls_ind]
            order = np.argsort(-confs_cls)
            cls_mask = cls_mask[order]
            # 计算曲线
            tpr_curve = (cls_mask).cumsum() / num_pos
            fpr_curve = (~cls_mask).cumsum() / num_neg
            # 计算面积
            fpr_curve = np.concatenate(([0.0], fpr_curve))
            fpr_dt = fpr_curve[1:] - fpr_curve[:-1]
            aucs[cls_ind] = np.sum(fpr_dt * tpr_curve)
    aucs = np.array(aucs)
    return aucs


# 分类别计算AP
def ap_per_class(pos_mask, neg_mask, confs, pred_cls, gt_cls, num_cls=20):
    order = np.argsort(-confs)
    pos_mask, neg_mask, confs, pred_cls = pos_mask[order], neg_mask[order], confs[order], pred_cls[order]
    aps = np.zeros(num_cls)
    for cls_ind in range(num_cls):
        mask_pred_pos = pred_cls == cls_ind
        num_gt = (gt_cls == cls_ind).sum()
        num_pred = mask_pred_pos.sum()
        if num_pred == 0 or num_gt == 0:
            aps[cls_ind] = 0
        else:
            fp_nums = (neg_mask[mask_pred_pos]).cumsum()  # 累加和列表
            tp_nums = (pos_mask[mask_pred_pos]).cumsum()
            # 计算曲线
            recall_curve = tp_nums / (num_gt + 1e-16)
            precision_curve = tp_nums / (tp_nums + fp_nums + 1e-16)
            # 计算面积
            recall_curve = np.concatenate(([0.0], recall_curve, [1.0]))
            precision_curve = np.concatenate(([1.0], precision_curve, [0.0]))
            for i in range(precision_curve.size - 1, 0, -1):
                precision_curve[i - 1] = np.maximum(precision_curve[i - 1], precision_curve[i])
            aps[cls_ind] = np.sum((recall_curve[1:] - recall_curve[:-1]) * precision_curve[1:])

    aps = np.array(aps)
    return aps


if __name__ == '__main__':
    a = sorted(np.random.rand(10))
    b = sorted(np.random.rand(10))[::-1]
