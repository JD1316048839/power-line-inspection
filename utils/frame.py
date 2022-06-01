from collections import Iterable
from abc import ABCMeta, abstractmethod
from functools import partial
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.device import select_device, pack_module, PACK
from utils.loadsd import load_fmt, refine_chans, MATCH_TYPE
from utils.label import *
from collections import OrderedDict
from utils.logger import print


class TASK_TYPE:
    DET = 'det'
    CLS = 'cls'
    AUTO = 'auto'


class LoaderFrame(torch.utils.data.DataLoader):
    def __init__(self, ds, shuffle=False, num_workers=0, batch_size=1, pin_memory=False, distribute=False,
                 aug_seqTp=None, img_size=(416, 416), num_cls=0, cls2name=None):
        sampler = torch.utils.data.distributed.DistributedSampler(ds) if distribute else None
        shuffle = shuffle and not distribute
        super(LoaderFrame, self).__init__(dataset=ds, shuffle=shuffle, sampler=sampler, num_workers=num_workers,
                                          batch_size=batch_size, pin_memory=pin_memory)
        self.aug_seqTp = aug_seqTp
        self.img_size = img_size
        self.num_cls = num_cls
        self.cls2name = cls2name

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size if isinstance(img_size, Iterable) else (img_size, img_size)
        self.aug_seq = None if self.aug_seqTp is None else self.aug_seqTp(self._img_size)

    @property
    def aug_seq(self):
        return self._aug_seq

    @aug_seq.setter
    def aug_seq(self, aug_seq):
        self._aug_seq = aug_seq
        self.collate_fn = partial(LoaderFrame.collate_fn, aug_seq=self.aug_seq)

    @staticmethod
    def collate_fn(batch, aug_seq):
        imgs = []
        labels = []
        for img, ann in batch:
            labels.append(ann)
            imgs.append(img)
        if aug_seq is not None:
            imgs, labels = aug_seq(imgs, labels)
        return imgs, labels

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.__getitem__([item])
        elif isinstance(item, list) or isinstance(item, tuple):
            batch = [self.dataset.__getitem__(index) for index in item]
            imgs, labels = self.collate_fn(batch)
            return imgs, labels
        else:
            raise Exception('index err')


class DataSourceFrame():
    def __init__(self, pth, img_size=(224, 224), sets=('train', 'test'), num_cls=0):
        self.img_size = img_size
        self.pth = pth
        self.sets = sets
        self.num_cls = num_cls

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size if isinstance(img_size, Iterable) else (img_size, img_size)

    def loader(self, set_name, batch_size=8, pin_memory=False, num_workers=0,
               aug_seqTp=None, img_size=None, shuffle=True, distribute=False, **kwargs):
        if isinstance(set_name, str):
            assert set_name in self.sets or None in self.sets, 'err set ' + str(set_name)
            set_name = self.dataset(set_name=set_name, **kwargs)
        img_size = self._img_size if img_size is None else img_size
        loader = LoaderFrame(
            set_name,
            shuffle=shuffle,
            aug_seqTp=aug_seqTp,
            num_workers=num_workers,
            batch_size=batch_size,
            img_size=img_size,
            pin_memory=pin_memory,
            distribute=distribute,
            num_cls=self.num_cls,
            cls2name=self.cls2name,
        )
        return loader

    def dataset(self, set_name, **kwargs):
        raise NotImplementedError

    def cls2name(self, cls):
        return NotImplementedError

    def name2cls(self, name):
        return NotImplementedError


class IClassificationInfer(metaclass=ABCMeta):

    # 测试
    @abstractmethod
    def imgs2clses(self, imgs, cls2name=None, **kwargs):
        pass


class IClassification(IClassificationInfer):

    # 标签转化
    @abstractmethod
    def cts2tarc(self, cts, **kwargs):
        pass

    # 计算损失
    @abstractmethod
    def imgs_tarc2loss(self, imgs, target, **kwargs):
        pass


class IDetectionInfer(metaclass=ABCMeta):
    # 可视化
    @abstractmethod
    def imgs2boxss(self, imgs, cls2name=None, **kwargs):
        pass


class IDetection(IDetectionInfer):
    # 标签转化
    @abstractmethod
    def gtss2tard(self, gtss, **kwargs):
        pass

    # 计算损失
    @abstractmethod
    def imgs_tard2loss(self, imgs, target, **kwargs):
        pass


class ModelInferFrame(metaclass=ABCMeta):
    def __init__(self):
        if isinstance(self, IDetectionInfer):
            self.task_type = TASK_TYPE.DET
        elif isinstance(self, IClassificationInfer):
            self.task_type = TASK_TYPE.CLS
        else:
            raise Exception('At least implement one interface')

    @property
    @abstractmethod
    def img_size(self):
        pass

    @img_size.setter
    @abstractmethod
    def img_size(self, img_size):
        pass

    @property
    def task_type(self):
        return self._task_type

    @task_type.setter
    def task_type(self, task_type):
        self._task_type = task_type
        if task_type == TASK_TYPE.DET:
            assert isinstance(self, IDetectionInfer), 'err task type'
            print('Detection Mode')
            self.imgs2labels = self.imgs2boxss
        elif task_type == TASK_TYPE.CLS:
            assert isinstance(self, IClassificationInfer), 'err task type'
            print('Classification Mode')
            self.imgs2labels = self.imgs2clses  # 可视化
        else:
            raise Exception('err task type')
        self.__call__ = self.imgs2labels


class ModelFrame(nn.Module, ModelInferFrame):
    def __init__(self, device=None, pack=PACK.AUTO, **modules):
        super(ModelFrame, self).__init__()
        super(nn.Module, self).__init__()
        device_ids = select_device(device, min_thres=0.01, one_thres=0.5)
        device = torch.device('cpu' if device_ids[0] is None else 'cuda:' + str(device_ids[0]))
        self.device = device
        self.pkd_modules = {}
        for name, module in modules.items():
            self.__setattr__(name, module)
            module.to(device)
            self.pkd_modules[name] = pack_module(module, device_ids, pack=pack)

    # 保存权重
    def save(self, file_name):
        file_name = file_name if str.endswith(file_name, '.pth') else file_name + '.pth'
        torch.save(self.state_dict(), file_name)
        return None

    # 读取权重
    def load(self, file_name, transfer=False):
        if isinstance(file_name, str):
            file_name = file_name if str.endswith(file_name, '.pth') else file_name + '.pth'
            print('Load weight ' + file_name)
            sd = torch.load(file_name, map_location=self.device)
        elif isinstance(file_name, OrderedDict):
            sd = file_name
        else:
            raise Exception('dict err')
        if load_fmt(self, sd_ori=sd, match_type=MATCH_TYPE.SIZE | MATCH_TYPE.FULL_NAME, only_fullmatch=True):
            refine_chans(self)
            self.to(self.device)
            return None
        print('Struct changed, Try to match by others')
        if load_fmt(self, sd_ori=sd, match_type=MATCH_TYPE.SIZE | MATCH_TYPE.LAST_NAME, only_fullmatch=not transfer):
            refine_chans(self)
            self.to(self.device)
            return None
        print('Tolerates imperfect matches')
        load_fmt(self, sd_ori=sd, match_type=MATCH_TYPE.SIZE | MATCH_TYPE.FULL_NAME, only_fullmatch=False)
        self.to(self.device)
        return None

    @abstractmethod
    def export_onnx(self, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def export_onnx_trt(self, **kwargs):
        raise NotImplementedError()

    @ModelInferFrame.task_type.setter
    def task_type(self, task_type):
        super(ModelFrame, ModelFrame).task_type.__set__(self, task_type)
        if task_type == TASK_TYPE.DET:
            self.labels2target = self.gtss2tard  # 标签转化
            self.imgs_target2loss = self.imgs_tard2loss  # 训练
        elif task_type == TASK_TYPE.CLS:
            self.labels2target = self.cts2tarc  # 标签转化
            self.imgs_target2loss = self.imgs_tarc2loss  # 训练
        else:
            raise Exception('err task type')

    # 快速转化loss
    def imgs_labels2loss(self, imgs, labels, show=False):
        target = self.labels2target(labels)
        loss = self.imgs_target2loss(imgs, target)
        loss, losses, names = ModelFrame.process_loss(loss)
        if show:
            print(''.join([n + ' %-10.5f  ' % l for l, n in zip(losses, names)]))
        return loss

    @staticmethod
    def process_loss(loss):
        if isinstance(loss, dict):
            losses, names = list(loss.values()), list(loss.keys())
            for l, n in zip(losses, names):
                assert not torch.isnan(l), 'nan in loss ' + n
                assert not torch.isinf(l), 'inf in loss ' + n
            loss = torch.sum(torch.stack(losses))
            losses = [l.item() for l in losses]
            losses.insert(0, loss.item())
            names.insert(0, 'Loss')
        elif isinstance(loss, torch.Tensor):
            assert not torch.isnan(loss), 'nan in loss'
            assert not torch.isinf(loss), 'inf in loss'
            losses = [loss.item()]
            names = ['Loss']
        else:
            raise Exception('err loss')
        return loss, losses, names


class OneStageModelFrame(ModelFrame):
    def __init__(self, backbone, device=None, pack=None):
        super(OneStageModelFrame, self).__init__(backbone=backbone, device=device, pack=pack)

    def export_onnx(self, onnx_pth, batch_size=1):
        return True

    def export_onnx_trt(self, onnx_pth, trt_pth, batch_size=1):
        return True


class OneStageClassifier(OneStageModelFrame, IClassification):
    def __init__(self, backbone, device=None, pack=None, img_size=(224, 224), num_cls=10):
        super(OneStageClassifier, self).__init__(backbone=backbone, device=device, pack=pack)
        self.img_size = img_size
        self.num_cls = num_cls

    @property
    def img_size(self):
        return self._img_size

    @img_size.setter
    def img_size(self, img_size):
        self._img_size = img_size

    def cts2tarc(self, cts, **kwargs):
        clsesN = clses2clsesN(cts)
        target = clsesN2oclsesN(clsesN, num_cls=self.num_cls)
        return target

    def imgs_tarc2loss(self, imgs, target, **kwargs):
        imgs = imgs.to(self.device)
        pred = self.pkd_modules['backbone'](imgs)
        target = torch.as_tensor(target).to(pred.device, non_blocking=True)
        loss = F.cross_entropy(pred, target, reduction='mean')
        return loss

    @torch.no_grad()
    def imgs2clses(self, imgs, cls2name=None, **kwargs):
        self.eval()
        imgs = imgs.to(self.device)
        pred = self.pkd_modules['backbone'](imgs)
        pred = torch.softmax(pred, dim=-1)
        clses = oclsesT2clses(oclsesT=pred, cls2name=cls2name)
        return clses

if __name__ == '__main__':
    model = ModelFrame()
