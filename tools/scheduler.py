import math
from abc import abstractmethod, ABCMeta
import random
from collections import Iterable


class IMScheduler(metaclass=ABCMeta):

    @abstractmethod
    def __getitem__(self, epoch_ind):
        pass

    @staticmethod
    def Const(img_size):
        return IMSConst(img_size=img_size)

    @staticmethod
    def Rand(min_size, max_size, devisor=32, keep_ratio=True, keep_epoch=1, max_first=True):
        return IMSRand(min_size=min_size, max_size=max_size, devisor=devisor, keep_ratio=keep_ratio,
                       keep_epoch=keep_epoch, max_first=max_first)


class IMSConst(IMScheduler):
    def __init__(self, img_size):
        self.W, self.H = img_size if isinstance(img_size, Iterable) else (img_size, img_size)

    def __getitem__(self, epoch_ind):
        return (self.W, self.H)


class IMSRand(IMScheduler):
    def __init__(self, min_size, max_size, devisor=32, keep_ratio=True, keep_epoch=1, max_first=True):
        max_W, max_H = max_size if isinstance(max_size, Iterable) else (max_size, max_size)
        min_W, min_H = min_size if isinstance(min_size, Iterable) else (min_size, min_size)
        self.min_size = min_size
        self.devisor = devisor
        self.max_w, self.max_h = int(math.floor(max_W / devisor)), int(math.floor(max_H / devisor))
        self.min_w, self.min_h = int(math.ceil(min_W / devisor)), int(math.ceil(min_H / devisor))
        self.keep_ratio = keep_ratio
        self.keep_epoch = keep_epoch
        self.max_first = max_first
        self.last_size = self._rand_size()
        self.kpd = 0

    def _rand_size(self):
        if self.max_first:
            self.max_first = False
            w = self.max_w
        else:
            w = random.randint(self.min_w, self.max_w)
        if self.keep_ratio:
            h = int(1.0 * (w - self.min_w) / (self.max_w - self.min_w) * (self.max_h - self.min_h) + self.min_h)
        else:
            h = random.randint(self.min_h, self.max_h) * self.devisor
        return (w * self.devisor, h * self.devisor)

    def __getitem__(self, epoch_ind):
        if self.kpd == self.keep_epoch:
            self.last_size = self._rand_size()
            self.kpd = 1
        else:
            self.kpd = self.kpd + 1
        return self.last_size


if __name__ == '__main__':
    ims = IMSRand(max_size=(800, 800), min_size=(480, 480), devisor=32, keep_epoch=2)
    ls = [ims[i] for i in range(10)]
    print(ls)


class LRScheduler(metaclass=ABCMeta):
    def __init__(self, total_epoch=10, iter_per_epoch=1):
        self.total_epoch = total_epoch
        self.iter_per_epoch = iter_per_epoch
        self.init()

    @property
    def iter_per_epoch(self):
        return self._iter_per_epoch

    @iter_per_epoch.setter
    def iter_per_epoch(self, iter_per_epoch):
        self._iter_per_epoch = iter_per_epoch
        self.total_iter = self.total_epoch * iter_per_epoch

    @property
    def lr_list(self):
        lrs = []
        for i in range(self.total_iter):
            lrs.append(self.__getitem__(i))
        return lrs

    @abstractmethod
    def __getitem__(self, item):
        pass

    def __imul__(self, other):
        return self

    def init(self):
        self.ptr = 0
        return self

    def set_epoch(self, epoch_ind):
        assert epoch_ind < self.total_epoch and epoch_ind >= 0, 'epoch_ind err'
        self.ptr = epoch_ind * self.iter_per_epoch
        return self

    def step(self):
        if self.ptr == self.total_iter - 1:
            self.ptr = 0
        else:
            self.ptr = self.ptr + 1
            return None

    def modify_optimizer(self, optimizer):
        lr = self.__getitem__(self.ptr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return None

    @staticmethod
    def WARM_COS(lr=0.1, warm_epoch=50, total_epoch=100):
        lrs = LRSCompose(
            LRSQuad(lr=1e-8, lr_end=lr, total_epoch=warm_epoch),
            LRSCos(lr=lr, lr_end=0, total_epoch=total_epoch - warm_epoch)
        )
        return lrs

    @staticmethod
    def WARM_COS_CONST(lr=0.1, warm_epoch=50, const_epoch=30, total_epoch=100, lr_ratio=0.2):
        lr_end = lr_ratio * lr
        lrs = LRSCompose(
            LRSQuad(lr=1e-8, lr_end=lr, total_epoch=warm_epoch),
            LRSCos(lr=lr, lr_end=lr_end, total_epoch=total_epoch - warm_epoch - const_epoch),
            LRSConst(lr=lr_end, total_epoch=const_epoch)
        )
        return lrs

    @staticmethod
    def WARM_STEP(lr=0.1, warm_epoch=50, milestones=(0, 0), gamma=0.1, total_epoch=100):
        lrs = LRSCompose(
            LRSQuad(lr=1e-8, lr_end=lr, total_epoch=warm_epoch),
            LRSMultiStep(lr=lr, milestones=milestones, gamma=gamma, total_epoch=total_epoch - warm_epoch),
        )
        return lrs

    @staticmethod
    def MultiStep(lr=0.1, milestones=(0, 1), gamma=0.1, total_epoch=10, iter_per_epoch=1):
        return LRSMultiStep(lr=lr, milestones=milestones, gamma=gamma, total_epoch=total_epoch,
                            iter_per_epoch=iter_per_epoch)

    @staticmethod
    def Const(lr=0.1, total_epoch=10, iter_per_epoch=1):
        return LRSConst(lr=lr, total_epoch=total_epoch, iter_per_epoch=iter_per_epoch)


class LRSCompose(LRScheduler):
    def __init__(self, *schedulers, iter_per_epoch=1):
        total_epoch = 0
        for scheduler in schedulers:
            total_epoch += scheduler.total_epoch
        self.schedulers = schedulers
        super(LRSCompose, self).__init__(total_epoch=total_epoch, iter_per_epoch=iter_per_epoch)

    def __imul__(self, other):
        for scheduler in self.schedulers:
            scheduler.__imul__(other)
        return self

    def __getitem__(self, iter_ind):
        for i in range(len(self.milestones) - 1, -1, -1):
            if iter_ind >= self.milestones[i]:
                return self.schedulers[i].__getitem__(iter_ind - self.milestones[i])
        raise Exception('milestones err')

    @LRScheduler.iter_per_epoch.setter
    def iter_per_epoch(self, iter_per_epoch):
        super(LRSCompose, LRSCompose).iter_per_epoch.__set__(self, iter_per_epoch)
        total_iter = 0
        milestones = []
        for scheduler in self.schedulers:
            scheduler.iter_per_epoch = iter_per_epoch
            milestones.append(total_iter)
            total_iter += scheduler.total_iter
        self.milestones = milestones
        self.total_iter = total_iter


class LRSMultiStep(LRScheduler):
    def __init__(self, lr=0.1, milestones=(0, 1), gamma=0.1, total_epoch=10, iter_per_epoch=1):
        if isinstance(milestones, int):
            milestones = [milestones]
        self.milestones_epoch = list(milestones)
        self.gamma = gamma
        self.lr = lr
        super(LRSMultiStep, self).__init__(total_epoch=total_epoch, iter_per_epoch=iter_per_epoch)

    def __imul__(self, other):
        self.lr *= other
        return self

    def __getitem__(self, iter_ind):
        lr = self.lr
        for milestone in self.milestones_iter:
            lr *= self.gamma if iter_ind >= milestone else 1.0
        return lr

    @LRScheduler.iter_per_epoch.setter
    def iter_per_epoch(self, iter_per_epoch):
        super(LRSMultiStep, LRSMultiStep).iter_per_epoch.__set__(self, iter_per_epoch)
        milestones_iter = []
        for i in range(len(self.milestones_epoch)):
            milestones_iter.append(self.milestones_epoch[i] * iter_per_epoch)
        self.milestones_iter = milestones_iter


class LRSQuad(LRScheduler):
    def __init__(self, lr=0.1, lr_end=1e-8, total_epoch=10, iter_per_epoch=1):
        super(LRSQuad, self).__init__(total_epoch=total_epoch, iter_per_epoch=iter_per_epoch)
        self.lr = lr
        self.lr_end = lr_end

    def __getitem__(self, iter_ind):
        alpha = (iter_ind / self.total_iter) ** 2
        lr = (1 - alpha) * self.lr + alpha * self.lr_end
        return lr

    def __imul__(self, other):
        self.lr *= other
        self.lr_end *= other
        return self


class LRSExponential(LRScheduler):
    def __init__(self, lr=0.1, lr_end=1e-8, total_epoch=10, iter_per_epoch=1):
        super(LRSExponential, self).__init__(total_epoch=total_epoch, iter_per_epoch=iter_per_epoch)
        self.lr_log = math.log(lr)
        self.lr_end_log = math.log(lr_end)

    def __getitem__(self, iter_ind):
        alpha = iter_ind / self.total_iter
        lr_log = (1 - alpha) * self.lr_log + alpha * self.lr_end_log
        return math.exp(lr_log)

    def __imul__(self, other):
        self.lr_log += math.log(other)
        self.lr_end_log += math.log(other)
        return self


class LRSCos(LRScheduler):
    def __init__(self, lr=0.1, lr_end=0.0, total_epoch=10, iter_per_epoch=1):
        super(LRSCos, self).__init__(total_epoch=total_epoch, iter_per_epoch=iter_per_epoch)
        self.lr = lr
        self.lr_end = lr_end

    def __getitem__(self, iter_ind):
        alpha = iter_ind / self.total_iter
        lr = self.lr_end + (self.lr - self.lr_end) * 0.5 * (1.0 + math.cos(math.pi * alpha))
        return lr

    def __imul__(self, other):
        self.lr *= other
        self.lr_end *= other
        return self


class LRSConst(LRScheduler):
    def __init__(self, lr=0.1, total_epoch=10, iter_per_epoch=1):
        super(LRSConst, self).__init__(total_epoch=total_epoch, iter_per_epoch=iter_per_epoch)
        self.lr = lr

    def __getitem__(self, iter_ind):
        return self.lr

    def __imul__(self, other):
        self.lr *= other
        return self


class LRSLinear(LRScheduler):
    def __init__(self, lr=0.1, lr_end=0.1, total_epoch=10, iter_per_epoch=1):
        super(LRSLinear, self).__init__(total_epoch=total_epoch, iter_per_epoch=iter_per_epoch)
        self.lr = lr
        self.lr_end = lr_end

    def __getitem__(self, iter_ind):
        alpha = iter_ind / self.total_iter
        lr = (1 - alpha) * self.lr + alpha * self.lr_end
        return lr

    def __imul__(self, other):
        self.lr *= other
        self.lr_end *= other
        return self

# if __name__ == '__main__':
#     from visual import *
#
#     scheduler = LRScheduler.WARM_COS(lr=0.0025, warm_epoch=1, total_epoch=300)
#     # scheduler = LRScheduler.WARM_STEP(lr=0.1, warm_epoch=50, milestones=(20, 30), gamma=0.1, total_epoch=100)
#     scheduler.iter_per_epoch = 100
#     plt.plot(scheduler.lr_list)
