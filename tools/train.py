import torch
from tools.metric import Prefetcher
import pandas as pd
import numpy as np
from tools.scheduler import LRScheduler
import os
from functools import partial
from utils.frame import ModelFrame
import time
from utils.logger import print

# <editor-fold desc='检查'>
# 检查梯度
from utils.logger import rmv_handler, logfile


def check_grad(model, loader, accu_step=1):
    model.eval()
    print('Checking Grad')
    loader_iter = iter(loader)
    for i in range(accu_step):
        (imgs, labels) = next(loader_iter)
        loss = model.imgs_labels2loss(imgs, labels)
        loss.backward()
    for name, para in model.named_parameters():
        print('Grad %10.5f' % para.grad.norm().item() + ' --- ' + name)
    return None


# 检查参数
def check_para(model):
    print('Checking Para')
    for name, para in model.named_parameters():
        if torch.any(torch.isnan(para)):
            print('nan occur in models')
            para.data = torch.where(torch.isnan(para), torch.full_like(para, 0.1), para)
        if torch.any(torch.isinf(para)):
            print('inf occur in models')
            para.data = torch.where(torch.isinf(para), torch.full_like(para, 0.1), para)
        max = torch.max(para).item()
        min = torch.min(para).item()
        print('Range [ %10.5f' % min + ' , ' + '%10.5f' % max + ']  --- ' + name)
    return None


# </editor-fold>


# <editor-fold desc='虚拟loader'>

class TargetLoaderSingle():
    def __init__(self, imgs, target, step):
        super().__init__()
        self.imgs = imgs
        self.target = target
        self.step = step
        self.ptr = 0
        self.batch_size = len(imgs)

    def __len__(self):
        return self.step

    def __iter__(self):
        self.ptr = 0
        return self

    def __next__(self):
        if self.ptr >= self.step:
            raise StopIteration
        else:
            self.ptr = self.ptr + 1
            return self.imgs, self.target


class TargetLoader():
    def __init__(self, labels2target, loader):
        super().__init__()
        self.labels2target = labels2target
        self.loader = loader

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def batch_size(self):
        return self.loader.batch_size

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        return self

    def __next__(self):
        imgs, labels = next(self.loader_iter)
        target = self.labels2target(labels)
        return imgs, target


# </editor-fold>
def process_time(last_time, data_time, fwd_time, bkwd_time, opt_time):
    times = np.array([last_time, data_time, fwd_time, bkwd_time, opt_time])
    times = times[1:] - times[:-1]
    times = [opt_time - last_time] + list(times)
    names = ['Time', 'data', 'fwd', 'bkwd', 'opt']
    return times, names


def train_epoch(model, loader, optimizer, accu_step=1, scheduler=None, record=None, **kwargs):
    record = pd.DataFrame() if record is None else record
    model.train()
    last_time = time.time()
    num_batch = len(loader)
    for iter_ind, (imgs, target) in enumerate(loader):
        data_time = time.time()
        # loss
        loss = model.imgs_target2loss(imgs, target, **kwargs)
        loss, losses, lnames = ModelFrame.process_loss(loss)
        fwd_time = time.time()
        assert not torch.isnan(loss), 'nan occur in loss'
        (loss / accu_step).backward()
        bkwd_time = time.time()
        # lr
        if scheduler is not None:
            scheduler.modify_optimizer(optimizer)
            scheduler.step()
        # accu
        if (iter_ind + 1) % accu_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        opt_time = time.time()
        # record
        lr = optimizer.param_groups[0]['lr']
        times, tnames = process_time(last_time, data_time, fwd_time, bkwd_time, opt_time)
        columns = ['Iter', 'Lr'] + lnames + tnames
        vals = [iter_ind + 1, lr] + losses + times
        row = pd.DataFrame([dict(zip(columns, vals))], columns=columns)
        record = pd.concat([record, row])
        # show
        if iter_ind % 50 == 0 or iter_ind + 1 == num_batch:
            vals_aver = np.average(record.iloc[-50:, :], axis=0)
            div = 2 + len(lnames)
            print(
                columns[0] + ' %-5d' % (iter_ind + 1) +
                ' | ' + columns[1] + ' %-8.7f ' % vals_aver[1] +
                ' | ' + ''.join([columns[k] + ' %-8.5f ' % vals_aver[k] for k in range(2, div)]) +
                ' | ' + ''.join([columns[k] + ' %-6.5f ' % vals_aver[k] for k in range(div, len(columns))]) + '|'
            )
        last_time = opt_time
    return record


def train_single(model, imgs, labels, optimizer, scheduler, accu_step=1, record=None, **kwargs):
    # scheduler
    if isinstance(scheduler, int):
        lr = optimizer.param_groups[0]['lr']
        scheduler = LRScheduler.Const(lr=lr, total_epoch=scheduler)
    total_epoch = scheduler.total_epoch
    scheduler.iter_per_epoch = 1
    # loader
    target = model.labels2target(labels)
    loader = TargetLoaderSingle(target=target, imgs=imgs, step=total_epoch)
    loader = Prefetcher(loader, model.device) if model.device.index is not None else loader

    record = train_epoch(model, loader, optimizer, accu_step=accu_step, scheduler=scheduler,
                         record=record, **kwargs)
    return record


def sec2hour_min_sec(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def get_file_names(file_name):
    if file_name is None or len(file_name) == 0:
        return '', '', '', '', ''
    file_name_pure, extend = os.path.splitext(file_name)
    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    log_name = file_name_pure + '.log'
    record_name = file_name_pure + '.xlsx'
    wei_name = file_name_pure + '.pth'
    wei_best_name = file_name_pure + '_best.pth'
    opt_name = file_name_pure + '.opt'
    return wei_name, wei_best_name, opt_name, record_name, log_name


def train(model, loader, optimizer, lrscheduler, accu_step=1, save_step=10, file_name='',
          test_step=10, metric=None, best_pfrmce=0, imscheduler=None, new_proc=True, **kwargs):
    # train
    if new_proc:
        print('Start new train process')
    else:
        print('Continue train process')

    # file_name
    wei_name, wei_best_name, opt_name, record_name, log_name = get_file_names(file_name)

    # loader
    loader_tar = TargetLoader(labels2target=model.labels2target, loader=loader)
    loader_tar = Prefetcher(loader_tar, model.device) if model.device.index is not None else loader_tar

    # optimizer
    if not new_proc and os.path.isfile(opt_name):
        print('Load optimizer from ' + opt_name)
        optimizer.load_state_dict(torch.load(opt_name))

    # scheduler
    lrscheduler.iter_per_epoch = len(loader)
    lrscheduler.modify_optimizer(optimizer)
    total_epoch = lrscheduler.total_epoch

    # test
    with_test = test_step > 0 and metric is not None
    with_save = save_step > 0 and os.path.exists(os.path.dirname(wei_name))

    # map
    if with_test and best_pfrmce <= 0:
        print('Init Test')
        best_pfrmce = metric(model)

    # log
    if with_save:
        handler = logfile(log_name, new_log=new_proc)
    else:
        handler = None

    # record
    if not new_proc and os.path.isfile(record_name):
        record = pd.read_excel(record_name)
    else:
        record = None

    # train
    start_time = time.time()
    for epoch_ind in range(total_epoch):
        if imscheduler is not None:
            img_size = imscheduler[epoch_ind]
            loader.img_size = img_size
            model.img_size = img_size
            if metric is not None:
                metric.img_size = img_size

        # show
        cur_time = time.time()
        eta = 0 if epoch_ind == 0 else (cur_time - start_time) / epoch_ind * (total_epoch - epoch_ind)
        epoch_msg = 'Epoch %d' % (epoch_ind + 1) + '  Length %d' % len(loader) + \
                    '  Batch %d' % loader.batch_size + '[%d]' % accu_step + \
                    '  ImgSize ' + str(loader.img_size) + \
                    '  ETA ' + sec2hour_min_sec(eta)
        print(epoch_msg)

        if isinstance(loader.sampler, torch.utils.data.distributed.DistributedSampler):
            loader.sampler.set_epoch(epoch_ind)

        # train_epoch
        record = train_epoch(model, loader_tar, optimizer, accu_step=accu_step,
                             scheduler=lrscheduler, record=record)

        # test
        if with_test and (epoch_ind + 1) % test_step == 0:
            print('Step Test')
            pfrmce = metric(model)
            if best_pfrmce < pfrmce:
                best_pfrmce = max(best_pfrmce, pfrmce)
                print('Save best ' + wei_best_name)
                model.save(wei_best_name)

        # save
        if with_save and (epoch_ind + 1) % save_step == 0:
            print('Save at ' + wei_name)
            model.save(wei_name)
            record.to_excel(record_name, index=False)
            # torch.save(optimizer.state_dict(), opt_name)

    if with_save:
        rmv_handler(handler)

    return record


class OPTIMIZERTP():
    @staticmethod
    def SGD(momentum=0.9, weight_decay=5e-5):
        return partial(torch.optim.SGD, momentum=momentum, weight_decay=weight_decay, lr=0.001)

    @staticmethod
    def ADAM():
        return None



