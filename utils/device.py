import pynvml
import torch
import torchvision
import numpy as np
import sys
import torch.nn as nn
import time


# 得到GPU占用
def cuda_usage():
    pynvml.nvmlInit()
    num_cuda = pynvml.nvmlDeviceGetCount()
    usages = []
    for ind in range(num_cuda):
        handle = pynvml.nvmlDeviceGetHandleByIndex(ind)  # 0表示第一块显卡
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        usage = meminfo.used / meminfo.total
        usages.append(usage)
    return usages


# 小于min_thres的都会被占用
# 上述不匹配，取最小一台，若该台占用小于one_thres则占用
def ditri_gpu(min_thres=0.1, one_thres=0.3, show=False):
    usages = cuda_usage()
    if show:
        print("Python  version : {}".format(sys.version.replace('\n', ' ')))
        print("Torch   version : {}".format(torch.__version__))
        print("Vision  version : {}".format(torchvision.__version__))
        print("GPU usage")
        for i in range(len(usages)):
            percent = usages[i] * 100
            print('    cuda:', str(i), ' using:', "%3.3f" % percent, '%')

    if np.min(usages) < min_thres:
        inds = [int(i) for i in range(len(usages)) if usages[i] < min_thres]
    elif np.min(usages) < one_thres:
        inds = [int(np.argmin(usages))]
    else:
        raise Exception('No available GPU')
    # 交换顺序device_ids[0]第一个出现
    inds = sorted(inds, key=lambda x: usages[x])
    return inds


# 自动确定device
def select_device(device=None, min_thres=0.01, one_thres=0.5):
    if device.__class__.__name__ == 'device':
        return [device.index]
    elif isinstance(device, int):
        return [device]
    elif device is None or len(device) == 0:
        print('Auto select device')
        if not torch.cuda.is_available():
            print('No gpu, use cpu')
            return [None]
        else:
            inds = ditri_gpu(min_thres=min_thres, one_thres=one_thres, show=False)
            return inds
    elif isinstance(device, list) or isinstance(device, tuple):
        return device
    elif isinstance(device, str) and len(device) > 0:
        return [torch.device(device).index]
    else:
        raise Exception('err device ' + str(device))


class PACK():
    DP = 'dp'
    DDP = 'DDP'
    AUTO = 'auto'
    NOOP = 'noop'


def pack_module(module, device_ids, pack=PACK.AUTO):
    if pack == PACK.NOOP:
        name = 'cpu' if device_ids[0] is None else str(device_ids[0])
        if len(device_ids) > 1:
            print('Find many devices, recommend DP')
        print('Module device --- ' + name)
        return module
    elif pack == PACK.DP:
        if device_ids[0] is None:
            raise Exception('No DataParallel on cpu')
        elif len(device_ids) == 1:
            print('Only single device, recommend NOOP')
        print('DataParallel devices --- ' + str(device_ids))
        return nn.DataParallel(module, device_ids=device_ids, output_device=device_ids[0])
    elif pack == PACK.DDP:
        if device_ids[0] is None:
            raise Exception('No DistributedDataParallel on cpu')
        print('DistributedDataParallel visible device --- ' + str(device_ids[0]))
        # module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module)
        return nn.parallel.DistributedDataParallel(
            module, device_ids=device_ids, output_device=device_ids[0])
    elif pack == PACK.AUTO or pack is None:
        print('* Auto pack')
        if len(device_ids) > 1:
            return pack_module(module, device_ids, pack=PACK.DP)
        else:
            return pack_module(module, device_ids, pack=PACK.NOOP)
    else:
        raise Exception('err pack')


def wait_until(device=None, thres=0.5, wait_time=1):
    device = select_device(device)
    while True:
        usages = np.array(cuda_usage())
        if np.all(usages[device] < thres):
            break
        else:
            print('Waiting')
            time.sleep(wait_time)
    print('Program continue')
    return True


if __name__ == '__main__':
    # wait_until(device=1, thres=0.8, wait_time=60)
    pass


# if __name__ == '__main__':
#     ids = select_device([1, 2])
#     import time
#
#     a = torch.zeros(100, 100, 100, 100)
#     time1 = time.time()
#     a = a.to(device=torch.device('cuda:0'))
#     time2 = time.time()
#     a = a.to(device=torch.device('cuda:0'))
#     time3 = time.time()
#     print(time2 - time1)
#     print(time3 - time2)
