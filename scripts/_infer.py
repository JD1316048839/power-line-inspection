import os
import sys

PROJECT_PTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PTH)
from data import *
from models import *
from tools import *
from utils import *

if __name__ == '__main__':
    ds = InsulatorD(pth='//home//data-storage//JD//InsulatorD')

    img_size = (960, 640)
    model = YoloV5.Small(num_cls=ds.num_cls, device=[1, 2, 3], img_size=img_size, pack=PACK.AUTO)
    file_name = os.path.join(PROJECT_PTH, 'ckpt/iusd_yv5')

    train_loader = ds.loader(set_name='trainval', batch_size=36, pin_memory=False, shuffle=True,
                             num_workers=4, aug_seqTp=AugSeq.DET_AUG_V3, img_size=img_size)
    test_loader = ds.loader(set_name='test', batch_size=64, pin_memory=False, shuffle=True, resample=False,
                            num_workers=4, aug_seqTp=AugSeq.DET_NORM, img_size=img_size)

    metric = Metric.MAP(loader=test_loader, conf_thres=0.001, iou_thres=0.45,
                        nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU)
    # metric(model)

    lrscheduler = LRScheduler.WARM_COS_CONST(lr=0.001, warm_epoch=3, total_epoch=200, lr_ratio=0.01, const_epoch=30)
    optimizer = OPTIMIZERTP.SGD(momentum=0.9, weight_decay=5e-4)(model.parameters())
    imscheduler = IMScheduler.Rand(max_size=img_size, min_size=(640, 320), devisor=64, keep_epoch=10, max_first=True)

    train(model=model, loader=train_loader, optimizer=optimizer, lrscheduler=lrscheduler,
          accu_step=2, save_step=10, file_name=file_name, imscheduler=imscheduler,
          test_step=10, metric=metric, best_pfrmce=0.5)