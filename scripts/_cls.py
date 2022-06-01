import os
import sys

PROJECT_PTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PTH)
from data import *
from models import *
from tools import *
from utils import *

if __name__ == '__main__':
    ds = FaultC(pth='/ses-data/Release/FaultC', resample={'insulator_blast': 3})
    img_size = (224, 224)

    # model = EfficientNet.B0(device=[0,1], pack=PACK.AUTO, num_cls=ds.num_cls, img_size=img_size)
    # model = MobileNet.V3Large(device=2, pack=PACK.AUTO, num_cls=ds.num_cls, img_size=img_size)
    # model = VGG.DC(device=3, pack=PACK.AUTO, num_cls=ds.num_cls, img_size=img_size)
    model = ResNet.R18(device=2, pack=PACK.AUTO, num_cls=ds.num_cls, img_size=img_size)

    file_name = os.path.join(PROJECT_PTH, 'ckpt/iuscp_res18')

    train_loader = ds.loader(set_name='train', batch_size=128, pin_memory=False, shuffle=True, resample=True,
                             num_workers=5, aug_seqTp=AugSeq.CLS_AUG_V3, img_size=img_size)
    test_loader = ds.loader(set_name='test', batch_size=64, pin_memory=False, shuffle=True, resample=False,
                            num_workers=5, aug_seqTp=AugSeq.CLS_NORM, img_size=img_size)

    metric = Metric.PrecRcal(loader=test_loader, total_epoch=1)
    # metric(model)

    lrscheduler = LRScheduler.WARM_COS(lr=0.01, warm_epoch=5, total_epoch=300)
    optimizer = OPTIMIZERTP.SGD(momentum=0.9, weight_decay=1e-5)(filter(lambda x: x.requires_grad, model.parameters()))
    train(model=model, loader=train_loader, optimizer=optimizer, lrscheduler=lrscheduler,
          accu_step=1, grad_clip=100, save_step=5, file_name=file_name,
          test_step=5, metric=metric, best_pfrmce=0.8)

    sys.exit(0)
