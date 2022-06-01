import os
import sys

PROJECT_PTH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_PTH)
from data import *
from models import *
from tools import *
from utils import *


def draw_img(img,boxes):
    draw = PIL.ImageDraw.ImageDraw(img)
    for box in boxes:
        xyxy = box['xyxy']
        xlyl = xyxyN2xlylN(xyxy)
        draw.line(list(xlyl.reshape(-1)) + list(xlyl[0]), fill='black', width=5)
    return img



if __name__ == '__main__':
    dir_src = os.path.join(PROJECT_PTH, 'src')
    dir_dst = os.path.join(PROJECT_PTH, 'dst')
    img_size_rpn = (960, 640)
    img_size_cls = (224, 224)

    seq_rpn = AugSeq.CLS_NORM(img_size=img_size_rpn)
    rpn = YoloV5.Small(device=[1, 2, 3], pack=PACK.AUTO, num_cls=1, img_size=img_size_rpn)
    # rpn.load('/')

    seq_cls = AugSeq.CLS_NORM(img_size=img_size_cls)
    cls = ResNet.R18(device=2, pack=PACK.AUTO, num_cls=7, img_size=img_size_cls)
    # clsr.load('/')

    img_names = os.listdir(dir_src)
    for img_name in img_names:
        img_pth_src = os.path.join(dir_src, img_name)
        img_pth_dst = os.path.join(dir_dst, img_name.replace('.jpg', '').replace('.JPG', '') + '_det.jpg')
        img = Image.open(img_pth_src)
        imgs_rpn, _ = seq_rpn([img], labels=None)
        boxes=rpn.imgs2boxss(imgs_rpn, conf_thres=0.3, iou_thres=0.45, cls2name=None,
                              nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU)
        draw_img(img,boxes)
        img.save(img_pth_dst)

