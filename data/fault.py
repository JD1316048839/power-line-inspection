import os
from collections import Counter

from utils.others import write_txt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from data.voc import VocDataset, dict2node, node2dict
from utils.frame import DataSourceFrame
from torch.utils.data import Dataset
import shutil
from aug import AugSeq, iaa
import torch
import numpy as np
from PIL import Image


class FaultC(DataSourceFrame):
    def __init__(self, pth, resample=None, expend=None):
        super().__init__(img_size=(224, 224), pth=pth, sets=('train', 'test'), num_cls=7)
        self.pth = pth
        self.cls_names = ['insulator_normal', 'insulator_blast', 'insulator_comp', 'metal_normal', 'metal_rust',
                          'clamp_normal', 'clamp_rust']
        self.resample = resample
        self.expend = expend

    def dataset(self, set_name, **kwargs):
        if set_name == 'train':
            dataset = FolderDataset(pth=os.path.join(self.pth, 'Train'),
                                    name2cls=self.name2cls, resample=self.resample, set_name=set_name, preload=False)
            if self.expend is not None:
                dataset.append(self.expend)
            return dataset
        else:
            return FolderDataset(pth=os.path.join(self.pth, 'Test'),
                                 name2cls=self.name2cls, resample=None, set_name=set_name, preload=False)

    def cls2name(self, cls):
        return self.cls_names[cls]

    def name2cls(self, name):
        return self.cls_names.index(name)


class FolderDataset(Dataset):
    def __init__(self, pth, name2cls, set_name='', resample=None, preload=False, folders=None):
        self.name2cls = name2cls
        self.pth = pth
        self.set_name = set_name
        if folders is None:
            folders = os.listdir(pth)
        img_pths, cls_names = FolderDataset.get_pths(pth, folders=folders)
        self.img_pths = img_pths
        self.cls_names = cls_names
        if resample is not None:
            presv_inds = VocDataset.resample_by_names(self.cls_names, resample=resample)
            self.img_pths = [self.img_pths[ind] for ind in presv_inds]
            self.cls_names = [self.cls_names[ind] for ind in presv_inds]
        self.preload = False
        if preload:
            print('Pre load Data')
            data = []
            total_num = self.__len__()
            for i in range(total_num):
                if i % 100 == 0 or i + 1 == total_num:
                    print('Loading %5d /' % (i + 1) + '%-5d' % total_num)
                data.append(self.__getitem__(i))
            self.data = data
            self.preload = True

    @staticmethod
    def get_pths(set_dir, folders):
        img_pths = []
        cls_names = []
        for cls_name in folders:
            tar_dir = os.path.join(set_dir, cls_name)
            for img_name in os.listdir(tar_dir):
                if not str.endswith(img_name, '.jpg'):
                    continue
                img_pths.append(os.path.join(tar_dir, img_name))
                cls_names.append(cls_name)
        return img_pths, cls_names

    @staticmethod
    def create_xml(img_pth, img_size, cls_name='', database='', annotation=''):
        xml_dict = {
            'folder': os.path.dirname(img_pth),
            'filename': os.path.basename(img_pth),
            'source': {
                'database': database,
                'annotation': annotation,
                'image': img_pth
            },
            'size': {
                'width': img_size[0],
                'height': img_size[1],
                'depth': 3
            },
            'object': {
                'name': cls_name,
                'bndbox': {
                    'xmin': 0,
                    'ymin': 0,
                    'xmax': img_size[0],
                    'ymax': img_size[1]

                }
            },
        }
        return xml_dict

    def __len__(self):
        return len(self.img_pths)

    def __getitem__(self, index):
        if self.preload:
            return self.data[index]
        img_pth, cls_name = self.img_pths[index], self.cls_names[index]
        img = np.array(Image.open(img_pth))
        h, w, _ = img.shape
        cls = int(self.name2cls(cls_name))
        target = {
            'cls': cls,
            'name': cls_name,
            'w': w, 'h': h,
            'meta': os.path.basename(img_pth).split('.')[0]
        }
        return img, target

    def append(self, dir_dict):
        for cls_name, dir in dir_dict.items():
            img_names = os.listdir(dir)
            for img_name in img_names:
                img_pth = os.path.join(dir, img_name)
                self.img_pths.append(img_pth)
                self.cls_names.append(cls_name)
        return True

    def stat(self):
        num_dict = Counter(self.cls_names)
        for cls_name, num in num_dict.items():
            print('%10s ' % cls_name + ' %5d' % num)
        return None

    def delete(self):
        print('Delete dataset ' + self.set_name)
        for pth in self.img_pths:
            if os.path.isfile(pth):
                os.remove(pth)
        print('Deleting complete')
        return None
