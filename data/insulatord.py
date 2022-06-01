import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from data.voc import VocDataset, dict2node
from utils.frame import DataSourceFrame
import shutil
from utils.label import *


class InsulatorD(DataSourceFrame):
    SEV_OLD_PTH = '/home/user/JD/Datasets/InsulatorD'
    SEV_NEW_ORI_PTH = '//ses-data//CY//jueyuanzi'
    SEV_NEW_PTH = '//home//data-storage//JD//InsulatorD'
    BOARD_PTH = '/home/jd/data/DataSets/InsulatorD'
    DES_PTH = 'D://Datasets//Insulator//'

    @staticmethod
    def DES():
        return InsulatorD(pth=InsulatorD.DES_PTH)

    @staticmethod
    def BOARD():
        return InsulatorD(pth=InsulatorD.BOARD_PTH)

    @staticmethod
    def SEV_OLD():
        return InsulatorD(pth=InsulatorD.SEV_OLD_PTH)

    @staticmethod
    def SEV_NEW_ORI():
        return InsulatorD(pth=InsulatorD.SEV_NEW_ORI_PTH)

    @staticmethod
    def SEV_NEW():
        return InsulatorD(pth=InsulatorD.SEV_NEW_PTH, anno_folder='Annotations')

    @staticmethod
    def SEV_NEW_2():
        return InsulatorD(pth=InsulatorD.SEV_NEW_PTH, anno_folder='Annotations2')

    def __init__(self, pth,
                 set_folder='ImageSets/Main', img_folder='JPEGImages', anno_folder='Annotations'):
        super().__init__(img_size=(416, 416), pth=pth, sets=('train', 'test', 'val', 'trainval'), num_cls=1)
        self.cls_names = ['insulator']
        # self.cls_names = ['jueyuanziww']
        self.set_folder = set_folder
        self.img_folder = img_folder
        self.anno_folder = anno_folder

    def dataset(self, set_name, **kwargs):
        return VocDataset(pth=self.pth, set_name=set_name, name2cls=self.name2cls)

    def cls2name(self, cls):
        return self.cls_names[cls]
        # return 'ius'

    def name2cls(self, name):
        return self.cls_names.index(name)

    def trans(self, pth, aug_seq,
              set_folder='ImageSets/Main', img_folder='JPEGImages', anno_folder='Annotations'):
        if not os.path.exists(pth):
            os.makedirs(pth)
        imgs_dir = os.path.join(pth, img_folder)
        annos_dir = os.path.join(pth, anno_folder)
        sets_dir = os.path.join(pth, set_folder)
        if os.path.exists(sets_dir):
            shutil.rmtree(sets_dir)
        shutil.copytree(os.path.join(self.pth, set_folder), sets_dir)
        for dir in [imgs_dir, annos_dir]:
            if not os.path.exists(dir):
                os.makedirs(dir)
        for set in ['test', 'val']:
            print('Set ' + set)
            ds = VocDataset(pth=self.pth, set_name=set, name2cls=self.name2cls)
            ds.trans(imgs_dir=imgs_dir, annos_dir=annos_dir, aug_seq=aug_seq)
        return None


class InsulatorD2(DataSourceFrame):
    SEV_NEW_PTH = '//home//data-storage//JD//InsulatorD2'

    @staticmethod
    def SEV_NEW():
        return InsulatorD2(pth=InsulatorD2.SEV_NEW_PTH, anno_folder='Annotations')

    def __init__(self, pth,
                 set_folder='ImageSets/Main', img_folder='JPEGImages', anno_folder='Annotations'):
        super().__init__(img_size=(416, 416), pth=pth, sets=('train', 'test', 'val', 'trainval'), num_cls=1)
        self.cls_names = ['glass_insulator', 'composite_insulator', 'clamp', 'metal']
        # self.cls_names = ['insulator_normal', 'insulator_blast', 'insulator_comp', 'metal_normal', 'metal_rust',
        #                   'clamp_normal', 'clamp_rust']
        self.set_folder = set_folder
        self.img_folder = img_folder
        self.anno_folder = anno_folder

    def dataset(self, set_name, **kwargs):
        return VocDataset(pth=self.pth, set_name=set_name, name2cls=self.name2cls)

    def cls2name(self, cls):
        return self.cls_names[cls]

    def name2cls(self, name):
        return self.cls_names.index(name.lower())

