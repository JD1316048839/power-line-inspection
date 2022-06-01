import os
from collections import Counter

from utils.others import write_txt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torch.utils.data import ConcatDataset
from utils.frame import DataSourceFrame
from utils.label import *
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import ast


def node2dict(node):
    full_dict = {}
    for sub_node in node:
        if len(sub_node) == 0:
            try:
                val = ast.literal_eval(sub_node.text)
            except Exception as e:
                val = sub_node.text
            full_dict[sub_node.tag] = val
        else:
            full_dict[sub_node.tag] = node2dict(sub_node)
    return full_dict


def dict2node(full_dict, node=None):
    node = ET.Element(node) if isinstance(node, str) else node
    if isinstance(full_dict, dict):
        node.text = '\r\n' if node.text is None else node.text + '\r\n'
        node.tail = '\r\n' if node.tail is None else node.tail + '\r\n'
        for key, val in full_dict.items():
            sub_node = ET.SubElement(node, key) if node.find(key) is None else node.find(key)
            dict2node(val, node=sub_node)
    else:
        node.text = str(full_dict)
    return node


class VocDataset(Dataset):
    def __init__(self, pth, set_name, name2cls, resample=None,
                 set_folder='ImageSets/Main', img_folder='JPEGImages', anno_folder='Annotations'):
        self.pth = pth
        self.name2cls = name2cls
        self.set_name = set_name
        self.set_folder = set_folder
        self.img_folder = img_folder
        self.anno_folder = anno_folder
        self.img_pths, self.anno_pths = VocDataset.load_pth(
            pth, set_name, set_folder=set_folder, img_folder=img_folder, anno_folder=anno_folder)
        if resample is not None:
            presv_inds = VocDataset.resample_by_names(self.cls_names, resample=resample)
            self.img_pths = [self.img_pths[ind] for ind in presv_inds]
            self.anno_pths = [self.anno_pths[ind] for ind in presv_inds]

    @property
    def cls_names(self):
        cls_names = []
        for img_pth, anno_pth in zip(self.img_pths, self.anno_pths):
            if not os.path.exists(anno_pth):
                continue
            gts = VocDataset.parse_voc_xml(anno_pth, self.name2cls)
            cls_names.append(gts[0]['name'])
        return cls_names

    @staticmethod
    def resample_by_names(cls_names, resample):
        presv_inds = []
        for i, cls_name in enumerate(cls_names):
            if not cls_name in resample.keys():
                presv_inds.append(i)
                continue
            resamp_num = resample[cls_name]
            low = np.floor(resamp_num)
            high = np.ceil(resamp_num)
            resamp_num_rand = np.random.uniform(low=low, high=high)
            resamp_num = int(low if resamp_num_rand > resamp_num else high)
            for j in range(resamp_num):
                presv_inds.append(i)
        return presv_inds

    @staticmethod
    def load_pth(pth, set_name, set_folder='ImageSets/Main', img_folder='JPEGImages', anno_folder='Annotations'):
        anno_path = os.path.join(pth, set_folder, set_name + ".txt")
        with open(anno_path, 'r') as f:
            lines = f.readlines()
        codes = [line.replace('\n', '') for line in lines]
        imgs_pth = [os.path.join(pth, img_folder, code + '.jpg') for code in codes]
        annos_pth = [os.path.join(pth, anno_folder, code + '.xml') for code in codes]
        return imgs_pth, annos_pth

    @staticmethod
    def rename_obj_xml(anno_pth, rename_dict):
        root = ET.parse(anno_pth).getroot()
        objs = root.findall('object')
        for i, obj in enumerate(objs):
            name_node = obj.find('name')
            if name_node.text in rename_dict.keys():
                name_node.text = rename_dict[name_node.text]
        root_new = ET.ElementTree(root)
        root_new.write(anno_pth, encoding='utf-8')
        return True

    @staticmethod
    def parse_voc_xml(anno_pth, name2cls=None):
        root = ET.parse(anno_pth).getroot()
        code = root.find('filename').text.split('.')[0]
        objs = root.findall('object')
        boxes = BoxList(code=code)
        for i, obj in enumerate(objs):
            box = {}
            bndbox = obj.find('bndbox')
            if not bndbox is None:
                xyxy = np.array([
                    float(bndbox.find('xmin').text),
                    float(bndbox.find('ymin').text),
                    float(bndbox.find('xmax').text),
                    float(bndbox.find('ymax').text)
                ], dtype=np.int32)
                box['xyxy'] = xyxy
            robndbox = obj.find('robndbox')
            if not robndbox is None:
                xywha = np.array([
                    float(robndbox.find('cx').text),
                    float(robndbox.find('cy').text),
                    float(robndbox.find('w').text),
                    float(robndbox.find('h').text),
                    float(robndbox.find('angle').text)
                ], dtype=np.float32)
                box['xywha'] = xywha
            # 其它信息
            box['conf'] = 1
            difficult = obj.find('difficult')
            box['difficult'] = int(difficult.text) == 1 if difficult is not None else False
            box['name'] = obj.find('name').text
            if name2cls is not None:
                box['cls'] = name2cls(box['name'])
            # 框唯一标识
            box['meta'] = os.path.basename(anno_pth).split('.')[0]
            box['ind'] = i
            msg = obj.find('msg')
            if msg is not None:
                box['msg'] = node2dict(msg)
            boxes.append(box)
        return boxes

    @staticmethod
    def update_xml_msg(anno_pth, msg, anno_pth_new):
        if not os.path.exists(anno_pth):
            return None
        root = ET.parse(anno_pth).getroot()
        objs = root.findall('object')
        for i, obj in enumerate(objs):
            if i in msg.keys():
                node_msg = ET.SubElement(obj, 'msg') if obj.find('msg') is None else obj.find('msg')
                dict2node(msg[i], node=node_msg)
        root_new = ET.ElementTree(root)
        root_new.write(anno_pth_new, encoding='utf-8')
        return root_new

    @staticmethod
    def delete_xml_msg(anno_pth):
        if not os.path.exists(anno_pth):
            return None
        root = ET.parse(anno_pth).getroot()
        objs = root.findall('object')
        for i, obj in enumerate(objs):
            for node_msg in obj.findall('msg'):
                obj.remove(node_msg)
        root_new = ET.ElementTree(root)
        root_new.write(anno_pth, encoding='utf-8')
        return root_new

    @staticmethod
    def creat_xml(img_pth, w, h, boxes):
        anno_dict = {
            'folder': 'JPEGImages',
            'filename': os.path.basename(img_pth),
            'source': {'database': 'Unknown'},
            'size': {'width': w, 'height': h, 'depth': 3},
            'segmented': 0
        }
        root = dict2node(anno_dict, node='annotation')
        for box in boxes:
            obj = ET.SubElement(root, 'object')
            if 'xyxy' in box:
                xyxy = box['xyxy']
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = str(xyxy[0])
                ET.SubElement(bndbox, 'ymin').text = str(xyxy[1])
                ET.SubElement(bndbox, 'xmax').text = str(xyxy[2])
                ET.SubElement(bndbox, 'ymax').text = str(xyxy[3])
            if 'xywha' in box:
                xywha = box['xywha']
                robndbox = ET.SubElement(obj, 'robndbox')
                ET.SubElement(robndbox, 'cx').text = str(xywha[0])
                ET.SubElement(robndbox, 'cy').text = str(xywha[1])
                ET.SubElement(robndbox, 'w').text = str(xywha[2])
                ET.SubElement(robndbox, 'h').text = str(xywha[3])
                ET.SubElement(robndbox, 'angle').text = str(xywha[4])
            ET.SubElement(obj, 'difficult').text = '1' if 'difficult' in box.keys() and box['difficult'] else '0'
            ET.SubElement(obj, 'name').text = box['name']
        return root

    @staticmethod
    def trans_img_xml(img_pth, anno_pth, img_pth_new, anno_pth_new, aug_seq):
        img = Image.open(img_pth).convert("RGB")
        img = np.array(img)
        gts = VocDataset.parse_voc_xml(anno_pth, name2cls=None) if os.path.exists(anno_pth) else []
        imgs, gts = aug_seq([img], [gts])
        img, gts = arr2pil(imgs[0]), gts[0]
        w, h = img.size
        if not os.path.exists(anno_pth):
            root = VocDataset.creat_xml(img_pth, w, h, boxes=[])
        else:
            root = ET.parse(anno_pth).getroot()
            root.find('size').find('width').text = str(w)
            root.find('size').find('height').text = str(h)
            root.find('folder').text = 'JPEGImages'
            objs = root.findall('object')
            for i, obj in enumerate(objs):
                bndbox = obj.find('bndbox')
                xyxy = gts[i]['xyxy']
                bndbox.find('xmin').text = str(xyxy[0])
                bndbox.find('ymin').text = str(xyxy[1])
                bndbox.find('xmax').text = str(xyxy[2])
                bndbox.find('ymax').text = str(xyxy[3])
        root = ET.ElementTree(root)
        root.write(anno_pth_new, encoding='utf-8')
        img.save(img_pth_new)
        return img, root

    def __len__(self):
        return len(self.img_pths)

    def __getitem__(self, index):
        img_pth = self.img_pths[index]
        img = Image.open(img_pth).convert("RGB")
        img = np.array(img)
        anno_pth = self.anno_pths[index]
        gts = VocDataset.parse_voc_xml(anno_pth, self.name2cls) if os.path.exists(anno_pth) else []
        return img, gts

    def delete_msg(self):
        for anno_pth in self.anno_pths:
            VocDataset.delete_xml_msg(anno_pth)
        print('Deleting completed')
        return None

    def update_msg(self, msg_ds):
        print('Total %5d ' % len(self.anno_pths) + ' Msg %5d' % len(msg_ds))
        for anno_pth in self.anno_pths:
            # print('Update msg ' + anno_pth)
            meta = os.path.basename(anno_pth).split('.')[0]
            if meta in msg_ds.keys():
                msg_xml = msg_ds[meta]
                VocDataset.update_xml_msg(anno_pth=anno_pth, msg=msg_xml, anno_pth_new=anno_pth)
        print('Updating completed')
        return None

    def trans(self, imgs_dir, annos_dir, aug_seq):
        total_num = len(self.img_pths)
        for i, (img_pth, anno_pth) in enumerate(zip(self.img_pths, self.anno_pths)):
            print('Proc %06d' % i + '/%06d ' % total_num + img_pth)
            anno_pth_new = os.path.join(annos_dir, os.path.basename(anno_pth))
            img_pth_new = os.path.join(imgs_dir, os.path.basename(img_pth))
            VocDataset.trans_img_xml(img_pth=img_pth, anno_pth=anno_pth, img_pth_new=img_pth_new,
                                     anno_pth_new=anno_pth_new, aug_seq=aug_seq)
        print('Aug completed')
        return None

    def rename(self, rename_dict):
        for anno_pth in self.anno_pths:
            if os.path.exists(anno_pth):
                VocDataset.rename_obj_xml(anno_pth, rename_dict)
        print('Rename completed')
        return None

    # 通过采样建立新数据集
    def samp_set(self, set_name='train2', samp_rate=0.1):
        assert not set_name == self.set_name, 'set repeat'
        anno_path = os.path.join(self.pth, self.set_folder, self.set_name + ".txt")
        with open(anno_path, 'r') as f:
            lines = f.readlines()
        codes = np.array([line.replace('\n', '') for line in lines])
        np.random.shuffle(codes)
        cls_names=[]
        for code in codes:
            anno_pth = os.path.join(self.pth, self.anno_folder, code + '.xml')
            gts = VocDataset.parse_voc_xml(anno_pth, self.name2cls)
            cls_names.append(gts[0]['name'])
        cls_names = np.array(cls_names)
        cls_types = np.unique(cls_names)
        codes_new = []
        for cls_type in cls_types:
            codes_cls = codes[cls_names == cls_type]
            codes_cls = codes[:int(samp_rate * len(codes_cls))]
            codes_new.append(codes_cls)
        codes_new = np.concatenate(codes_new)
        txt_pth = os.path.join(self.pth, self.set_folder, set_name + '.txt')
        write_txt(txt_pth, codes_new)
        return codes

    # 统计样本个数
    def stat(self):
        num_dict = Counter(self.cls_names)
        for cls_name, num in num_dict.items():
            print('%10s ' % cls_name + ' %5d' % num)
        return None


if __name__ == '__main__':
    xml_pth1 = 'D://Desktop//000002.xml'
    xml_pth2 = 'D://Desktop//000003.xml'
    msg = {0: {'w': 's', 'h': 'r', 'a': 3}, 1: {'wid': 'r'}}
    VocDataset.update_xml_msg(xml_pth1, msg, xml_pth2)
    root = ET.parse(xml_pth2).getroot()
    a = VocDataset.parse_voc_xml(xml_pth2, lambda x: 1)
    print(a)


class Voc(DataSourceFrame):
    SEV_NEW_PTH = '//home//data-storage//VOC'
    SEV_OLD_PTH = '/home/exspace/dataset//VOC2007'
    DES_PTH = 'D://Datasets//VOC//'

    @staticmethod
    def SEV_NEW():
        return Voc(pth=Voc.SEV_NEW_PTH)

    @staticmethod
    def SEV_OLD():
        return Voc(pth=Voc.SEV_OLD_PTH)

    @staticmethod
    def DES():
        return Voc(pth=Voc.DES_PTH)

    def __init__(self, pth):
        super().__init__(img_size=(416, 416), pth=pth,
                         sets=('train', 'test', 'val', 'val07', 'trainval', 'train07', 'test07', 'trainval0712'),
                         num_cls=20)

        self.cls_names = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                          'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                          'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def dataset(self, set_name='train', **kwargs):
        if set_name == 'trainval0712' or set_name == 'trainval':
            ds07 = VocDataset(os.path.join(self.pth, 'VOCdevkit', 'VOC2007'), set_name='trainval',
                              name2cls=self.name2cls)
            ds12 = VocDataset(os.path.join(self.pth, 'VOCdevkit', 'VOC2012'), set_name='trainval',
                              name2cls=self.name2cls)
            ds = ConcatDataset([ds07, ds12])
        elif set_name == 'train07' or set_name == 'train':
            ds = VocDataset(os.path.join(self.pth, 'VOCdevkit', 'VOC2007'), set_name='train', name2cls=self.name2cls)
        elif set_name == 'test07' or set_name == 'test':
            ds = VocDataset(os.path.join(self.pth, 'VOCdevkit', 'VOC2007'), set_name='test', name2cls=self.name2cls)
        elif set_name == 'val07' or set_name == 'val':
            ds = VocDataset(os.path.join(self.pth, 'VOCdevkit', 'VOC2007'), set_name='val', name2cls=self.name2cls)
        else:
            raise Exception('err set')
        return ds

    def cls2name(self, cls):
        return self.cls_names[cls]

    def name2cls(self, name):
        return self.cls_names.index(name)

