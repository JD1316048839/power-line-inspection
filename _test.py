from models import *

bboxes = [np.array([8.89662720e+02, 6.25356140e+02, 1.30760193e+03, 8.58992126e+02,
                    9.64460075e-01, 0.00000000e+00]), np.array([21.8694706, 593.03039551, 407.25167847, 711.02197266,
                                                                0.89487875, 0.]),
          np.array([297.96942139, 11.30475998, 388.54150391, 376.59225464,
                    0.79861397, 0.]), np.array([491.27716064, 1.87624943, 610.88287354, 428.40148926,
                                                0.66030598, 0.]),
          np.array([4.72189819e+02, 3.75249023e+02, 6.39991882e+02, 4.51802795e+02,
                    3.00794601e-01, 2.00000000e+00])]


def pad_resize_img(img, size=224):
    img_arr = np.array(img)
    h, w, _ = img_arr.shape
    if h > w:
        wid = (h - w) // 2
        img_arr = np.pad(img_arr, pad_width=((0, 0), (wid, h - w - wid), (0, 0)))
    else:
        wid = (w - h) // 2
        img_arr = np.pad(img_arr, pad_width=((wid, w - h - wid), (0, 0), (0, 0)))
    img = Image.fromarray(img_arr.astype('uint8')).convert('RGB')
    img = img.resize([size, size])
    return img


if __name__ == '__main__':
    model = ResNet.R18(device=None, num_cls=6, img_size=(224, 224))
    sd_pth = '/home/user/JD/Python/Detect/chk/fau_res18_best.pth'
    model.load(sd_pth)
    image_path = '/ses-data/CY/jueyuanzi_detection/docs/images/000079.jpg'
    # img_pth = '/ses-data/Release/FaultC/Train/insulator_normal/000001_500KV_N.jpg'
    img = Image.open(image_path)
    for bbox in bboxes:
        xyxy = bbox[:4]
        xywh = xyxyN2xywhN(xyxy)
        xywh[2:4] *= 1.4
        xyxy = xywhN2xyxyN(xywh)
        xyxy, _ = xyxyN_clip(xyxy, xy_min=[0, 0], xy_max=original_image_size)
        xyxy = [int(v) for v in xyxy]
        ptch = img.crop(xyxy)
        ptch = pad_resize_img(ptch, size=224)
        ptch = pil2ten(ptch)
        cls = model.imgs2clses(ptch)[0]
        bbox[5] = cls['cls']
