from .modules import *
from utils import *


# SPP
class SPP(nn.Module):
    def __init__(self, kernels=(13, 9, 5), stride=1, shortcut=True):
        super(SPP, self).__init__()
        self.pools = nn.ModuleList()
        for kernel in kernels:
            padding = (kernel - 1) // 2
            self.pools.append(nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding))
        self.shortcut = shortcut

    def forward(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            outs = []
            for pool in self.pools:
                outs.append(pool(x))
            if self.shortcut:
                outs.append(x)
            outs = torch.cat(outs, dim=1)
        return outs


class CSPConv(nn.Module):
    def __init__(self, in_channels, out_channels, repeat_num=1, act=ACT.LK):
        super().__init__()
        inner_channels = out_channels // 2
        self.shortcut = Conv1(in_channels=in_channels, out_channels=inner_channels, act=act)

        backbone = [Conv1(in_channels=in_channels, out_channels=inner_channels, act=act)]
        for n in range(repeat_num):
            backbone.append(Conv1(in_channels=inner_channels, out_channels=inner_channels, act=act))
            backbone.append(Conv3(in_channels=inner_channels, out_channels=inner_channels, act=act))
        self.backbone = nn.Sequential(*backbone)

        self.concater = Conv1(in_channels=inner_channels * 2, out_channels=out_channels, act=act)

    def forward(self, x):
        out = torch.cat([self.backbone(x), self.shortcut(x)], dim=1)
        out = self.concater(out)
        return out


class Focus(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)


# Conv+Conv+Res
class ConvResidual(nn.Module):
    def __init__(self, channels, inner_channels=None, act=ACT.LK):
        super(ConvResidual, self).__init__()
        if inner_channels is None:
            inner_channels = channels // 2
        self.backbone = nn.Sequential(
            Conv1(channels, inner_channels, act=act),
            Conv3(inner_channels, channels, act=act),
        )

    def forward(self, x):
        x = x + self.backbone(x)
        return x


# ConvResidual*repeat_num
def ConvResidualRepeat(channels, inner_channels=None, repeat_num=1, act=ACT.LK):
    backbone = nn.Sequential()
    for i in range(repeat_num):
        backbone.add_module(str(i), ConvResidual(channels=channels, inner_channels=inner_channels, act=act))

    return backbone


# ConvResidualRepeat+Conv+Res
class CSPBlockV5(nn.Module):
    def __init__(self, channels, shortcut_channels, backbone_channels, backbone_inner_channels,
                 repeat_num, act=ACT.LK):
        super(CSPBlockV5, self).__init__()
        self.shortcut = Conv1(in_channels=channels, out_channels=shortcut_channels, act=act)
        self.backbone = nn.Sequential(
            Conv1(in_channels=channels, out_channels=backbone_channels, act=act),
            ConvResidualRepeat(channels=backbone_channels, inner_channels=backbone_inner_channels,
                               repeat_num=repeat_num, act=act)
        )
        self.concater = Conv1(in_channels=shortcut_channels + backbone_channels, out_channels=channels, act=act)

    def forward(self, x):
        out = torch.cat([self.backbone(x), self.shortcut(x)], dim=1)
        out = self.concater(out)
        return out


class YoloFrame(OneStageModelFrame, IDetection):

    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        super(YoloFrame, self).__init__(backbone=backbone, device=device, pack=pack)

    @property
    def img_size(self):
        return self.backbone.img_size

    @img_size.setter
    def img_size(self, img_size):
        self.backbone.img_size = img_size

    @property
    def num_pbx(self):
        return np.sum([layer.num_pbx for layer in self.backbone.layers])

    @property
    def pboxes(self):
        return torch.cat([layer.pboxes for layer in self.backbone.layers], dim=0)

    def imgs_tard2loss(self, imgs, target, **kwargs):
        imgs = imgs.to(self.device)
        pred = self.pkd_modules['backbone'](imgs)
        loss = self.pred_tard2loss(pred, target)
        return loss

    @abstractmethod
    def pred_tard2loss(self, pred, target, **kwargs):
        pass

    @torch.no_grad()
    def imgs2boxss(self, imgs, conf_thres=0.7, iou_thres=0.7, by_cls=True,
                   nms_type=NMS_TYPE.HARD, iou_type=IOU_TYPE.IOU, cls2name=None):
        imgs = imgs.to(self.device)
        pred = self.pkd_modules['backbone'](imgs)
        ###############################调试
        pred_np = pred.detach().cpu().numpy()
        print(pred_np)
        ################################
        max_val, clsessT = torch.max(pred[..., 5:], dim=2)
        confssT = pred[..., 4] * max_val
        boxssT = xywhsT2xyxysT(pred[..., :4])
        boxss = []
        for boxsT, confsT, clsesT in zip(boxssT, confssT, clsessT):
            prsv_msks = confsT > conf_thres
            if not torch.any(prsv_msks):
                boxss.append([])
                continue
            boxsT, confsT, clsesT = boxsT[prsv_msks], confsT[prsv_msks], clsesT[prsv_msks]
            prsv_inds = nmsT(boxesT=boxsT, confsT=confsT, clsesT=clsesT, box_fmt=BOX_FMT.XYXY,
                             iou_thres=iou_thres, nms_type=nms_type, iou_type=iou_type)
            if len(prsv_inds) == 0:
                boxss.append([])
                continue
            boxsT, confsT, clsesT = boxsT[prsv_inds], confsT[prsv_inds], clsesT[prsv_inds]
            boxs = boxesT_confsT_clsesT2boxes(boxesT=boxsT, confsT=confsT, clsesT=clsesT, cls2name=cls2name,
                                              box_fmt=BOX_FMT.XYXY)
            boxss.append(boxs)
        return boxss

    # 静态匹配策略
    @staticmethod
    def gtss2tardStaticMatch(gtss, layers, layer_matcher, num_cls, **kwargs):
        inds_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_pos = [np.zeros(shape=0, dtype=np.int32)]
        inds_layer = [np.zeros(shape=0, dtype=np.int32)]
        xywh = [np.zeros(shape=(0, 4))]
        cls = [np.zeros(shape=(0, num_cls))]
        inds_neg = [np.zeros(shape=0, dtype=np.int32)]
        inds_b_neg = [np.zeros(shape=0, dtype=np.int32)]

        for i, gts in enumerate(gtss):
            gts_ocls = boxes2oclsesN(gts, num_cls=num_cls)
            gts_xyxy = boxes2boxesN(gts, box_fmt=BOX_FMT.XYXY)
            gts_xywh = xyxysN2xywhsN(gts_xyxy)
            if gts_xyxy.shape[0] == 0:
                num_pbx = np.sum([layer.num_pbx for layer in layers])
                inds_b_neg.append(np.full(fill_value=i, shape=num_pbx))
                inds_neg.append(np.arange(num_pbx))
                continue

            offset_layer = 0
            for j, layer in enumerate(layers):
                num_pbx_layer = layer.num_pbx

                ids, ids_gt = layer_matcher(layer=layer, gts_xyxy=gts_xyxy, gts_xywh=gts_xywh, **kwargs)

                inds_b_pos.append(np.full(fill_value=i, shape=len(ids)))
                inds_layer.append(np.full(fill_value=j, shape=len(ids)))
                inds_pos.append(offset_layer + ids)
                cls.append(gts_ocls[ids_gt])
                xywh.append(gts_xywh[ids_gt])
                inds_b_neg.append(np.full(fill_value=i, shape=num_pbx_layer - len(ids)))
                inds_neg.append(offset_layer + np.delete(np.arange(num_pbx_layer), ids))
                offset_layer = offset_layer + num_pbx_layer

        inds_b_neg = np.concatenate(inds_b_neg, axis=0)
        inds_neg = np.concatenate(inds_neg, axis=0)
        inds_b_pos = np.concatenate(inds_b_pos, axis=0)
        inds_pos = np.concatenate(inds_pos, axis=0)
        xywh = np.concatenate(xywh, axis=0)
        cls = np.concatenate(cls, axis=0)
        inds_layer = np.concatenate(inds_layer, axis=0)
        target = (inds_b_pos, inds_pos, xywh, cls, inds_layer, inds_b_neg, inds_neg)
        return target

    # 静态计算loss
    @staticmethod
    def pred_tard2lossStaticMatch(pred, target, calc_loss_detial, **kwargs):
        Nb = pred.size(0)
        inds_b_pos, inds_pos, xywh, cls, inds_layer, inds_b_neg, inds_neg = target
        pred_xywh = pred[inds_b_pos, inds_pos, :4]
        pred_pos = pred[inds_b_pos, inds_pos, 4]
        pred_cls = pred[inds_b_pos, inds_pos, 5:]
        pred_neg = pred[inds_b_neg, inds_neg, 4]

        loss = calc_loss_detial(pred_xywh=pred_xywh, pred_pos=pred_pos, pred_cls=pred_cls, pred_neg=pred_neg,
                                xywh=xywh, cls=cls, inds_layer=inds_layer, Nb=Nb, **kwargs)
        return loss


class YoloV5Bkbn(nn.Module):
    def __init__(self, act=ACT.SILU, channels=64, repeat_num=3):
        super().__init__()
        self.pre = nn.Sequential(
            Focus(),
            Conv3(in_channels=12, out_channels=channels, act=act),
            # Conv(in_channels=3, out_channels=channels, kernel_size=6, stride=2, act=act), #无Focus版本
        )
        self.stage1 = nn.Sequential(
            ConvAP(in_channels=channels, out_channels=channels * 2, kernel_size=3, stride=2, act=act),
            CSPBlockV5(channels=channels * 2, shortcut_channels=channels, backbone_channels=channels,
                       backbone_inner_channels=channels, repeat_num=repeat_num, act=act)
        )
        self.stage2 = nn.Sequential(
            ConvAP(in_channels=channels * 2, out_channels=channels * 4, kernel_size=3, stride=2, act=act),
            CSPBlockV5(channels=channels * 4, shortcut_channels=channels * 2, backbone_channels=channels * 2,
                       backbone_inner_channels=channels * 2, repeat_num=repeat_num * 2, act=act)
        )
        self.stage3 = nn.Sequential(
            ConvAP(in_channels=channels * 4, out_channels=channels * 8, kernel_size=3, stride=2, act=act),
            CSPBlockV5(channels=channels * 8, shortcut_channels=channels * 4, backbone_channels=channels * 4,
                       backbone_inner_channels=channels * 4, repeat_num=repeat_num * 3, act=act)
        )
        self.stage4 = nn.Sequential(
            ConvAP(in_channels=channels * 8, out_channels=channels * 16, kernel_size=3, stride=2, act=act),
            CSPBlockV5(channels=channels * 16, shortcut_channels=channels * 8, backbone_channels=channels * 8,
                       backbone_inner_channels=channels * 8, repeat_num=repeat_num, act=act)
        )

    def forward(self, imgs):
        feat = self.pre(imgs)
        out1 = self.stage1(feat)
        out2 = self.stage2(out1)
        out3 = self.stage3(out2)
        out4 = self.stage4(out3)
        return out2, out3, out4


class YoloV5UpSamper(nn.Module):
    def __init__(self, act=ACT.SILU, channels=64, repeat_num=3):
        super(YoloV5UpSamper, self).__init__()
        self.c3 = nn.Sequential(
            Conv1(in_channels=channels * 16, out_channels=channels * 8, act=act),
            SPP(kernels=(13, 9, 5), stride=1, shortcut=True),
            Conv1(in_channels=channels * 32, out_channels=channels * 16, act=act),
            Conv1(in_channels=channels * 16, out_channels=channels * 8, act=act)
        )

        self.c2_3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.c2 = nn.Sequential(
            CSPConv(in_channels=channels * 16, out_channels=channels * 8, repeat_num=repeat_num, act=act),
            Conv1(in_channels=channels * 8, out_channels=channels * 4, act=act),
        )

        self.c1_2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.c1 = CSPConv(in_channels=channels * 8, out_channels=channels * 4, repeat_num=repeat_num, act=act)

    def forward(self, featmaps):
        feat1, feat2, feat3 = featmaps
        out3 = self.c3(feat3)
        pre2 = torch.cat((self.c2_3(out3), feat2), dim=1)
        out2 = self.c2(pre2)
        pre1 = torch.cat((self.c1_2(out2), feat1), dim=1)
        out1 = self.c1(pre1)
        return out1, out2, out3


class YoloV5DownSamper(nn.Module):
    def __init__(self, act=ACT.SILU, channels=64, repeat_num=3):
        super(YoloV5DownSamper, self).__init__()
        self.c2_1 = ConvAP(in_channels=channels * 4, out_channels=channels * 4, kernel_size=3, stride=2, act=act)
        self.c2 = CSPConv(in_channels=channels * 8, out_channels=channels * 8, repeat_num=repeat_num, act=act)

        self.c3_2 = ConvAP(in_channels=channels * 8, out_channels=channels * 8, kernel_size=3, stride=2, act=act)
        self.c3 = CSPConv(in_channels=channels * 16, out_channels=channels * 16, repeat_num=repeat_num, act=act)

    def forward(self, featmaps):
        feat1, feat2, feat3 = featmaps
        out1 = feat1
        pre2 = torch.cat((self.c2_1(out1), feat2), dim=1)
        out2 = self.c2(pre2)
        pre3 = torch.cat((self.c3_2(out2), feat3), dim=1)
        out3 = self.c3(pre3)
        return out1, out2, out3


class YoloV5Layer(AnchorLayerImg):
    def __init__(self, in_channels, anchors, stride, num_cls, img_size=(0, 0)):
        super().__init__(anchors=anchors, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.reg_cls = Conv1Pure(in_channels=in_channels, out_channels=self.Na * (num_cls + 5))
        init_sig(self.reg_cls.conv.bias[4:self.Na * (num_cls + 5):(num_cls + 5)], prior_prob=0.1)

    def forward(self, featmap):
        featmap = self.reg_cls(featmap)
        pred = YoloV5Layer.decode(
            featmap=featmap, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
            stride=self.stride, num_cls=self.num_cls)
        return pred

    @staticmethod
    def decode(featmap, xy_offset, wh_offset, stride, num_cls):
        xy_offset = xy_offset.to(featmap.device, non_blocking=True)
        wh_offset = wh_offset.to(featmap.device, non_blocking=True)
        Hf, Wf, Na, _ = list(xy_offset.size())
        featmap = featmap.permute(np.int32(0), np.int32(2), np.int32(3), np.int32(1))
        featmap = featmap.reshape(-1, Hf, Wf, Na, num_cls + 5).contiguous()

        x = (torch.sigmoid(featmap[..., 0]) * 2 - 0.5 + xy_offset[..., 0]) * stride
        y = (torch.sigmoid(featmap[..., 1]) * 2 - 0.5 + xy_offset[..., 1]) * stride
        w = (torch.sigmoid(featmap[..., 2]) * 2) ** 2 * wh_offset[..., 0]
        h = (torch.sigmoid(featmap[..., 3]) * 2) ** 2 * wh_offset[..., 1]

        conf_cls = torch.sigmoid(featmap[..., 4:])
        pred = torch.cat([x[..., None], y[..., None], w[..., None], h[..., None], conf_cls], dim=-1).contiguous()
        pred = pred.reshape(-1, Na * Wf * Hf, num_cls + 5)
        return pred


class YoloV5Main(nn.Module):
    def __init__(self, num_cls=80, act=ACT.SILU, channels=64, repeat_num=3, img_size=(0, 0)):
        super(YoloV5Main, self).__init__()
        self.num_cls = num_cls
        self.W, self.H = img_size
        self.backbone = YoloV5Bkbn(act=act, channels=channels, repeat_num=repeat_num)
        self.necker = nn.Sequential(
            YoloV5UpSamper(act=act, channels=channels, repeat_num=repeat_num),
            YoloV5DownSamper(act=act, channels=channels, repeat_num=repeat_num)
        )
        self.layers = self.layers = nn.ModuleList([
            YoloV5Layer(in_channels=channels * 4, anchors=([10, 13], [16, 30], [33, 23]), stride=8,
                        num_cls=num_cls, img_size=img_size),
            YoloV5Layer(in_channels=channels * 8, anchors=([30, 61], [62, 45], [59, 119]), stride=16,
                        num_cls=num_cls, img_size=img_size),
            YoloV5Layer(in_channels=channels * 16, anchors=([116, 90], [156, 198], [373, 326]), stride=32,
                        num_cls=num_cls, img_size=img_size)
        ])

    @property
    def img_size(self):
        return (self.W, self.H)

    @img_size.setter
    def img_size(self, img_size):
        self.W, self.H = img_size
        for layer in self.layers:
            layer.img_size = img_size

    def forward(self, imgs):
        featmaps = self.backbone(imgs)
        featmaps = self.necker(featmaps)
        pred = [layer(feat) for layer, feat in zip(self.layers, featmaps)]
        pred = torch.cat(pred, dim=1)
        return pred

    NANO_PARA = dict(channels=16, repeat_num=1)  # depth=0.33,width=0.25
    SMALL_PARA = dict(channels=32, repeat_num=1)  # depth=0.33,width=0.5
    MEDIUM_PARA = dict(channels=48, repeat_num=2)
    LARGE_PARA = dict(channels=64, repeat_num=3)
    XLARGE_PARA = dict(channels=80, repeat_num=4)

    @staticmethod
    def Nano(num_cls=80, act=ACT.SILU, img_size=(0, 0)):
        return YoloV5Main(**YoloV5Main.NANO_PARA, num_cls=num_cls, act=act, img_size=img_size)

    @staticmethod
    def Small(num_cls=80, act=ACT.SILU, img_size=(0, 0)):
        return YoloV5Main(**YoloV5Main.SMALL_PARA, num_cls=num_cls, act=act, img_size=img_size)

    @staticmethod
    def Medium(num_cls=80, act=ACT.SILU, img_size=(0, 0)):
        return YoloV5Main(**YoloV5Main.MEDIUM_PARA, num_cls=num_cls, act=act, img_size=img_size)

    @staticmethod
    def Large(num_cls=80, act=ACT.SILU, img_size=(0, 0)):
        return YoloV5Main(**YoloV5Main.LARGE_PARA, num_cls=num_cls, act=act, img_size=img_size)

    @staticmethod
    def XLarge(num_cls=80, act=ACT.SILU, img_size=(0, 0)):
        return YoloV5Main(**YoloV5Main.XLARGE_PARA, num_cls=num_cls, act=act, img_size=img_size)


class YoloV5ConstLayer(AnchorLayerImg):
    def __init__(self, batch_size, anchors, stride, num_cls, img_size=(0, 0)):
        super().__init__(anchors=anchors, stride=stride, img_size=img_size)
        self.num_cls = num_cls
        self.featmap = nn.Parameter(torch.zeros(batch_size, self.Na * (num_cls + 5), self.Hf, self.Wf))
        init_sig(self.featmap[:, 4:self.Na * (num_cls + 5):(num_cls + 5), :, :], prior_prob=0.1)

    def forward(self, featmap):
        featmap = self.featmap
        pred = YoloV5Layer.decode(featmap=featmap, xy_offset=self.xy_offset, wh_offset=self.wh_offset,
                                  stride=self.stride, num_cls=self.num_cls)
        return pred


class YoloV5ConstMain(nn.Module):
    def __init__(self, num_cls=80, img_size=(416, 352), batch_size=3):
        super(YoloV5ConstMain, self).__init__()
        self.num_cls = num_cls
        self.img_size = img_size
        self.layers = nn.ModuleList([
            YoloV5ConstLayer(batch_size=batch_size, anchors=([10, 13], [16, 30], [33, 23]), stride=8,
                             num_cls=num_cls, img_size=img_size),
            YoloV5ConstLayer(batch_size=batch_size, anchors=([30, 61], [62, 45], [59, 119]), stride=16,
                             num_cls=num_cls, img_size=img_size),
            YoloV5ConstLayer(batch_size=batch_size, anchors=([116, 90], [156, 198], [373, 326]), stride=32,
                             num_cls=num_cls, img_size=img_size)
        ])

    def forward(self, imgs):
        pred = torch.cat([layer(None) for layer in self.layers], dim=1)
        return pred


class YoloV5(YoloFrame):
    def __init__(self, backbone, device=None, pack=PACK.AUTO):
        super().__init__(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def layer_matcher(layer, gts_xyxy, gts_xywh, whr_thresh=4, **kwargs):
        num_gt = gts_xyxy.shape[0]
        Na, stride, Wf, Hf = layer.Na, layer.stride, layer.Wf, layer.Hf
        anchors = layer.anchors.detach().cpu().numpy()

        gt_offset = np.array([[0, 0], [0, 1], [1, 0], [0, -1], [-1, 0]])
        ixy = np.repeat(gts_xywh[:, None, :2], axis=1, repeats=5) + gt_offset * 0.5 * stride
        ixy = (ixy // stride).astype(np.int32)
        ids = (np.clip(ixy[:, :, 1], 0, Hf - 1) * Wf + np.clip(ixy[:, :, 0], 0, Wf - 1)) * Na
        ids = np.repeat(ids[:, :, None], axis=2, repeats=Na) + np.arange(Na)

        ids_gt = np.broadcast_to(np.arange(num_gt)[:, None, None], shape=(num_gt, 5, Na))
        whr_val = gts_xywh[:, None, None, 2:4] / anchors[None, None, :, :]
        whr_val = np.max(np.maximum(whr_val, 1 / whr_val), axis=3)
        whr_val = np.repeat(whr_val, axis=1, repeats=5)
        whr_fliter = whr_val < whr_thresh
        # 去除重复
        ids, ids_gt = ids[whr_fliter], ids_gt[whr_fliter]
        ids, repeat_fliter = np.unique(ids, return_index=True)
        ids_gt = ids_gt[repeat_fliter]
        return ids, ids_gt

    def gtss2tard(self, gtss, whr_thresh=4):
        target = YoloFrame.gtss2tardStaticMatch(
            layers=self.backbone.layers, num_cls=self.backbone.num_cls, gtss=gtss,
            layer_matcher=YoloV5.layer_matcher, whr_thresh=whr_thresh)
        return target

    @staticmethod
    def calc_loss_detial(pred_xywh, pred_pos, pred_cls, pred_neg, xywh, cls, inds_layer, img_size, Nb, **kwargs):
        xywh = torch.as_tensor(xywh, dtype=torch.float).to(pred_xywh.device, non_blocking=True)
        cls = torch.as_tensor(cls, dtype=torch.float).to(pred_cls.device, non_blocking=True)
        Np = xywh.size(0)
        area = img_size[0] * img_size[1]
        # IOU损失
        ious = clac_iou_arrT(xywhsT2xyxysT(pred_xywh), xywhsT2xyxysT(xywh), iou_type=IOU_TYPE.CIOU)
        iou_power = 2 - xywh[:, 2] * xywh[:, 3] / area
        iou_loss = torch.mean((1 - ious) * iou_power)
        # 目标检出损失
        pos_loss = Focalloss(pred_pos, ious.detach().clamp(0), alpha=0.25, gamma=2, reduction='none')
        balance = torch.Tensor([4, 1, 0.4]).to(device=pos_loss.device, non_blocking=True)[inds_layer]  # 不同层权重不同
        pos_loss = torch.mean(pos_loss * balance) * 3
        neg_loss = -torch.sum(torch.log(1 - pred_neg + 1e-8)) / Np * 0.5
        # 分类损失
        cls_loss = Focalloss(pred_cls, cls, alpha=0.25, gamma=2, reduction='mean') * cls.size(1)

        return OrderedDict(pos=pos_loss, neg=neg_loss, iou=iou_loss, cls=cls_loss)

    def pred_tard2loss(self, predd, target, **kwargs):
        loss = YoloFrame.pred_tard2lossStaticMatch(
            pred=predd, target=target, calc_loss_detial=YoloV5.calc_loss_detial,
            img_size=self.img_size)
        return loss

    @staticmethod
    def Nano(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV5Main.Nano(num_cls=num_cls, act=ACT.SILU, img_size=img_size)
        return YoloV5(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Small(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV5Main.Small(num_cls=num_cls, act=ACT.SILU, img_size=img_size)
        return YoloV5(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Medium(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV5Main.Medium(num_cls=num_cls, act=ACT.SILU, img_size=img_size)
        return YoloV5(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Large(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV5Main.Large(num_cls=num_cls, act=ACT.SILU, img_size=img_size)
        return YoloV5(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def XLarge(device=None, num_cls=80, img_size=(416, 352), pack=PACK.AUTO):
        backbone = YoloV5Main.XLarge(num_cls=num_cls, act=ACT.SILU, img_size=img_size)
        return YoloV5(backbone=backbone, device=device, pack=pack)

    @staticmethod
    def Const(device=None, num_cls=80, img_size=(416, 352), batch_size=1):
        backbone = YoloV5ConstMain(num_cls=num_cls, batch_size=batch_size, img_size=img_size)
        return YoloV5(backbone=backbone, device=device, pack=PACK.NOOP)


if __name__ == '__main__':
    model = YoloV5Main.Small(img_size=(416, 416))
    # input = torch.zeros(1, 3, 416, 416)
    # y = model(input)
