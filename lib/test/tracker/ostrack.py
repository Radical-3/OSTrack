import math

from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)  # params存放的搜索和模板区域的大小和相对于bbox的倍数和训练好的模型参数的保存路径
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)  # self.params.checkpoint训练好的模型权重
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE  # self.feat_sz：16
        # motion constrain # self.output_window 二维汗宁窗，不知道有什么用？？ (1,1,16,16)
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once # z_patch_arr：(以bbox为中心放大x倍进行)裁减填充然后缩放到指定self.params.template_size大小的图像(128,128,3)，resize_factor：self.params.template_size/裁减填充后的图像的边长 (128,128) z_amask_arr：裁减填充并缩放到指定大小的矩阵，如果填充了则是true，非填充是false
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)  # 包含mask和tensors，mask是将维度变为了(1,128,128)的tensor tensors是将维度变为了(1,3,128,128)并进行标准化后的tensor
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # template_bbox：原始bbox转换成的裁减填充且缩放后的bbox的坐标
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
            # self.box_mask_z：(1,64) (bs,图像分成的小块的数量) 除了最中心的块为true，其他的块的值都为false
        # save states
        self.state = info['init_bbox']  # 最开始的bbox
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1  # sample_target是対图像进行裁减填充缩放到指定大小x_patch_arr：得到的图片resize_factor：指定大小的图像边长/裁减填充后的图像边长 x_amask_arr：填充的地方为true，其他为false
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)  # 存储mask：转化为(1,256,256)的tensor tensors：转化为(1,3,256,256)的tensor(进行了标准化)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(  # self.z_dict1.tensors第一帧图片(模板图片)的输入到网络之前的图像(1,3,128,128) self.box_mask_z是将模板图片分为小块之后的tensor(1,64)，最中心的为true，其他的为false
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)
            # 这里out_dict存储了7个键值対，pred_boxes(bs,1,4) 预测的bbox score_map(bs,1,16,16)搜索图中每一块是中心点的得分 size_map(bs,2,16,16)搜索图中的每一块的大小 offset_map(bs,2,16,16) 搜索块中每一块的偏移量 backbone_feat(bs,320,768) 经过全部transformer并且已经排列好顺序且填充好的搜索区域和模板区域的特征 (B,总块数,特征维度) attn:搜索区域和模板区域经过所有transformer的注意力权重 removed_indexes_s:搜索区域删除掉的index
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map  # self.output_window(1,1,16,16)二维汉宁窗口 不知道得到的是什么？？？
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)  # pred_boxes是使用上面汗宁窗口乘每一块的中心点得分和大小尺寸预测出来的搜索图像的bbox (1,4) 进行了标准化
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1] 那这里的pred_box就是预测的bbox求出对应原图裁减填充后的预测的bbox
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)  # 这个self.state好像是预测的搜索图像的bbox在原始图像中的坐标 (xmin,ymin,w,h)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]  # 模板帧的原图的bbox的中心坐标
        cx, cy, w, h = pred_box  # 这里好像是中心坐标
        half_side = 0.5 * self.params.search_size / resize_factor  # half_side是裁减填充后的图像的边长的一半，也就是缩放前的边长的一半
        cx_real = cx + (cx_prev - half_side)  # 预测框在搜索区域的中心坐标+搜索区域相对于原始图像的偏移量=预测框在原始图像中的中心坐标
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]  # 最后返回的是中心坐标

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack
