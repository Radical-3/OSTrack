"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head x是经过全部transformer并且已经排列好顺序且填充好的搜索区域和模板区域的特征 (B,总块数,特征维度) aux_dict存了attn：搜索区域和模板区域的注意力权重，removed_inexes_s:搜索区域删除的index
        feat_last = x  # (1,320,768)
        if isinstance(x, list):
            feat_last = x[-1]  # out中存了score_map:每一个搜索区域的小块是否是中心点的得分(bs,1,16,16) pred_bbox:每一张搜索图的最可能的bbox(中心点得分最大的bbox)(bs,1,4) size_map:每一个搜索块的大小(bs,2,16,16) offset_map:每一个搜索块的偏移量(bs,2,16,16)
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)  # 将aux_dict中的内容添加到out中
        out['backbone_feat'] = x  # 这里out存储了7个键值対，pred_boxes(bs,1,4) 预测的bbox score_map(bs,1,16,16)搜索图中每一块是中心点的得分 size_map(bs,2,16,16)搜索图中的每一块的大小 offset_map(bs,2,16,16) 搜索块中每一块的偏移量 backbone_feat(bs,320,768) 经过全部transformer并且已经排列好顺序且填充好的搜索区域和模板区域的特征 (B,总块数,特征维度) attn:搜索区域和模板区域经过所有transformer的注意力权重 removed_indexes_s:搜索区域删除掉的index
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """  # cat_feature是经过全部transformer的输出 模板块的特征+搜索块排序好且填充好的特征(B,总块数,C)
        enc_opt = cat_feature[:, -self.feat_len_s:]  # 提取搜索块的token  encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()  # .contiguous()确保内存连续 opt(B,1,C,总块数)
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)  # 变成(B,C,总块数开根号，总块数开根号)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head opt_feat：(B,C,总块数开根号，总块数开根号)  gt_score_map：None
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox) score_map_ctr:每一个搜索区域的小块是否是中心点的得分(bs,1,16,16) bbox:每一张搜索图的最可能的bbox(中心点得分最大的bbox)(bs,4) size_map:每一个搜索块的大小(ba,2,16,16) offset_map:每一个搜索块的偏移量(bs,2,16,16)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)  # outputs_coord_new(bs,1,4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


# 加载预训练模型放进骨干模型，和头
# 预训练模型: pretrained_models/mae_pretrain_vit_base.pth
# 骨干模型: vit_base_patch16_224_ce
def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    # box_head可以预测三个值
    # 中心点得分：该位置是中心点的可能性
    # 该位置相对于实际中心点的x和y的偏移量
    # 预测目标的尺寸，宽和高
    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
