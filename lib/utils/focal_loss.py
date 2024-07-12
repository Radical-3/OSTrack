from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F


# 这段代码实现了Focal Loss的计算，通过对正负样本的损失进行不同的缩放，解决了类别不平衡的问题。
# Focal Loss通过引入alpha和beta参数，使得模型更关注难分类的样本，从而提高了模型在不平衡数据集上的表现。
# alpha是定义的超参数，用于设置难区分样本的关注度，类比于飞书中的beita
class FocalLoss(nn.Module, ABC):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        # target是一个0和1的张量，1表示正样本，0表示负样本
        # positive_index返回一个和target相同的张量，1则为true，0为false
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()
        # negative_weights应该就是那个阿尔法t
        negative_weights = torch.pow(1 - target, self.beta)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        prediction = torch.clamp(prediction, 1e-12)
        # 正样本的positive_index为1，negative_index为0
        # 负样本的positive_index为0，negative_index为1
        positive_loss = torch.log(prediction) * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) * torch.pow(prediction,
                                                              self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss = -negative_loss
        else:
            loss = -(positive_loss + negative_loss) / num_positive

        return loss


class LBHinge(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """
    def __init__(self, error_metric=nn.MSELoss(), threshold=None, clip=None):
        super().__init__()
        self.error_metric = error_metric
        self.threshold = threshold if threshold is not None else -100
        self.clip = clip

    def forward(self, prediction, label, target_bb=None):
        negative_mask = (label < self.threshold).float()
        positive_mask = (1.0 - negative_mask)

        prediction = negative_mask * F.relu(prediction) + positive_mask * prediction

        loss = self.error_metric(prediction, positive_mask * label)

        if self.clip is not None:
            loss = torch.min(loss, torch.tensor([self.clip], device=loss.device))
        return loss