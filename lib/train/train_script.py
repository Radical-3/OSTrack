import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.ostrack import build_ostrack
# forward propagation related
from lib.train.actors import OSTrackActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss


def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'
    # 1.参数更新：读取到cfg的值，从experiments中更新cfg的值，从cfg更新settings的值
    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    # config_module是lib/config/config 存储模型参数
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    # cfg是模型参数  lib/config/config
    cfg = config_module.cfg
    # 如果experiments/ostrack中的参数如果存在在cfg中，则更新为experiments中的值，否则就是lib/config/config中cfg的值
    # setting.cfg_file:'/home/he/project_code/OSTrack/experiments/ostrack/vitb_256_mae_ce_32x4_ep300.yaml'
    config_module.update_config_from_file(settings.cfg_file)
    # 展示参数
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    # 从cfg中更新settings的参数值
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))
    # 2.构建dataloader
    # Build dataloaders
    loader_train, loader_val = build_dataloaders(cfg, settings)

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir

    # 3.创建网络
    # Create network
    if settings.script_name == "ostrack":
        net = build_ostrack(cfg)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")

    # Loss functions and Actors
    if settings.script_name == "ostrack":
        # 4.定义loss
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss, 'cls': BCEWithLogitsLoss()}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 1., 'cls': 1.0}
        # 5.定义对象，包含网络，损失函数，损失函数权重，设置，cfg
        actor = OSTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    # 训练函数
    trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    # 开始训练
    trainer.train(cfg.TRAIN.EPOCH, load_latest=False, fail_safe=True)  # 官方赋值load_latest=True
