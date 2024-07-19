from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.ostrack.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()  # 这里好像是lib/config.ostrack/config.py中的值
    prj_dir = env_settings().prj_dir  # prj_dir：'/home/he/project_code/OSTrack'
    save_dir = env_settings().save_dir  # save_dir：'/home/he/project_code/OSTrack/output'
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/ostrack/%s.yaml' % yaml_name)  # yaml的路径
    update_config_from_file(yaml_file)  # 从yaml中更新config的信息 yaml有的就变成yaml的值，没有的就是config
    params.cfg = cfg  # 将更新后的配置信息赋值给params.cfg
    print("test config: ", cfg)

    # template and search region  # 搜索区域的大小256*256相对于bbox的4倍 模板区域128*128 2倍
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path  # 读取的模型权重'/home/he/project_code/OSTrack/output/checkpoints/train/ostrack/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar'
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/ostrack/%s/OSTrack_ep%04d.pth.tar" %
                                     (yaml_name, cfg.TEST.EPOCH))

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
