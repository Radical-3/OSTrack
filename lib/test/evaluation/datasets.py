from collections import namedtuple
import importlib
from lib.test.evaluation.data import SequenceList

DatasetInfo = namedtuple('DatasetInfo', ['module', 'class_name', 'kwargs'])

pt = "lib.test.evaluation.%sdataset"  # Useful abbreviations to reduce the clutter

dataset_dict = dict(
    otb=DatasetInfo(module=pt % "otb", class_name="OTBDataset", kwargs=dict()),
    nfs=DatasetInfo(module=pt % "nfs", class_name="NFSDataset", kwargs=dict()),
    uav=DatasetInfo(module=pt % "uav", class_name="UAVDataset", kwargs=dict()),
    tc128=DatasetInfo(module=pt % "tc128", class_name="TC128Dataset", kwargs=dict()),
    tc128ce=DatasetInfo(module=pt % "tc128ce", class_name="TC128CEDataset", kwargs=dict()),
    trackingnet=DatasetInfo(module=pt % "trackingnet", class_name="TrackingNetDataset", kwargs=dict()),
    got10k_test=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='test')),
    got10k_val=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='val')),
    got10k_ltrval=DatasetInfo(module=pt % "got10k", class_name="GOT10KDataset", kwargs=dict(split='ltrval')),
    lasot=DatasetInfo(module=pt % "lasot", class_name="LaSOTDataset", kwargs=dict()),
    lasot_lmdb=DatasetInfo(module=pt % "lasot_lmdb", class_name="LaSOTlmdbDataset", kwargs=dict()),

    vot18=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict()),
    vot22=DatasetInfo(module=pt % "vot", class_name="VOTDataset", kwargs=dict(year=22)),
    itb=DatasetInfo(module=pt % "itb", class_name="ITBDataset", kwargs=dict()),
    tnl2k=DatasetInfo(module=pt % "tnl2k", class_name="TNL2kDataset", kwargs=dict()),
    lasot_extension_subset=DatasetInfo(module=pt % "lasotextensionsubset", class_name="LaSOTExtensionSubsetDataset",
                                       kwargs=dict()),
    local_data_test=DatasetInfo(module=pt % "local_data_", class_name="local_data_Dataset", kwargs=dict(split='test')),
    fgsm_data_test=DatasetInfo(module=pt % "local_data_", class_name="local_data_Dataset", kwargs=dict(split='test')),
)


def load_dataset(name: str):
    """ Import and load a single dataset."""
    name = name.lower()
    dset_info = dataset_dict.get(name)  # dset_info:lib.test.evaluation.got10kdataset
    if dset_info is None:
        raise ValueError('Unknown dataset \'%s\'' % name)

    m = importlib.import_module(dset_info.module)
    dataset = getattr(m, dset_info.class_name)(**dset_info.kwargs)  # dataset: got10kdataset类  # Call the constructor
    return dataset.get_sequence_list()  # dataset里面存了base_path:test的位置 env_settings:设置，可能是lib/test/local和别的结合 sequence_list:选取的测试集样本名称的列表


def get_dataset(*args):
    """ Get a single or set of datasets."""
    dset = SequenceList()
    for name in args:
        dset.extend(load_dataset(name))  # dset存的是Sequence的列表(选取的样本) 每一个Sequence存的是选的样本，里面的信息有name：样本名称 frames：样本中每一帧的路径 ground_truth_rect：变成(1,4)的二维数组的bbox init_data：字典(bbox:bbox的值)
    return dset