from easydict import EasyDict as edict
import yaml

_C = edict()
config = _C

# ------------------------------------------------------------------------------------- #
# Common options
# ------------------------------------------------------------------------------------- #
_C.RNG_SEED = -1


# ------------------------------------------------------------------------------------- #
# Common options
# ------------------------------------------------------------------------------------- #
_C.DATASET = edict()
_C.DATASET.ROOT_PATH = "/data/datasets/"
_C.DATASET.DATA_PATH = "ActionGenome/"
_C.DATASET.FRAMES_PATH = "dataset/ag/frames"
_C.DATASET.TRAIN_ANNOTATIONS_PATH = "/workspace/ayushk4/Create_stdvc/cache/idxed_train.json"
_C.DATASET.VAL_ANNOTATIONS_PATH = "/workspace/ayushk4/Create_stdvc/cache/idxed_val.json"
_C.DATASET.DATASET_NAME = "unnamed_dataset"

_C.DATASET.TOY = False
_C.DATASET.TOY_SAMPLES = 10

_C.DATASET.SAMPLING_STRIDE = 2
_C.DATASET.SAMPLING_COUNT = 8

_C.DATASET.TRAIN_SPLIT = "train"
_C.DATASET.VAL_SPLIT = "val"




def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in config[k]:
                            if vk == 'LR_STEP':
                                config[k][vk] = tuple(float(s) for s in vv.split(','))
                            elif vk == 'LOSS_LOGGERS':
                                config[k][vk] = [tuple(str(s) for s in vvi.split(',')) for vvi in vv]
                            else:
                                config[k][vk] = vv
                        else:
                            raise ValueError("key {}.{} not in config.py".format(k, vk))
                else:
                    if k == 'SCALES':
                        config[k] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("key {} not in config.py".format(k))
