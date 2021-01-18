from easydict import EasyDict as edict
import yaml

_C = edict()
config = _C

# ------------------------------------------------------------------------------------- #
# Common options
# ------------------------------------------------------------------------------------- #
_C.RNG_SEED = 5

_C.NUM_WORKERS_PER_GPU = 1
_C.GPUS = [0, 1, 2, 3, 4, 5, 6, 7]
_C.RUN = ""
_C.OUTPUT_PATH = ""
_C.MODEL = ""
_C.TASK_TYPE = "Classification"
_C.PROJECT = "stc"

# ------------------------------------------------------------------------------------- #
# Train options
# ------------------------------------------------------------------------------------- #
_C.TRAIN = edict()
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.SHUFFLE = True
_C.TRAIN.LR = -10
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.01
_C.TRAIN.OPTIMIZER = 'AdamW'
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 10
# _C.TRAIN.CLIP_GRAD_NORM = 5.0
# _C.TRAIN.GRADIENT_ACCUMULATE_STEPS = 1

# ------------------------------------------------------------------------------------- #
# Val options
# ------------------------------------------------------------------------------------- #
_C.VAL = edict()
_C.VAL.BATCH_SIZE = 2
_C.VAL.SHUFFLE = False


# ------------------------------------------------------------------------------------- #
# Dataset options
# ------------------------------------------------------------------------------------- #
_C.DATASET = edict()
_C.DATASET.DATA_PATH = ""
_C.DATASET.DATASET_NAME = ""

_C.DATASET.CLASS_TO_INDEX_FILE = "" # Relative to DATASET.DATA_PATH
_C.DATASET.TRAIN_ANNOTATIONS_PATH = "" # Relative to DATASET.DATA_PATH
_C.DATASET.VAL_ANNOTATIONS_PATH = "" # Relative to DATASET.DATA_PATH

_C.DATASET.TRAIN_VIDEO_RESIZED_PATH = "" # Relative to DATASET.DATA_PATH
_C.DATASET.VAL_VIDEO_RESIZED_PATH = "" # Relative to DATASET.DATA_PATH

_C.DATASET.TOY = False
_C.DATASET.TOY_SAMPLES = 10

_C.DATASET.SAMPLING_STRIDE = 2

_C.DATASET.TRAIN_SPLIT = "train"
_C.DATASET.VAL_SPLIT = "val"


# ------------------------------------------------------------------------------------- #
# Network options
# ------------------------------------------------------------------------------------- #

_C.NETWORK = edict()
_C.NETWORK.BACKBONE = ""
_C.NETWORK.BACKBONE_LOAD_PRETRAINED = True
_C.NETWORK.PRETRAINED_MODEL = ''

_C.NETWORK.TEMPORAL_MLP_DIMS = 512
_C.NETWORK.TEMPORAL_MLP_ACTIVATION = "" # The code will prepend `torch.nn.` and do eval over the string.

_C.NETWORK.PASS_SPATIAL_TO_TRANSFORMER = True

_C.NETWORK.TRANSFORMER_DIMS = 512
_C.NETWORK.TRANSFORMER_HEADS = 8
_C.NETWORK.TRANSFORMER_ENCODER_CNT = 8
_C.NETWORK.TRANSFORMER_DROPOUT = 0.1
_C.NETWORK.TRANSFORMER_FEEDFORWARD_DIMS = 2048

_C.NETWORK.POSITIONAL_DROPOUT = 0.1
_C.NETWORK.NUM_CLASSES = 700

_C.NETWORK.PARTIAL_PRETRAIN = False


# ------------------------------------------------------------------------------------- #
# Update Config
# ------------------------------------------------------------------------------------- #


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
