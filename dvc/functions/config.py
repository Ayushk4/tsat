from easydict import EasyDict as edict
import yaml

_C = edict()
config = _C

# ------------------------------------------------------------------------------------- #
# Common options
# ------------------------------------------------------------------------------------- #
_C.RNG_SEED = -1


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
