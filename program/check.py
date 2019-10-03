import numpy as np
import sys
import json

sys.path.append('..')
import config


if __name__ == '__main__':
    f = open('nan_images', 'w')
    for idx in config.TRAIN_SET:
        arr = np.load(config.DATA_DIR + 'uv_' + idx + '.npy')
        if np.any(np.isnan(arr)):
            f.write(idx + '\n')
            nan_pos = np.argwhere(np.isnan(arr))
            str = json.dumps(nan_pos.tolist())
            f.write(str + '\n')
    f.close()
