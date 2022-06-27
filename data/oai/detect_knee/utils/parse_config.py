import os

import numpy as np


def parse_model_cfg(path):
    '''
    Parse the yolo *.cfg file and output module definitions
    '''
    if not path.endswith('.cfg'):  # add .cfg suffix if omitted
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):  # add cfg/ prefix if omitted
        path = 'cfg' + os.sep + path
    # all fields are supported
    supports = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh', 'probability']

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  
    model_definition = []  # module definitions
    for line in lines:
        if line.startswith('['):  
            model_definition.append({})
            model_definition[-1]['type'] = line[1:-1].rstrip()
            if model_definition[-1]['type'] == 'convolutional':
                model_definition[-1]['batch_normalize'] = 0  
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                model_definition[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))  # np anchors
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):  # return array
                model_definition[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():  # return int or float
                    model_definition[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    model_definition[-1][key] = val  # return string

    f = []  # fields
    for x in model_definition[1:]:
        [f.append(k) for k in x if k not in f]
    # check supoort fields
    u = [x for x in f if x not in supports]  
    assert not any(u), "Unsupported fields %s in %s." % (u, path)

    return model_definition


