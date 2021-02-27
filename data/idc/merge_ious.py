import json
import numpy as np
def iou(bb1, bb2):
    # x1, y1, x2, y2
    dict_to_list = lambda bb: [bb['x'], bb['y'], bb['x'] + bb['width'], bb['y'] + bb['height']]
    if type(bb1) == dict:
        bb1 = dict_to_list(bb1)
    if type(bb2) == dict:
        bb2 = dict_to_list(bb2)

    bb1 = {'x1': bb1[0], 'x2': bb1[2], 'y1': bb1[1], 'y2': bb1[3]}
    bb2 = {'x1': bb2[0], 'x2': bb2[2], 'y1': bb2[1], 'y2': bb2[3]}
    
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def merge_dataset_ious(savepath, loadpath):

    dataset = json.load(open(loadpath))

    cnt = 0
    for k in dataset:
        flag = True
        this_caps = [[c] for c in dataset[k]['caps']]
        while flag:
            flag = False
            # Find all pair ious
            ious = np.array([[max([iou(ck[2], cl[2]) for ck in ci for cl in cj]) if i != j else 0
                for i, ci in enumerate(this_caps)]
            for j, cj in enumerate(this_caps)])
            
            argmax_iou, max_iou = np.argmax(ious), np.max(ious)
            n = len(this_caps)
            argmax_i, argmax_j = int(argmax_iou/n), argmax_iou % n
            assert argmax_i != argmax_j
            if max_iou >= 0.7:
                caps_j = this_caps.pop(argmax_j)
                this_caps[argmax_i] += caps_j
                flag = True
        print(argmax_iou, max_iou, cnt)
        cnt += 1
        dataset[k]['caps'] = this_caps

    print("Done:", loadpath)
    json.dump(dataset, open(savepath, 'w+'))

merge_dataset_ious("preprocessed/test.json", "preprocessed/test_1.json")
merge_dataset_ious("preprocessed/val.json", "preprocessed/val_1.json")