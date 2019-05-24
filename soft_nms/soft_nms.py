# --------------------------------------------------------
# soft_nms
# Copyright (c) 2018 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianfeng Yang
# --------------------------------------------------------

import numpy as np
from shapely.geometry import Polygon
def py_cpu_nms(S, sigma, Nt, thresh, method):
    """Pure Python NMS baseline."""
    x1 = S[:, 0]
    y1 = S[:, 1]
    x2 = S[:, 4]
    y2 = S[:, 5]
    scores = S[:, 8]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order = scores.argsort()[::-1]
    #keep = []
 
    order = np.argsort(S[:, 8])[::-1]
    keep = []

    while order.size > 0:
        inds =0
        i = order[inds]
        keep.append(i)
        #order = np.argsort(order[:])[::-1]
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h#交集
        ovr = inter / (areas[i] + areas[order[1:]] - inter)#交并比，iou between max box and detection box
                  
        N=order.shape[0]
        weight= (N-1)*['']
        for n in range(1,N-1):
            if method == 1: # linear
                        if ovr[n] > Nt: 
                            weight[n] = 1 - ovr[n]
                        else:
                            weight[n] = 1
            elif method == 2: # gaussian
                        weight[n] = np.exp(-(ovr[n] * ovr[n])/sigma)
            else: # original NMS
                        if ovrovr[n] > Nt: 
                            weight[n] = 0
                        else:
                            weight[n] = 1

            scores[n] = weight[n]*scores[n]

        #inds = np.where(scores[order[1:]] <= 0.2)[0]
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return S[keep]
