# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------

from soft_nms.config import cfg
#from soft_nms.gpu_nms import gpu_nms
from soft_nms.soft_nms import py_cpu_nms
import numpy as np


def soft_nms(polys, sigma=0.8, Nt=0.3, threshold=0.3, method=2):

    keep = py_cpu_nms(np.ascontiguousarray(polys, dtype=np.float64),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))

    if len(polys) == 0:
        return np.array([])
    return keep

# Original NMS implementation
#def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
 #   if dets.shape[0] == 0:
  #      return []
    #if cfg.USE_GPU_NMS and not force_cpu:
        #return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
   # else:
    #    return cpu_nms(dets, thresh)

