# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------

import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b):
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b):
    return a if a <= b else b
#np.ndarray[float, ndim=2] boxes
def cpu_soft_nms(S, float sigma, float Nt, float threshold, unsigned int method):
    cdef unsigned int N = S.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov

    for i in range(N):
        maxscore = S[i, 8]
        maxpos = i

        tx1 = S[i, 0]
        ty1 = S[i, 1]
        tx2 = S[i, 4]
        ty2 = S[i, 5]
        ts = S[i, 8]

        pos = i + 1
	# get max box
        while pos < N:
            if maxscore < s[pos, 8]:
                maxscore = s[pos, 8]
                maxpos = pos
            pos = pos + 1

	# add max box as a detection ,S[i,]=score max boxes
      	#S[i,0] = S[maxpos,0]
        #S[i,1] = S[maxpos,1]
        #S[i,2] = S[maxpos,4]
        #S[i,3] = S[maxpos,5]
        #S[i,4] = S[maxpos,8]

	# swap ith box with position of max box,S[maxpos,]=boxes[i],S[i,]=score max boxes=tx1,ty1,tx2.ty2
        #S[maxpos,0] = tx1
        #S[maxpos,1] = ty1
        #S[maxpos,2] = tx2
        #S[maxpos,3] = ty2
        #S[maxpos,4] = ts

        #tx1 = boxes[i,0]
        #ty1 = boxes[i,1]
        #tx2 = boxes[i,2]
        #ty2 = boxes[i,3]
        #ts = boxes[i,4]

        #pos = i + 1
	# NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = S[pos, 0]
            y1 = S[pos, 1]
            x2 = S[pos, 4]
            y2 = S[pos, 5]
            s = S[pos, 8]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1

                    S[pos, 8] = weight*S[pos, 8]
		    
		    # if box score falls below threshold, discard the box by swapping with last box
		    # update N
                    if S[pos, 8] < threshold:
                        S[pos,0] = S[N-1, 0]
                        S[pos,1] = S[N-1, 1]
                        S[pos,4] = S[N-1, 4]
                        S[pos,5] = S[N-1, 5]
                        S[pos,8] = S[N-1, 8]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    #keep = [i for i in range(N)]
    #return keep
     return cpu_nms(np.array(S), thres)


def cpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 4]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 5]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 8]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    keep = []
    for _i in range(ndets):
        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]
        iarea = areas[i]
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr >= thresh:
                suppressed[j] = 1

    return keep
