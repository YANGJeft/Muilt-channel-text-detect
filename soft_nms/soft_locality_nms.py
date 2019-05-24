import numpy as np
from shapely.geometry import Polygon


def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0

    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union


def weighted_merge(g, p):
    g[:8] = (g[8] * g[:8] + p[8] * p[:8])/(g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres):
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds+1]

    return S[keep]


def weighted_scores(S1, sigma, Nt, method):
    orders = np.argsort(S1[:, 8])[::-1]
    keeps = []
    x1 = S1[:, 0]
    y1 = S1[:, 1]
    x2 = S1[:, 4]
    y2 = S1[:, 5]
    scores = S1[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    orders=np.array(orders)
    m = orders[0]
    while orders.size > 0:
        #m = orders[0]
        #keeps.append(m)

        xx1 = np.maximum(x1[m], x1[orders[1:]])
        yy1 = np.maximum(y1[m], y1[orders[1:]])
        xx2 = np.minimum(x2[m], x2[orders[1:]])
        yy2 = np.minimum(y2[m], y2[orders[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inters = w * h#交集
        ovr = inters / (areas[m] + areas[orders[1:]] - inters)#交并比，iou between max box and detection box
                  
        N=orders.shape[0]
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
                        if ovr[n] > Nt: 
                            weight[n] = 0
                        else:
                            weight[n] = 1

            scores[n] = weight[n]*scores[n]
      #inds = np.where(ovr <= Nt)[0]
        indss=0
        orders = orders[indss+1]
    return S1[keeps]



def nms_locality(polys, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []
    p = None
    ploys=weighted_scores(np.array(polys),sigma=0.5,Nt=0.3,method=1)
    for g in polys:
        if p is not None and intersection(g, p) > thres:
            p = weighted_merge(g, p)
        else:
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)

    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(polys), thres)


if __name__ == '__main__':
    # 343,350,448,135,474,143,369,359
    print(Polygon(np.array([[343, 350], [448, 135],
                            [474, 143], [369, 359]])).area)
