import tensorflow as tf
import numpy as np
import cv2

PIXEL_CLS_WEIGHT_all_ones = 'PIXEL_CLS_WEIGHT_all_ones' 
PIXEL_CLS_WEIGHT_bbox_balanced = 'PIXEL_CLS_WEIGHT_bbox_balanced'
PIXEL_NEIGHBOUR_TYPE_4 = 'PIXEL_NEIGHBOUR_TYPE_4'
PIXEL_NEIGHBOUR_TYPE_8 = 'PIXEL_NEIGHBOUR_TYPE_8'

DECODE_METHOD_join = 'DECODE_METHOD_join'


#形态学处理 
def dilate_image(img, kernel_size=(2, 2), iter_size=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    img_dilate = cv2.dilate(img, kernel, iter_size)
    return img_dilate

def erode_image(img, kernel_size=(3, 3), iter_size=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    img_erode = cv2.erode(img, kernel, iter_size)
    return img_erode

def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta]. 
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return box, w * h

def rect_to_xys(rect, image):
    """Convert rect to xys, i.e., eight points
    The `image_shape` is used to to make sure all points return are valid, i.e., within image area
    """
    image_shape = image.shape
    h, w = image_shape[0:2]
    def get_valid_x(x):
        if x < 0:
            return 1
        if x >= w:
            return w - 1
        return x
    
    def get_valid_y(y):
        if y < 0:
            return 1
        if y >= h:
            return h - 1
        return y
    
    #rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
    points = cv2.boxPoints(rect)
    points = np.int0(points)
    '''for i_xy, (x, y) in enumerate(points):
        x = get_valid_x(x)
        y = get_valid_y(y)
        points[i_xy, :] = [x, y]'''
    points = np.reshape(points, -1)
    return points

def fix_boxes(boxes,img):
 #预测顶点修正
    #height_score,width_score=score.shape[1:3]
    #height_score-=1
    #width_score-=1
    #coor_mask=np.zeros_like(score)
    min_area = 300
    min_height = 10
    img_shape = img.shape
    #print(img.shape)
    h,w = img_shape[0:2]
    fix_boxess = np.zeros((boxes.shape[0], 8), dtype=np.float32)
    def get_valid_x(x):
        if x < 0:
            return 0
        if x >= w:
            return w - 1
        return x
    
    def get_valid_y(y):
        if y < 0 :
            return 1
        if y >= h:
            return h - 1
        return y
    
    
    if boxes is not None:
        for i,box in enumerate(boxes):
            x1,y1,x2,y2,x3,y3,x4,y4=np.asarray(box[:8],np.int32)
            if x1 > x2 and y2 > y3:
                xx,yy = x2 , y2
                x2,y2 = x4 , y4
                x4,y4 = xx , yy

            x1=get_valid_x(x1)
            x2=get_valid_x(x2)
            x3=get_valid_x(x3)
            x4=get_valid_x(x4)
            
            y1=get_valid_y(y1)
            y2=get_valid_y(y2)
            y3=get_valid_y(y3)
            y4=get_valid_y(y4)

            
            polys = boxes[i,:8].reshape(4,2)
            d1 = np.linalg.norm(polys[1] - polys[2])
            d2 = np.linalg.norm(polys[2] - polys[3])
            #d1 = abs(x1 - x2)
            #d2 = abs(y2 - y3)
            
            
            rect_area = d1 * d2
            if min(d1, d2) < min_height:
               continue
            if rect_area < min_area:
               continue
            
            
            fix_boxess[i,:8] = x1,y1,x2,y2,x3,y3,x4,y4
    return fix_boxess


def filter_low_boxes(contours,score_map):
    score_map_thresh = 0.8
    sum = 0
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
   # print('contours:',contours,len(contours),type(contours))
    # filter the score map
    contours_filter = []
    xy_text = np.argwhere(score_map > score_map_thresh)
    for i,cnt in enumerate(contours):
       # print('cnt:',cnt.shape,type(cnt))
        for n ,(w,h) in enumerate(xy_text):
            #print(xy.shape,type(xy))
            res_filter = cv2.pointPolygonTest(cnt,(w,h),False)
            if res_filter < 0:
                res_filter = 0
            else:
                res_filter = res_filter
            sum += res_filter

        if sum == 0 :   
            continue
        else:
            contours_filter.append(cnt)
        
   # contours_filter = np.array(contours_filter)
    return contours_filter

def filter_blur_boxes(in_boxes,res_img):
    img2gray = cv2.cvtColor(res_img,cv2.COLOR_BGR2GRAY)
    get_boxes = np.zeros((in_boxes.shape[0], 9), dtype=np.float32)
      
    #boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    for i ,box in enumerate(in_boxes):
        point_box=box[:8].reshape(-1.4,2)
        #point1 = box[:2]
        #point2 = box[4:6]
        #poly = (point1,point2)
        Xs = [i[0] for i in point_box]
        Ys = [i[1] for i in point_box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        crop_img = img2gray[y1:y1+hight, x1:x1+width]
        #crop_img = img2gray.crop(poly)
        score = cv2.Laplacian(crop_img,cv2.CV_64F).var()
        if  score < 20:
            continue
        get_boxes[i,:9] = box[:9]
		
		
    return get_boxes

def dilate_boxes(in_boxes,score_img):
    #img2gray = cv2.cvtColor(res_img,cv2.COLOR_BGR2GRAY)
    get_boxes = np.zeros((in_boxes.shape[0], 8), dtype=np.float32) 
    #boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    for i ,box in enumerate(in_boxes):
        #point1 = box[:2]
        #point2 = box[4:6]
        #poly = (point1,point2)
        Xs = [i[0] for i in in_boxes]                                                                     
        Ys = [i[1] for i in in_boxes]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        crop_img = score_img[y1:y1+hight, x1:x1+width]
        dilate_crop_img = dilate_image(np.asarray(crop_img))
        img, contours, hierarchy = cv2.findContours(dilate_crop_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        for j,cnt in enumerate(contours):
        #bboxes =[]
        #poly_area2 = cv2.contourArea(cnt)
            rect2 = cv2.minAreaRect(cnt)
        # rect,rect_area = find_boxes.min_area_rect(cnt)
            bboxes2 = cv2.boxPoints(rect2)	
           # get_boxes[i,:8] = bboxes2[:8]
		
    return bboxes2













