import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf
import icdar
import find_boxes
import re
import datetime

import soft_nms.nms_wrapper as nms_wrapper
import locality_aware_nms as nms_locality
import lanms

tf.app.flags.DEFINE_string('test_data_path', './tmp/ch4_test_images/images/', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', './tmp/east_icdar2015_resnet_v1_50_rbox/', '')
tf.app.flags.DEFINE_string('output_dir', './tmp/h4_test_images/images/', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

import model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def findContours_func(input_score_image,image):
    min_area = 50
    box_thresh = 0.4
    score_map_thresh = 0.80
    image_shape = image.shape
    w,h = image_shape[0:2]
    socre_h,socre_w= input_score_image.shape[0:2]
    score_image = cv2.resize(input_score_image, (socre_w*4,socre_h*4), interpolation=cv2.INTER_LINEAR) 
    gray=np.asarray(score_image*255,np.uint8)
    ret, imgbw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    img_dilate = find_boxes.dilate_image(np.asarray(imgbw))
    #img_dilate = find_boxes.erode_image(img_dilate)
    img, contours, hierarchy = cv2.findContours(
       img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #contours_res = find_boxes.filter_low_boxes(contours,imgbw)
    print(len(contours))
    boxes_array=np.zeros([len(contours),8])
    theat=np.zeros(len(contours))
    expanded_poly = np.zeros((len(contours), 8), dtype=np.float32)

    r = [None, None, None, None]
    
   # print('contours_res:',len(contours_res),type(contours_res))
    for i,cnt in enumerate(contours):
        #bboxes =[]
        poly_area = cv2.contourArea(cnt)
        #rect = cv2.minAreaRect(cnt)
        rect,rect_area = find_boxes.min_area_rect(cnt)
        #w = rect[2]*4
        #h = rect[3]*4
        #rect_area2 = w * h
        #print(len(rect))
        #if min(w, h) < min_height:
           # continue
        #if poly_area < min_poly_area:
         #  continue
        #if rect_area2 < min_area:
           #continue
        theat[i] = rect[4]
        rect = ((rect[0], rect[1]), (rect[2], rect[3]), rect[4])
        bboxes = cv2.boxPoints(rect)
        #bboxes = np.int0(bboxes)
        #box = find_boxes.dilate_boxes(bboxes,imgbw)
        #bboxes.append(box)
        #print(bboxes.shape[0])
        #bboxess = np.array(bboxes)
        box2 = bboxes.reshape(-1, 8)
        boxes_array[i,:8]=box2[:8]#*4
        poly = boxes_array[i,:8].reshape(4,2)
        d1 = np.linalg.norm(poly[1] - poly[2])
        d2 = np.linalg.norm(poly[2] - poly[3])
        rect_area = d1 *d2
        d_min = min(d1,d2)
       # if  d2 < min_with:
        #   continue
        #if rect_area < min_area:
         #  continue
        for j in range(4):
               r[j] = min(np.linalg.norm(poly[j] - poly[(j + 1) % 4]),
                       np.linalg.norm(poly[j] - poly[(j - 1) % 4]))
                #print(len(r))    
        #expanded_poly = boxes_array[i,:8].reshape(i,4,2)
        #expanded_poly=np.array(expanded_poly)

        boxes_array2= icdar.expand_poly(poly, r,m=min_area,S=rect_area,d_m=d_min).astype(np.int32)[np.newaxis, :, :]
       # print('boxes_array2:',boxes_array2.shape ,type(boxes_array2))
        #boxes_list = list(boxes_array2)

        expanded_poly[i,:8] = boxes_array2.reshape(-1,8)
        #print('expanded_poly:',expanded_poly.shape ,type(expanded_poly))
        #expanded_poly = icdar.expand_poly(poly,d_min)#.astype(np.int32)[np.newaxis, :, :]
        #expanded_poly = expanded_poly.reshape(-1,8)
    fixed_poly = find_boxes.fix_boxes(expanded_poly,image)
    
    xy_text = np.argwhere(score_image > score_map_thresh)
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    fix_boxes = np.zeros((fixed_poly.shape[0], 9), dtype=np.float32)
    fix_boxes[:, :8] = fixed_poly.reshape((-1, 8))
    #fix_boxes[:, 8] = input_score_image[xy_text[:, 0], xy_text[:, 1]]
        
    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(fix_boxes):
        mask = np.zeros_like(score_image, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32), 1)
        fix_boxes[i, 8] = cv2.mean(score_image, mask)[0]
    fix_boxes = fix_boxes[fix_boxes[:, 8] > box_thresh]
    #final_box = find_boxes.filter_blur_boxes(fix_boxes,image)
    #print(len(final_box))
    #return final_box
    return fix_boxes,theat



def detect(score_map, geo_map, timer, score_map_thresh=0.85, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1]*4, geo_map[xy_text[:, 0], xy_text[:, 1], :]) # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    #boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list


    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        f_score, f_geometry = model.model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
            #start_totoltime = time.time()
            im_fn_list = get_images()
            for im_fn in im_fn_list:
                im = cv2.imread(im_fn)[:, :, ::-1]
                start_time = time.time()
                im_resized, (ratio_h, ratio_w) = resize_image(im)

                timer = {'net': 0, 'restore': 0, 'nms': 0}
                start = time.time()
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                timer['net'] = time.time() - start

                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                boxes1,theat=findContours_func(score.reshape(score.shape[1],score.shape[2]),im)#,im_resized
                if boxes is not None and boxes1.size:
                 #   if len(boxes) != 0:
                 
                    boxes =np.append(boxes,boxes1,axis = 0)
                #boxes = nms_locality.standard_nms(boxes.astype(np.float64), nms_thres=0.2)
                    nms_thres=0.2
                    #boxes = nms_wrapper.soft_nms(boxes.astype(np.float64), nms_thres)
                    boxes = nms_locality.standard_nms(boxes.astype(np.float64), nms_thres)
                #boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
                #boxes = findContours_func(score.reshape(score.shape[1],score.shape[2]))
               
                
                
                print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                    im_fn, timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h

                duration = time.time() - start_time

                #totol_time = 0 
               # fduration = float(duration)
                #f_duration = round(fduration ,[3])
                #start_totoltime = start_totoltime + f_duration
                print('[timing] {}'.format(duration))
                #date1 = datetime.datetime.strptime(duration,"%H:%M:%S")
                #totol_time = totol_time + date1
                #print('[totol_timing] {}'.format(totol_time))
                # save to file
                if boxes is not None:
                    s=os.path.basename(im_fn).split('.')[0]
                    s=re.sub("\D", "", s)
                    res_file = os.path.join(
                        FLAGS.output_dir,
                        'res_img_{}.txt'.format(
                            s))#os.path.basename(im_fn).split('.')[0])

                    with open(res_file, 'w') as f:
                        for box in boxes:
                            # to avoid submitting errors
                            box = sort_poly(box.astype(np.int32))
                            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                                continue
                            
                            if box[0, 0]>box[2, 0] or box[0, 1]>box[2, 1]:
                                continue
                            #jungle the rangle whether is colockrise 
                            #if ((box[1, 0] - box[0, 0])*(box[2, 1]-box[0, 1])-(box[2, 0] - box[0, 0])*(box[1, 1]-box[0, 1])) < 0:
                             #   continue
                            #else:
                                
                                #box[1],box[3] = box[3],box[1]
                             #   box[1, 0],box[3, 0] = box[3, 0],box[1, 0]
                           
                              #  box[1, 1],box[3, 1] = box[3, 1], box[1, 1]
                            f.write('{},{},{},{}\r\n'.format(
                                box[0, 0], box[0, 1], box[2, 0], box[2, 1],
                            ))
                            cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=2)
                
                if not FLAGS.no_write_images:
                    img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
                    cv2.imwrite(img_path, im[:, :, ::-1])

if __name__ == '__main__':
    tf.app.run()
