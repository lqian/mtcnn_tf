# coding: utf-8
import os
import math
from os.path import join, exists
import cv2
import numpy as np
import random
import sys
import argparse
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
from tools.common_utils import getBboxLandmarkFromTxt, IoU, BBox
from tools.landmark_utils import rotate,flip, show_landmark
from training.mtcnn_config import config

def gen_landmark_data(srcTxt, net, augment=False):
    '''
    srcTxt: each line is: 0=path, 1-4=bbox, 5-12=landmark 4points
    net: PNet or RNet or ONet
    augment: if enable data augmentation
    '''
    print(">>>>>> Start landmark data create...Stage: %s"%(net))
    srcTxt = os.path.join(rootPath, srcTxt)
    saveFolder = os.path.join(config.ROOT_PATH, "tmp/data/%s/"%(net))
    saveImagesFolder = os.path.join(saveFolder, "landmark")
    
    if net not in config.CV_RESIZE_OF_NET:
        raise Exception("The net type error!")
    if not os.path.isdir(saveImagesFolder):
        os.makedirs(saveImagesFolder)
    saveF = open(join(saveFolder, "landmark.txt"), 'w')
    imageCnt = 0
    cv_size = tuple(config.CV_RESIZE_OF_NET[net])
    # image_path bbox landmark(5*2)
    for (imgPath, bbox, landmarkGt) in getBboxLandmarkFromTxt(srcTxt):
        F_imgs = []
        F_landmarks = []        
        if (bbox is None or landmarkGt is None):
            print("ignore sample: ", imgPath)
            continue
            
#         img = cv2.imread(imgPath)
        if not exists(imgPath):
            print("not exists file:", imgPath)
            continue

        img = cv2.imread(imgPath)
        assert(img is not None)
        img_h, img_w, img_c = img.shape
        gt_box = np.array([bbox.left, bbox.top, bbox.right, bbox.bottom])
        f_face = img[bbox.top: bbox.bottom+1, bbox.left: bbox.right+1]
        f_face = cv2.resize(f_face, cv_size)
        landmark = np.zeros((4, 2))        
        #normalize with bbox x, y, w, h
        for index, one in enumerate(landmarkGt):
#             rv = ((one[0]-gt_box[0])/(gt_box[2]-gt_box[0]), (one[1]-gt_box[1])/(gt_box[3]-gt_box[1]))
            rv = ((one[0] - bbox.x)/bbox.w, (one[1] - bbox.y)/bbox.h)
            landmark[index] = rv
        F_imgs.append(f_face)
        F_landmarks.append(landmark.reshape(8))
        landmark = np.zeros((4, 2))        
        if augment:
            x1, y1, x2, y2 = gt_box
            #gt's width
            gt_w = x2 - x1 + 1
            #gt's height
            gt_h = y2 - y1 + 1        
            if max(gt_w, gt_h) < 40 or x1 < 0 or y1 < 0:
                continue
            #random shift
            for i in range(10):
                bbox_size_w = np.random.randint(int(gt_w * 0.8), np.ceil(1.25 * gt_w))
                bbox_size_h = np.random.randint(int(gt_h * 0.8), np.ceil(1.25 * gt_h))
#                 bbox_size = np.random.randint(int(min(gt_w, gt_h) * 0.8), np.ceil(1.25 * max(gt_w, gt_h)))
                delta_x = np.random.randint(-gt_w * 0.2, gt_w * 0.2)
                delta_y = np.random.randint(-gt_h * 0.2, gt_h * 0.2)
                nx1 = int(max(x1+gt_w/2-bbox_size_w/2+delta_x,0))
                ny1 = int(max(y1+gt_h/2-bbox_size_h/2+delta_y,0))
                
                nx2 = int(nx1 + bbox_size_w)
                ny2 = int(ny1 + bbox_size_h)
                if nx2 > img_w or ny2 > img_h:
                    continue
                crop_box = np.array([nx1,ny1,nx2,ny2])
                cropped_im = img[ny1:ny2+1,nx1:nx2+1,:]
                resized_im = cv2.resize(cropped_im, cv_size)
                #cal iou
                iou = IoU(crop_box, np.expand_dims(gt_box,0))
                if iou <= 0.65:
                    continue
                F_imgs.append(resized_im)
                #normalize
                for index, one in enumerate(landmarkGt):
                    rv = ((one[0]-nx1)/bbox_size_w, (one[1]-ny1)/ bbox_size_h)
                    landmark[index] = rv
                F_landmarks.append(landmark.reshape(8))
                landmark = np.zeros((4, 2))
                landmark_ = F_landmarks[-1].reshape(-1,2)
                bbox = BBox([nx1,ny1,nx2,ny2])                    

                #mirror                    
                if random.choice([0,1]) > 0:
                    face_flipped, landmark_flipped = flip(resized_im, landmark_)
                    face_flipped = cv2.resize(face_flipped, cv_size)
                    #c*h*w
                    F_imgs.append(face_flipped)
                    F_landmarks.append(landmark_flipped.reshape(8))
                #rotate
                if random.choice([0,1]) > 0:
                    face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                     bbox.reprojectLandmark(landmark_), 4)#逆时针旋转
                    #landmark_offset
                    landmark_rotated = bbox.projectLandmark(landmark_rotated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, cv_size)
                    F_imgs.append(face_rotated_by_alpha)
                    F_landmarks.append(landmark_rotated.reshape(8))
                
                    #flip
                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
#                     show_landmark(face_flipped, landmark_flipped)
                    face_flipped = cv2.resize(face_flipped, cv_size)
                    
                    F_imgs.append(face_flipped)
                    F_landmarks.append(landmark_flipped.reshape(8))                
                
                #inverse clockwise rotation
                if random.choice([0,1]) > 0: 
                    face_rotated_by_alpha, landmark_rotated = rotate(img, bbox, \
                                                                     bbox.reprojectLandmark(landmark_), -4)#顺时针旋转
                    landmark_rotated = bbox.projectLandmark(landmark_rotated)
                    face_rotated_by_alpha = cv2.resize(face_rotated_by_alpha, cv_size)
                    F_imgs.append(face_rotated_by_alpha)
                    F_landmarks.append(landmark_rotated.reshape(8))
                
                    face_flipped, landmark_flipped = flip(face_rotated_by_alpha, landmark_rotated)
                    face_flipped = cv2.resize(face_flipped, cv_size);
                    F_imgs.append(face_flipped)
                    F_landmarks.append(landmark_flipped.reshape(8)) 
        F_imgs, F_landmarks = np.asarray(F_imgs), np.asarray(F_landmarks)
        for i in range(len(F_imgs)):
            path = os.path.join(saveImagesFolder, "%d.jpg"%(imageCnt))
            cv2.imwrite(path, F_imgs[i])
            landmarks = map(str, list(F_landmarks[i]))
            saveF.write(path + " -2 " + " ".join(landmarks)+"\n")
            imageCnt += 1
        printStr = "\rCount: {}".format(imageCnt)
        sys.stdout.write(printStr)
        sys.stdout.flush()
    saveF.close()
    print ("\nLandmark create done!")

def parse_args():
    parser = argparse.ArgumentParser(description='Create hard bbox sample...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be pnet, rnet, onet',
                        default='unknow', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    stage = args.stage
    if stage not in ['pnet', 'rnet', 'onet']:
        raise Exception("Please specify stage by --stage=pnet or rnet or onet")
    # augment: data augmentation
    gen_landmark_data("dataset/labels.txt", stage, augment=True)

