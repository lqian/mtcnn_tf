#coding:utf-8

import sys, os
import argparse
import numpy as np
#from email.policy import default
import cv2

def parse_argument(argv):
    parser = argparse.ArgumentParser() 
    parser.add_argument("--source_dataset", type=str)
    parser.add_argument("--genlist", type=str, default="labels.txt")
    parser.add_argument("--aligment", type=bool, default=True)
    parser.add_argument("--auto_box", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    return parser.parse_args(argv)

def aligment(labels, img_height, img_width, keep_aspect_ration=True):
    maxx = max(labels[4::2])
    maxy = max(labels[5::2])
    minx = min(labels[4::2])
    miny = min(labels[5::2])
    
    width = maxx - minx
    height = maxy - miny 
    aspectRatio = height / width
    ## always make aspect ratio as 0.5
    if keep_aspect_ration:
        std_size = max(width + width/8, height*2 + height/4)
        dx = (std_size - width ) / 2 + 1
        dy = (std_size/2 - height ) / 2 + 1
    else:    
        dx = int( width / 16) + 1
        if aspectRatio <= 0.5 :
            dy = (width / 2 + 2*dx - height) / 2 + 1
        else:
            dy =  (width + 2*dx - height) / 2 + 1 
        
    minx -= dx
    miny -= dy
    maxx += dx
    maxy += dy
    if (minx < 0):
         minx = 0
    
    if (miny < 0) :
        miny = 0
    if (maxx >= img_width):
        maxx = img_width -1    
    if (maxy >= img_height):
        maxy = img_height - 1
        
    labels[0] = minx
    labels[1] = miny
    labels[2] = maxx
    labels[3] = maxy
    return labels        

def merge_gen_list(source, listfile, debug, auto_box=False):
    meta = open(listfile, 'a+')
    label_files = os.listdir(os.path.join(source, 'mtcnn'))
    for label_file in label_files:
        full_name = os.path.join(source, 'mtcnn', label_file)
        base_name = os.path.splitext(label_file)[0]
        print(full_name) 
        f = open(full_name, 'r') 
        lines = f.readlines() ## ignore the first line
        head = lines[0].split(',')
        img_width = int(head[0])
        img_height = int(head[1])
        lines = lines[1:]
        if (len(lines) > 1):
            print(base_name, len(lines))
        full_name = os.path.join(source, 'JPEGImages', base_name + '.jpg')
        if os.path.exists(full_name):    
            meta.write(full_name);
        
        img = None
        if (debug):
            img = cv2.imread(os.path.join(source, 'JPEGImages', base_name + '.jpg'))
            
        for line in lines: 
            labels = line.split(',')[1:]
            labels = np.array(labels, dtype=np.float) 
            labels = np.array(labels, dtype = np.int32)
            labels = aligment(labels, img_height, img_width, not auto_box) 
            meta.write(" {} {} {} {} {} {} {} {} {} {} {} {}".format( 
                labels[0], 
                labels[1], 
                labels[2], 
                labels[3],
                labels[4],
                labels[5],
                labels[6],
                labels[7],
                labels[8],
                labels[9],
                labels[10],
                labels[11])) 
        meta.write("\n")    
        f.close()
    print("-------------- done for: ", source)    
    meta.close()
    return
    
if __name__ == '__main__':
    if (os.path.exists("labels.txt")):
        os.remove("labels.txt")
        print("remove existed labels.txt")
        
    datasources = [
        "/train-data/plate-detection-dataset/pr-plate/parking-enteral/train/",
        "/train-data/plate-detection-dataset/pr-plate/parking-enteral/val/", 
        "/train-data/plate-detection-dataset/pr-plate/parking-enteral/floor/",
        "/train-data/plate-detection-dataset/pr-plate/parking-enteral/light/",
        "/train-data/plate-detection-dataset/pr-plate/xny-3/",
        "/train-data/plate-detection-dataset/pr-plate/xny-4/",
        "/train-data/plate-detection-dataset/pr-plate/2018-12-07-mixed/",
        
    ]
    for datasource in datasources:
        merge_gen_list(datasource, "labels.txt", False)   
    
