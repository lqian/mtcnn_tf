import argparse
import os
import cv2
import numpy as np
# from create_tf_record import *
from training.mtcnn_config import config
from training.mtcnn_model import P_Net, R_Net, O_Net
import tensorflow as tf
from tensorflow.python import graph_util
from training.mtcnn_config import net_name_dict


def test_pnet(PNet, model_path):
    graph = tf.Graph()
    with graph.as_default():
        # define tensor and op in graph(-1,1)
        image_op = tf.placeholder(tf.float32, name='input_image')
        width_op = tf.placeholder(tf.int32, name='image_width')
        height_op = tf.placeholder(tf.int32, name='image_height')
        image_reshape = tf.reshape(image_op, [1, height_op, width_op, 3])
        # cls_prob batch*2
        # bbox_pred batch*4
        # construct model here
        # cls_prob, bbox_pred = net_factory(image_reshape, training=False)
        # contains landmark
        
        conv1, conv2, conv3 = PNet(image_reshape, training=False)
#         cls_prob, bbox_pred, _ = net_factory(image_reshape, training=False)   
        # allow 
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
        saver = tf.train.Saver()
        # check whether the dictionary is valid
        model_dict = '/'.join(model_path.split('/')[:-1])
        ckpt = tf.train.get_checkpoint_state(model_dict)
        readstate = ckpt and ckpt.model_checkpoint_path
        assert  readstate, "the params dictionary is not valid"
        print("Restore param from: ", model_path)
        saver.restore(sess, model_path)
    
        for op in graph.get_operations():
            print(op.name, op.values())
        
        img = cv2.imread('images/std-pnet.jpg')
        img = np.asarray(img, dtype = np.float32)
        img -= 127.5
        img = img * 1/128
        image_height, image_width, _ = img.shape
        conv1, conv2, conv3 = sess.run([conv1, conv2, conv3], 
                                       feed_dict= {image_op: img, height_op:image_height, width_op: image_width})
        print(conv1)
        print(conv2)
        print(conv3)
        
if __name__ == '__main__':
    modelPath = os.path.join(config.ROOT_PATH, 'tmp/model/', 'pnet')
    modelPath = os.path.join(modelPath, "%s-%d"%('pnet', 30))
    test_pnet(PNet = P_Net, model_path = modelPath)