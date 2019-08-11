import tensorflow as tf
from tensorflow.python.framework import graph_util
import cv2
import numpy as np

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    with open('pnet.pb', 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        tf.import_graph_def(output_graph_def, name="")
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        sess.run(tf.global_variables_initializer())
        image_height_tensor = sess.graph.get_tensor_by_name('image_height:0')
        image_width_tensor = sess.graph.get_tensor_by_name('image_width:0')
        input_tensor = sess.graph.get_tensor_by_name('input_image:0')
#         input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
        cls_prob_tensor = sess.graph.get_tensor_by_name('cls_prob:0')    
        bbox_pred_tensor = sess.graph.get_tensor_by_name('bbox_pred:0')
        img = cv2.imread('images/1.png')
        img = np.asarray(img, dtype = np.float32)
        img -= 127.5
        img = img * 1/128
        image_height, image_width, _ = img.shape
        cls_prob, bbox_pred = sess.run([cls_prob_tensor, bbox_pred_tensor], 
                                       feed_dict= {input_tensor: img, image_height_tensor:image_height, image_width_tensor: image_width})
        
        print(cls_prob.shape)
        print(bbox_pred.shape)
        print(cls_prob[0, 0:100, 0])
        print(cls_prob[0, 0:100, 1])
        