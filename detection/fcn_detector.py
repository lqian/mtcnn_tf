import numpy as np
import tensorflow as tf
import sys, os
rootPath = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))
sys.path.insert(0, rootPath)
from training.mtcnn_config import config
from tensorflow.python import graph_util

class FcnDetector(object):
    #net_factory: which net
    #model_path: where the params'file is
    def __init__(self, net_factory, model_path, test_pb=False, export=False):
        if test_pb:
            with tf.Graph().as_default():
                output_graph_def = tf.GraphDef()
                with open('pnet.pb', 'rb') as f:
                    output_graph_def.ParseFromString(f.read())
                    tf.import_graph_def(output_graph_def, name="")
                self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
                self.sess.run(tf.global_variables_initializer())
                self.height_op = self.sess.graph.get_tensor_by_name('image_height:0')
                self.width_op = self.sess.graph.get_tensor_by_name('image_width:0')
                self.image_op = self.sess.graph.get_tensor_by_name('input_image:0')
#                    input_is_training_tensor = sess.graph.get_tensor_by_name("is_training:0")
                self.cls_prob = self.sess.graph.get_tensor_by_name('cls_prob:0')    
                self.bbox_pred = self.sess.graph.get_tensor_by_name('bbox_pred:0')
            return
        
        #create a graph
        graph = tf.Graph()
        with graph.as_default():
            #define tensor and op in graph(-1,1)
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            input_image = tf.placeholder(tf.float32, shape=[1, 12, 24, 3], name='input_image')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            #self.cls_prob batch*2
            #self.bbox_pred batch*4
            #construct model here
            #self.cls_prob, self.bbox_pred = net_factory(image_reshape, training=False)
            #contains landmark
            
            self.cls_prob, self.bbox_pred, _ = net_factory(input_image if export else image_reshape, training=False)
#             self.cls_prob, self.bbox_pred, _ = net_factory(image_reshape, training=False)   
            #allow 
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print("Restore param from: ", model_path)
            saver.restore(self.sess, model_path)
        
            if export:
                for op in graph.get_operations():
                    print(op.name, op.values())
                print('export pnet')
                input_graph_def = graph.as_graph_def()
                output_graph_def = graph_util.convert_variables_to_constants(   
                    sess=self.sess,
                    input_graph_def=input_graph_def, 
                    output_node_names=['cls_prob', 'bbox_pred']
                    ) ##'conv4_1/Softmax', 'conv4_2/BiasAdd'  ##'cls_prob', 'bbox_pred'
 
                with tf.gfile.GFile('pnet.pb', "wb") as f:  
                    f.write(output_graph_def.SerializeToString()) #?????
                print("%d ops in the final graph." % len(output_graph_def.node))  
    def predict(self, databatch):
        print(databatch.shape)
        height, width, _ = databatch.shape
        # print(height, width)
        cls_prob, bbox_pred = self.sess.run([self.cls_prob, self.bbox_pred],
                                                           feed_dict={self.image_op: databatch, self.width_op: width,
                                                                      self.height_op: height})
        return cls_prob, bbox_pred
