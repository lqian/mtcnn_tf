import argparse
import os
#from create_tf_record import *
from training.mtcnn_config import config
from training.mtcnn_model import P_Net, R_Net, O_Net
import tensorflow as tf
from tensorflow.python import graph_util
from training.mtcnn_config import net_name_dict
from prepare_data.gen_hard_bbox_pnet import net_size

def export_pb(args, net):
    modelPath = os.path.join(config.ROOT_PATH if args.model_dir is None else args.model_dir, 'tmp/model/', net)
    modelPath = os.path.join(modelPath, "%s-%d"%(net, args.epoch))
     
    if net == 'pnet':
        PNet_Export(P_Net, modelPath, for_mnn=args.for_mnn)
    elif net == 'rnet':
        RNet_Export(R_Net, modelPath)        
    elif net == 'onet':
        ONet_Export(O_Net,  modelPath)
                  
def parse_args():
    parser = argparse.ArgumentParser(description='mtcnn training checkpoint convert to PB tool ...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)  
    parser.add_argument('--model_dir', dest='model_dir', type=str, help='directory store all mtcnn model')
    parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='training epoch of checkpoint for restore model')
    parser.add_argument('--for_mnn', dest='for_mnn', type=bool, default=True, help='export model for mnn without input size placeholder')
    args = parser.parse_args()
    return args

class RNet_Export(object):
    def __init__(self, RNet, model_path):
        net = 'rnet'
        self.net_size = config.SIZE_OF_NET[net]
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[1, self.net_size[0], self.net_size[1], 3], name='input_image')
            #figure out landmark            
            self.cls_prob, self.bbox_pred, self.landmark_pred = RNet(self.image_op, training=False)
            self.sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print("Restore param from: ", model_path)
            saver.restore(self.sess, model_path)                        
        
        
        for op in graph.get_operations():
            print(op.name, op.values())
        print('export pb:', net)
        input_graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(   
            sess=self.sess,
            input_graph_def=input_graph_def, 
            output_node_names=net_name_dict[net]) 
        with tf.gfile.GFile(net +'.pb', "wb") as f:  
            f.write(output_graph_def.SerializeToString()) 
        print("%d ops in the final graph." % len(output_graph_def.node)) 

class ONet_Export(object):
    def __init__(self, RNet, model_path):
        net = 'onet'
        self.net_size = config.SIZE_OF_NET[net]
        graph = tf.Graph()
        with graph.as_default():
            self.image_op = tf.placeholder(tf.float32, shape=[1, self.net_size[0], self.net_size[1], 3], name='input_image')
            #figure out landmark            
            self.cls_prob, self.bbox_pred, self.landmark_pred = RNet(self.image_op, training=False)
            self.sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))
            saver = tf.train.Saver()
            #check whether the dictionary is valid
            model_dict = '/'.join(model_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(model_dict)
            readstate = ckpt and ckpt.model_checkpoint_path
            assert  readstate, "the params dictionary is not valid"
            print("Restore param from: ", model_path)
            saver.restore(self.sess, model_path)                        
        
        
        for op in graph.get_operations():
            print(op.name, op.values())
        print('export pb:', net)
        input_graph_def = graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(   
            sess=self.sess,
            input_graph_def=input_graph_def, 
            output_node_names=net_name_dict[net]) 
        with tf.gfile.GFile(net +'.pb', "wb") as f:  
            f.write(output_graph_def.SerializeToString()) 
        print("%d ops in the final graph." % len(output_graph_def.node)) 
                
class PNet_Export(object):
    def __init__(self, PNet, model_path, for_mnn=True):
        net = 'pnet'
        self.net_size = config.SIZE_OF_NET[net]
        graph = tf.Graph()
        with graph.as_default():
            #define tensor and op in graph(-1,1)
            self.image_op = tf.placeholder(tf.float32, name='input_image')
            self.width_op = tf.placeholder(tf.int32, name='image_width')
            self.height_op = tf.placeholder(tf.int32, name='image_height')
            input_image_std = tf.placeholder(tf.float32, shape=[1, self.net_size[0], self.net_size[1], 3], name='input_image_std')
            image_reshape = tf.reshape(self.image_op, [1, self.height_op, self.width_op, 3])
            #self.cls_prob batch*2
            #self.bbox_pred batch*4
            #construct model here
            #self.cls_prob, self.bbox_pred = net_factory(image_reshape, training=False)
            #contains landmark
            
            self.cls_prob, self.bbox_pred, _ = PNet(input_image_std if for_mnn else image_reshape, training=False)
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
        
            for op in graph.get_operations():
                print(op.name, op.values())
            print('export pnet')
            input_graph_def = graph.as_graph_def()
            output_graph_def = graph_util.convert_variables_to_constants(   
                sess=self.sess,
                input_graph_def=input_graph_def, 
                output_node_names= net_name_dict['pnet']
            ) ##'conv4_1/Softmax', 'conv4_2/BiasAdd'  ##'cls_prob', 'bbox_pred'
 
            with tf.gfile.GFile('pnet.pb', "wb") as f:  
                f.write(output_graph_def.SerializeToString()) #?????
            print("%d ops in the final graph." % len(output_graph_def.node))  
            
#             converter = tf.lite.TFLiteConverter.from_session(self.sess, ['input_image_std'], net_name_dict['pnet'])
#             tflite_model = converter.convert()
#             open("pnet.tflite", "wb").write(tflite_model)

            ## export for tflite
            converter = tf.lite.TFLiteConverter.from_frozen_graph('pnet.pb', ['input_image_std'],  net_name_dict['pnet'])
            tflite_model = converter.convert()
            open("pnet.tflite", "wb").write(tflite_model)
            
            # 量化 不正确
#             converter = tf.lite.TFLiteConverter.from_saved_model('./')
#             converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
#             converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] 
#             open("pnet.tflite-quantization", "wb").write(tflite_model)
if __name__ == '__main__':
    args = parse_args()
    export_pb(args, 'pnet')
    export_pb(args, 'rnet')
    export_pb(args, 'onet')