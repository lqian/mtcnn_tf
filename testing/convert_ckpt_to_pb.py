#coding:utf-8
## ref: https://github.com/lianjizhe/tensorflow_pb/blob/master/convert_ckpt_to_pb.py
"""
此文件可以把ckpt模型转为pb模型
这个方法直接从训练模型恢复，不适合推理使用
"""
import argparse
import tensorflow as tf
#from create_tf_record import *
from tensorflow.python.framework import graph_util

net_name_dict = {'pnet':['cls_prob', 'bbox_pred'], 
                'rnet':['cls_fc/Softmax', 'bbox_fc/BiasAdd'], 
                'onet':['cls_fc/Softmax', 'bbox_fc/BiasAdd', 'landmark_fc/BiasAdd']}
def freeze_graph(input_checkpoint,output_graph, net_name):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # 直接用最后输出的节点，可以在tensorboard中查找到，tensorboard只能在linux中使用
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph() # 获得默认的图
    for op in graph.get_operations():
        print(op.name, op.values())
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
 
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        output_graph_def = graph_util.convert_variables_to_constants(   
            sess=sess,
            input_graph_def=input_graph_def, 
            output_node_names=net_name_dict[net_name]) 
 
        with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
            f.write(output_graph_def.SerializeToString()) #序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点

def parse_args():
    parser = argparse.ArgumentParser(description='checkpoint convert to PB tool ...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--ckpt', dest= 'ckpt',  type=str, help='model training checkpoint file')
    parser.add_argument('--net', dest='net', type=str, help='mtcnn net name: onet, rnet, pnet')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args() 
    freeze_graph(args.ckpt, args.net+'.pb', args.net)