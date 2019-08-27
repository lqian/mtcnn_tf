#coding:utf-8

from easydict import EasyDict as edict

config = edict()

config.BATCH_SIZE = 384
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [25, 50, 75]

config.ROOT_PATH = '/train-data/DATA/mtcnn-tf-12x24/'

config.CV_RESIZE_OF_NET = {"pnet": (24, 12), "rnet": (48,24), "onet": (96,48)}
config.SIZE_OF_NET = {"pnet": (12, 24), "rnet": (24,48), "onet": (48,96)}

# config.CV_RESIZE_OF_NET = {"pnet": (16, 8), "rnet": (32,16), "onet": (64,32)}
# config.SIZE_OF_NET = {"pnet": (8, 16), "rnet": (16,32), "onet": (32,64)}

# config.CV_RESIZE_OF_NET = {"pnet": (30, 12), "rnet": (60,24), "onet": (120,48)}
# config.SIZE_OF_NET = {"pnet": (12, 30), "rnet": (24,60), "onet": (48,120)}

# net_name_dict = {
#     'pnet':['cls_prob', 'bbox_pred', 'landmark_pred'], 
# #     'pnet':['conv4_1/Softmax', 'conv4_2/BiasAdd', 'conv4_3/BiasAdd' ], 
#     'rnet':['cls_fc/Softmax', 'bbox_fc/BiasAdd', 'landmark_fc/BiasAdd'], 
#     'onet':['cls_fc/Softmax', 'bbox_fc/BiasAdd', 'landmark_fc/BiasAdd']
# }

net_name_dict = {
    'pnet':['cls_prob', 'bbox_pred'], 
#     'pnet':['conv4_1/Softmax', 'conv4_2/BiasAdd', 'conv4_3/BiasAdd' ], 
    'rnet':['cls_fc/Softmax', 'bbox_fc/BiasAdd'], 
    'onet':['cls_fc/Softmax', 'bbox_fc/BiasAdd', 'landmark_fc/BiasAdd']
}