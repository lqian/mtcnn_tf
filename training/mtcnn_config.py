#coding:utf-8

from easydict import EasyDict as edict


config = edict()

config.BATCH_SIZE = 384
config.CLS_OHEM = True
config.CLS_OHEM_RATIO = 0.7
config.BBOX_OHEM = False
config.BBOX_OHEM_RATIO = 0.7

config.EPS = 1e-14
config.LR_EPOCH = [6, 14, 20]

config.ROOT_PATH = '/train-data/DATA/mtcnn-tf/'

config.CV_RESIZE_OF_NET = {"pnet": (24, 12), "rnet": (48,24), "onet": (96,48)}
config.SIZE_OF_NET = {"pnet": (12, 24), "rnet": (24,48), "onet": (48,96)}
