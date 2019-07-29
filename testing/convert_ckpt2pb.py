import argparse
import os
#from create_tf_record import *
from training.mtcnn_config import config
from training.mtcnn_model import P_Net, R_Net, O_Net
from detection.detector import Detector
from detection.fcn_detector import FcnDetector

def export_pb(args, net):
    modelPath = os.path.join(config.ROOT_PATH if args.model_dir is None else args.model_dir, 'tmp/model/', net)
    modelPath = os.path.join(modelPath, "%s-%d"%(net, args.epoch))
    
    export = True
    if net == 'pnet':
        FcnDetector(P_Net, modelPath, export)
    elif net == 'rnet':
        Detector(R_Net, 'rnet', 1, modelPath, export)        
    elif net == 'onet':
        Detector(O_Net, 'onet', 1, modelPath, export)          
def parse_args():
    parser = argparse.ArgumentParser(description='mtcnn training checkpoint convert to PB tool ...',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)  
    parser.add_argument('--model_dir', dest='model_dir', type=str, help='directory store all mtcnn model')
    parser.add_argument('--epoch', dest='epoch', type=int, default=30, help='training epoch of checkpoint for restore model')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    export_pb(args, 'pnet')
    export_pb(args, 'rnet')
    export_pb(args, 'onet')