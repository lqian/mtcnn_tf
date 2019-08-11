from training.mtcnn_config import config
import argparse
import os
import shutil

seq=0

def parse_args():
    parser = argparse.ArgumentParser(description='merge all training data for calibration table of quantization',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--stage', dest='stage', help='working stage, can be rnet, onet',
                        default='unknow', type=str) 
    args = parser.parse_args()
    return args

def copy_with_list(list, destFold):
    for line in open(list, 'r'):
        sf = line.split(' ')[0]
        df = os.path.join(destFold, seq, '.jpg')
        shutil.copyfile(sf, df)
        seq+=1

def merge_cali(stage):
    sourceFold = os.path.join(config.ROOT_PATH, "tmp/data/%s/"%(stage))
    destFold = os.path.join(sourceFold, 'calibrations')
    if (os.path.exists(destFold)):
        shutil.rmtree(destFold)
    
    os.makedirs(destFold)
    
    ## merge txt files
    part = os.path.join(sourceFold, 'part.txt')
    pos = os.path.join(sourceFold, 'pos.txt')
    neg = os.path.join(sourceFold, 'neg.txt')
    

    copy_with_list(part, destFold)
    copy_with_list(pos, destFold)
    copy_with_list(neg, destFold) 
    
if __name__ == "__main__":
    args = parse_args();
    merge_cali(args.stage)
    print("done")