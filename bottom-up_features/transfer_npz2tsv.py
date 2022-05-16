# -------------------------------------------------------- 
 # ROSITA
 # Licensed under The Apache License 2.0 [see LICENSE for details] 
 # Written by Tong-An Luo
 # -------------------------------------------------------- 
 
import os
import os.path as op
import json
import numpy as np
import base64
from tqdm import tqdm
import argparse

from tsv_file import tsv_writer

def transfer_npz2tsv(npz_path, tsv_path):
    # To transfer npz files to a tsv file:
    tsv_file = os.path.join(tsv_path, 'imgfeat.tsv')
    npz_list = sorted(os.listdir(npz_path+'/val') + os.listdir(npz_path+'/train'))
    print('Starting transfer {} npz files'.format(len(npz_list)))

    rows = []
   

    for npz_file in tqdm(npz_list):
        if os.path.exists(op.join(npz_path+'/val', npz_file)):
            npz = np.load(op.join(npz_path+'/val', npz_file),  allow_pickle=True)
        else:
            npz = np.load(op.join(npz_path+'/train', npz_file),  allow_pickle=True)
        img_name = npz['info'][()]['image_id']
        image_id = int(img_name.split('_')[-1])

        img_feat = npz['x']
        img_h = int(npz['image_h'])
        img_w = int(npz['image_w'])
        num_boxes = int(npz['num_bbox'])
        boxes = npz['bbox']

        
        img_feat_encoded = base64.b64encode(img_feat)
        boxes_encoded = base64.b64encode(boxes)
        

        row = [image_id, img_feat_encoded, img_h, img_w, num_boxes, boxes_encoded ]
        rows.append(row)



    tsv_writer(rows, tsv_file)

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-Node Args')
    parser.add_argument('--npz-dir', dest='path_to_npz_files', type=str, default='/home/data_ti6_d/wanghk/coco/features')
    parser.add_argument('--tsv-dir', dest='path_to_tsv_files', type=str, default='/home/wanghk/image-captioning-bottom-up-top-down/bottom-up_features')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    transfer_npz2tsv(args.path_to_npz_files, args.path_to_tsv_files)