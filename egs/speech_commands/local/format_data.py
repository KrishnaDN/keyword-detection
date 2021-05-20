#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:51:58 2021

@author: krishna
"""

import os
import numpy as np
import kaldiio
from local.cmvn import load_cmvn
import argparse
import uuid
import yaml

class Format_data:
    def __init__(self, feat_scp, text_file, cmvn_file, store_folder, manifest, labels):
        self.feat_scp = feat_scp
        self.text_file = text_file
        self.cmvn_file = cmvn_file
        self.store_folder = store_folder
        self.manifest = manifest
        self.Label2Indx = labels
        self.mean, self.istd = load_cmvn(self.cmvn_file, is_json=False)   
        os.makedirs(self.store_folder, exist_ok=True)

    
    def get_label_dict(self,):
        label_dict = {}
        with open(self.text_file) as fid:
            for line in fid:
                label_dict[line.rstrip('\n').split(' ')[0]] = line.rstrip('\n').split(' ')[1]
                
        return label_dict
    

    
    def process(self, ):
        label_dict = self.get_label_dict()
        data = kaldiio.load_scp(self.feat_scp)
        with open(self.manifest,'w') as fid:
            for key in data:
                x = data[key]    
                x = x - self.mean
                x = x * self.istd
                label = self.Label2Indx[label_dict[key]]
                datum = {'feats':x, 'label':label}
                save_path = self.store_folder+'/'+str(uuid.uuid1())+'__'+key+'.npy'
                np.save(save_path, datum)
                fid.write(save_path+'\n')
                
                

if __name__=='__main__':
    parser = argparse.ArgumentParser("Configuration for data preparation")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--feat_scp", required=True, type=str)
    parser.add_argument("--text_file", required=True, type=str)
    parser.add_argument("--cmvn_file", required=True, type=str)
    parser.add_argument("--store_folder", required=True, type=str)
    parser.add_argument("--manifest", required=True, type=str)
    config = parser.parse_args()
    with open(config.config_file,'r') as f:
        params = yaml.safe_load(f)

    labels = params['data']['labels']
    formatter = Format_data(config.feat_scp, config.text_file, config.cmvn_file, config.store_folder,config.manifest, labels)
    formatter.process()