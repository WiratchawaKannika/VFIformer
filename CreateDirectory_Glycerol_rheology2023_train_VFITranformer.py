import json 
import pandas as pd
import numpy as np
from pathlib import Path
import PIL
from PIL import Image
import cv2 
import torch
import tqdm
import os
import shutil as sh
import glob
import argparse

# set number of CPUs to run on
ncore = "12"
# set env variables
# have to set these before importing numpy
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore


##function check broken images path.
def img_verify(sub_testlist):
    _except = []
    for file in range(len(sub_testlist)):
        #print(f"Load gen images : {file+1}")
        file_img = sub_testlist[file]
        try:
            img = Image.open(file_img)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            _except.append(file_img)
    return _except 


def main():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--set', type=str, help='[train , test]')
    args = my_parser.parse_args()

    subset = args.set
    
    if subset == 'train':
       ## Import data : path to all text
       df_text = pd.read_csv("/home/kannika/codes_AI/Rheology2023/Glycerol2023_text-train.csv")
       print(f"Train set ==> {df_text.shape[0]} Directory ")
       new_dir = 'HDD/rheology2023/train-vfi/sequence/Glycerol'    ## set path 
       old_dir = 'SSD/Frame_Inter_rheology2023/train_text/_3Frame'
    else: 
       ## Import data : path to all text
       df_text = pd.read_csv("/home/kannika/codes_AI/Rheology2023/Glycerol2023_text-test.csv")
       print(f"Test set ==> {df_text.shape[0]} Directory ")
       new_dir = 'HDD/rheology2023/train-vfi/sequence'    ## set path 
       old_dir = 'SSD/Frame_Inter_rheology2023/test_text/_3Frame'
        
    ## COpy and Createt directory. 
    train_Namelst = list()
    text_demo_ = df_text['Path2text'].tolist()
    for j in range(len(text_demo_)):
        flod_Name = text_demo_[j].split('-')[0]
        _flod_Name = flod_Name.replace(old_dir, new_dir)
        ##** Read Text. **
        with open(text_demo_[j], 'r') as txt:
             sequence_list = [line.strip() for line in txt]
        #** for seq in sequence_list: **
        for i in range(len(sequence_list)):
            img0_path, gt_path, img1_path = sequence_list[i].split(' ')
            sub_testlist = [img0_path, gt_path, img1_path]
            _except = img_verify(sub_testlist)
            if len(_except) == 0 :
                if not os.path.exists('{}/{}'.format(_flod_Name , i+1)) :
                     os.makedirs('{}/{}'.format(_flod_Name , i+1))
                sh.copy("{}".format(img0_path),"{}/{}/im1.jpg".format(_flod_Name , i+1))
                sh.copy("{}".format(gt_path),"{}/{}/im2.jpg".format(_flod_Name , i+1))
                sh.copy("{}".format(img1_path),"{}/{}/im3.jpg".format(_flod_Name , i+1))
                print(f'On Process Create Folder --> [ {_flod_Name}/{i+1} ]')
                train_Name = f'{_flod_Name}/{i+1}'
                _train_Name = train_Name.replace('/media/HDD/rheology2023/train-vfi/sequence/', "")
                train_Namelst.append(_train_Name)
           #             else:
#                 print(f"Images Batch broken ===>: {sub_testlist}")      
    with open(f'/media/HDD/rheology2023/train-vfi/{subset}list.txt', 'w') as f:
         for line in train_Namelst:
             f.write(f"{line}\n")
    print(f'On Process : Write text file name -> [ /media/HDD/rheology2023/train-vfi/{subset}list.txt ')   

            
            
            
            
## Run Function 
if __name__ == '__main__':
    main()
    