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
import glob
import argparse


my_parser = argparse.ArgumentParser()
my_parser.add_argument('--RootpathtoCSV', type=str, default='/media/SSD/Frame_Inter_rheology2023/_10GenFrame/VFIformer/Frame_Inter/Saliva2/gen1/', help='path to CSV. files')
args = my_parser.parse_args()

RootpathtoCSV = args.RootpathtoCSV
print(f'Process Data Set on Directory: [ {RootpathtoCSV}]')
print(f'-'*100)

files = glob.glob(f"{RootpathtoCSV}*.csv")

for f in files:
    nameFile = f.split('/')[-1]
    _nameFile = nameFile.split('.')[0]
    print(f"Files Gen : {_nameFile}")
    df = pd.read_csv(f)
    print(df.shape)
    lst_filepth = df['seq_inter'].tolist()
    
    ## Start Create Text files 
    df = pd.DataFrame(lst_filepth, columns =['Path'])
    df_ = df[:-1]
    df_ = df_.reset_index(drop=True)
    df2 = pd.DataFrame(lst_filepth, columns =['Path'])
    df2_ = df2[1:]
    df2_ = df2_.reset_index(drop=True)
    df_['Path_txt'] = ''
    for i in range(len(df_)):
        name1 = df_['Path'][i]
        name2 = df2_['Path'][i]
        df_.loc[df_.index[i], 'Path_txt'] = str(name1)+' '+str(name2) 
    print(df_.shape)
    list_path = df_['Path_txt'].tolist()
    with open(f'{RootpathtoCSV}{_nameFile}-2linedemo.txt', 'w') as f:
             for line in list_path:
                 f.write(f"{line}\n")
    print(f'On Process : Write text file name -> [ {RootpathtoCSV}{_nameFile}-2linedemo.txt ] ')
    print('*'*125)
    
    