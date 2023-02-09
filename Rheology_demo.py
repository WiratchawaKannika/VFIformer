import pandas as pd
import os
import sys
import time
import logging
import math
import glob
import cv2
import argparse
import numpy as np
from torch.nn.parallel import DataParallel, DistributedDataParallel
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as data
from skimage.color import rgb2yuv, yuv2rgb

from utils.util import setup_logger, print_args
from utils.pytorch_msssim import ssim_matlab
from models import modules
from models.modules import define_G



def load_networks(network, resume, strict=True):
    load_path = resume
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    load_net = torch.load(load_path, map_location=torch.device('cpu'))
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    if 'optimizer' or 'scheduler' in net_name:
        network.load_state_dict(load_net_clean)
    else:
        network.load_state_dict(load_net_clean, strict=strict)

    return network



def main():
    parser = argparse.ArgumentParser(description='inference for a single sample')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--name', default='demo', type=str)
    parser.add_argument('--phase', default='test', type=str)

    ## device setting
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    ## network setting
    parser.add_argument('--net_name', default='VFIformer', type=str, help='')

    ## dataloader setting
    parser.add_argument('--crop_size', default=192, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_root', default='/media/SSD/Frame_Inter_rheology2023/_10GenFrame/ues2Frame/pred_text/Saliva2/origin',type=str)
    parser.add_argument('--resume', default='./pretrained_models/net_220.pth', type=str)
    parser.add_argument('--resume_flownet', default='', type=str)
    #parser.add_argument('--save_folder', default='', type=str)
    parser.add_argument('--genNum', type=int)

    ## setup training environment
    args = parser.parse_args()

    ## setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    ## distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        args.dist = False
        args.rank = -1
        print('Disabled distributed training.')
    else:
        args.dist = True
        init_dist()
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()


    cudnn.benchmark = True
    ## ***save paths 
    #save_path = args.save_folder
   
    ## load model
    device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
    args.device = device
    net = define_G(args)
    net = load_networks(net, args.resume)
    net.eval()

    ## load data
    divisor = 64
    multi = 3
    
    genNum = args.genNum
    _genNum = f'gen{genNum}'
    genNum_old = genNum-1
    _genNumold = f'gen{genNum_old}'
    pth = args.data_root
    test_demo = glob.glob(f"{pth}/*-2linedemo.txt")
    test_demo.sort()
    for file in test_demo: 
         ##** Set SAve path 
        print(f'On Process Folder  -->> [ {file} ]')
        list_imgframe = list()
        if genNum == 1:
            folder_name_ = file.replace("-2linedemo", f"_{_genNum}-inter")
            folder_name_ = folder_name_.split('.')[0]
            save_pathimg = folder_name_.replace("ues2Frame/pred_text/Saliva2/origin", f"VFIformer/Frame_Inter/Saliva2/{_genNum}")
        else:
            folder_name_ = file.replace(f"{_genNumold}-2linedemo", f"{_genNum}-inter")
            folder_name_ = folder_name_.split('.')[0]
            save_pathimg = folder_name_.replace(_genNumold, _genNum)
                    
         ##**Mkdir Directory 
        import imageio
        os.makedirs(save_pathimg, exist_ok=True)
        ### Create name img path
        name_img = save_pathimg.split("/")[-1]
        name_img_ = name_img.split("_")[:-1]
        __name_img = '_'.join(name_img_)
        
        ## Create path to save CSV.
        save_csv = save_pathimg.split("/")[:-1]
        save_csv_ = '/'.join(save_csv)
        pathName_csv = save_csv_+'/'+__name_img+'_'+_genNum+'.csv'
        _pathName_csv = pathName_csv.replace(_genNumold, f"Frame_Inter/Saliva2/{_genNum}")
           
        ##read text files dataset
        with open(file, 'r') as txt:
             sequence_list = [line.strip() for line in txt]
        for i in range(len(sequence_list)): 
            img0_path, img1_path = sequence_list[i].split(' ')

            img0 = cv2.imread(img0_path)
            img1 = cv2.imread(img1_path)

            h, w, c = img0.shape
            if h % divisor != 0 or w % divisor != 0:
                h_new = math.ceil(h / divisor) * divisor
                w_new = math.ceil(w / divisor) * divisor
                pad_t = (h_new - h) // 2
                pad_d = (h_new - h) // 2 + (h_new - h) % 2
                pad_l = (w_new - w) // 2
                pad_r = (w_new - w) // 2 + (w_new - w) % 2
                img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)  # cv2.BORDER_REFLECT
                img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            else:
                pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0

            img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
            img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)

            with torch.no_grad():
                output, _ = net(img0, img1, None)
                h, w = output.size()[2:]
                output = output[:, :, pad_t:h-pad_d, pad_l:w-pad_r]

            imt = output[0].flip(dims=(0,)).clamp(0., 1.)
            out_name = os.path.join(save_pathimg, __name_img+'_inter'+str(i+1)+'_'+_genNum+'.jpg') 
            torchvision.utils.save_image(imt, out_name)
            #torchvision.utils.save_image(imt, os.path.join(save_pathimg, os.path.basename(img0_path).split('.')[0]+'_inter'+'.png'))
            #torch.cuda.empty_cache()
            #time.sleep(0.5)
            print('result saved!')
            ## Save sequence frame to CSV. 
              ## Save Sequence frame to csv.
            if i == len(sequence_list)-1 :
                list_imgframe.append(img0_path)
                list_imgframe.append(out_name)
                list_imgframe.append(img1_path)
            else:
                list_imgframe.append(img0_path)
                list_imgframe.append(out_name)
               
         ##save to CSV.       
        df = pd.DataFrame(list_imgframe, columns =['seq_inter'])
        df.to_csv(_pathName_csv)
        print('Frame Interpolation saVe at -->>', save_pathimg)
        print(f"Save Sequence Dataframe at -->> {_pathName_csv} With Shape: {df.shape}")
        print('*'*125)



if __name__ == '__main__':
    main()
