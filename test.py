import os

import torch
import tensorflow as tf
from tensorflow import metrics
from net import *
from utils import keep_image_size_open
from data import *
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net=UNet().to(device)

weights='params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights,map_location=device))
    print('successfully')
else:
    print('no loading')

imgpath=r'C:\Users\Yalan\GB-project-data\AFM_20240521\20240521_png'
savepath=r'C:\Users\Yalan\GB-project-data\AFM_20240521\20240521_png'
if os.path.isdir(imgpath):
    filenames=os.listdir(imgpath)
    for name in filenames:
        file = imgpath + "\\" + name
        _input = file
        img = keep_image_size_open(_input)
        img_data = transform(img).to(device)
        #save_image(img_data, savepath + "\\" + 'reshape' + name)
        img_data = torch.unsqueeze(img_data, dim=0)
        out = net(img_data)
        save_image(out, savepath + '\\' + 'result' + name)
elif os.path.isfile(imgpath):
    img = keep_image_size_open(imgpath)
    img_data = transform(img).to(device)
    img_data = torch.unsqueeze(img_data, dim=0)
    out = net(img_data)
    save_image(out, savepath +"\\" +'test.png')


