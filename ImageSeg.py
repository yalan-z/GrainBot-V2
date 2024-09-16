import os
import torch
from net import *
from utils import keep_image_size_open
from data import *
from torchvision.utils import save_image

class validation:
    def __init__(self,para_path):
        self.path=para_path
    def valid(self,imgpath):
        unet = UNet().to(torch.device('cpu'))
        # weights = 'params/unet.pth'
        if os.path.exists(self.path):
            unet.load_state_dict(torch.load(self.path, map_location=torch.device('cpu')))
            print('successfully segmented')
            self.filenames = os.listdir(imgpath)
            for name in self.filenames:
                file = imgpath + "\\" + name
                _input = file
                img = keep_image_size_open(_input)
                img_data = transform(img).to(torch.device('cpu'))
                img_data = torch.unsqueeze(img_data, dim=0)
                out = unet(img_data)
                save_image(out, imgpath + '\\' + 'seg_' + name)
        else:
            print('parameters do not exist')
        return
