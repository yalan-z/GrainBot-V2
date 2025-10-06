import os

from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image
import tensorflow as tf

device=torch.device('cuda' )
weight_path='params/unet.pth'
data_path=r'data'
validation_path=r'validationdata'
save_path='train_image'
save_trainloss = open("bceloss_w_train.txt",'w',encoding="utf-8")
save_valiloss = open("bceloss_w_vali.txt",'w',encoding="utf-8")
if __name__ == '__main__':
    data_loader=DataLoader(MyDataset(data_path),batch_size=2,shuffle=True)
    vali_loader = DataLoader(MyDataset(validation_path), batch_size=2, shuffle=False)

    net=UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt=optim.Adam(net.parameters())
    loss_fun=nn.BCELoss()

    epoch=1
    while True:
        loss=0.0
        net.train()
        for i,(image,segment_image) in enumerate(data_loader):
            image, segment_image=image.to(device),segment_image.to(device)

            out_image=net(image)
            train_loss=loss_fun(out_image,segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()

            #if i==0:
                #print(train_loss.item(), file=save_loss)
            if i%5==0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            if i%50==0:
                torch.save(net.state_dict(),weight_path)
            loss+=train_loss.item()



            _image=image[0]
            _segment_image=segment_image[0]
            _out_image=out_image[0]

            img=torch.stack([_image,_segment_image,_out_image],dim=0)
            save_image(img,f'{save_path}/{i}.png')
        avg_train_loss=loss / len(data_loader)
        print(avg_train_loss, file=save_trainloss)
        net.eval()
        val_loss = 0.0

        with torch.no_grad():
            for image, segment_image in vali_loader:
                image, segment_image = image.to(device), segment_image.to(device)

                out_image = net(image)
                val_loss += loss_fun(out_image, segment_image).item()

        avg_val_loss = val_loss / len(vali_loader)
        print(avg_val_loss, file=save_valiloss)

        epoch+=1


