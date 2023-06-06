# -*- coding: utf-8 -*-
"""
Created on 2023/05/10
tech:
(Real=False, fake=True)
Adam beta1 set 0.5

@author: Shawn YH Lee
"""
"""----------import package----------"""
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import dcgan
from PIL import Image
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms,datasets
from tqdm import tqdm

"""----------import package end----------"""

"""----------variable init----------"""
modelG_loss_point = []
modelD_loss_point = []
best_acc=0.0
#metrics=np.zeros((2,2),dtype=np.int32)
label_map ={
    0:'bad',
    1:'good',
}
"""----------variable init end----------"""
"""----------module switch setting----------"""
argparse_module = True  #don't False
tqdm_module =True
torchsummary_module = True
show_GANloss_switch = True
load_modelweights = True
"""----------module switch setting end----------"""
"""----------argparse init----------"""
if argparse_module:    
    parser = argparse.ArgumentParser(description = 'train model')
    parser.add_argument('--database_path',type=str,default='./database/')
    parser.add_argument('--database_path1',type=str,default='../CNN_classification_project/database/')
    parser.add_argument('--modelpath',type=str,default='./model/',help='output model save path')
    parser.add_argument('--numpy_data_path',type=str,default='./numpydata/',help='output numpy data')
    parser.add_argument('--training_data_path',type=str,default='./training_process_data/',help='output training data path')
    parser.add_argument('--image_size',type=int,default= 64,help='image size')
    parser.add_argument('--num_classes',type=int,default= 2,help='num classes')
    parser.add_argument('--batch_size',type=int,default= 64,help='batch_size')
    parser.add_argument('--num_epoch',type=int,default= 100,help='num_epoch')
    parser.add_argument('--nz',type=int,default= 100)
    parser.add_argument('--ngf',type=int,default= 16)
    parser.add_argument('--ndf',type=int,default= 16)
    parser.add_argument('--nc',type=int,default= 3)

    parser.add_argument('--modelD',type= str,default='modelD')
    parser.add_argument('--modelG',type= str,default='modelG')
    parser.add_argument('--lrG',type= float,default=2e-4,help='lr for netG')
    parser.add_argument('--lrD',type= float,default=2e-4,help='lr for NetD')

    args = parser.parse_args()
"""----------argparse init end----------"""

"""----------function----------"""
train_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((args.image_size,args.image_size),interpolation = 3),
    #transforms.RandomHorizontalFlip(p=0.5),  # 水平翻轉，概率為0.5
    #transforms.RandomVerticalFlip(p=0.5),    # 垂直翻轉，概率為0.5
    #transforms.RandomRotation(90),
    #transforms.ColorJitter(brightness=0.5, contrast=0.5) ,
    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'),

    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
])
def show_loss_graph():
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5),dpi=100)
    plt.title("GAN loss during training")
    plt.plot(modelD_loss_point,label="D")
    plt.plot(modelG_loss_point,label="G")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend() #加圖例
    plt.savefig(args.training_data_path + 'GANloss.png')
    #plt.show() #can close
"""----------function end----------"""

"""----------main----------"""
if __name__ == '__main__':
    
    train_set = datasets.ImageFolder(root=args.database_path,transform=train_transform) #輸出(圖片tensor,label)
    train_loader = DataLoader(dataset = train_set,batch_size = args.batch_size,shuffle=True,pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f'device:{device}')
    image_shape = (3,args.image_size,args.image_size)

    modelG = dcgan.NetG(args.nz,args.ngf,args.nc).to(device)
    modelD = dcgan.NetD(args.nc,args.ndf).to(device)
    # modelD = dcgan.NetD_custom(args.nc,args.ndf).to(device)
    # modelG = dcgan.NetG_custom(args.nz,args.ngf,args.nc).to(device)
    if load_modelweights:
        modelG.load_state_dict(torch.load(args.modelpath+args.modelG+'.pth'))
        modelD.load_state_dict(torch.load(args.modelpath+args.modelD+'.pth'))
    
    optimizerG = torch.optim.Adam(modelG.parameters(), lr=args.lrG,amsgrad=False,betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(modelD.parameters(), lr=args.lrD,amsgrad=False,betas=(0.5, 0.999))
    loss = nn.BCELoss()
    if tqdm_module:
        pbar = tqdm(range(args.num_epoch),desc='Epoch',unit='epoch',maxinterval=1)
    if torchsummary_module:
        summary(modelD,input_size=(args.nc,args.image_size,args.image_size))
        summary(modelG,input_size=(args.nz,1,1))

    for epoch in pbar:
        modelD.train()
        modelG.train()
        for i ,data in enumerate(train_loader):
            #train modelD let true image be 0
            modelD.zero_grad()
            train_pred = modelD(data[0].to(device)).view(-1)
            label_true = torch.full((data[0].size(0),),0,dtype=torch.float,device=device)
            batch_loss_true = loss(train_pred, label_true)
            batch_loss_true.backward()
            optimizerD.step()

            #modelG create fakeimage
            noise = torch.randn(data[0].size(0),args.nz,1,1).to(device)
            fake = modelG(noise)

            #trian modelD fake image be 1
            label_false = torch.full((data[0].size(0),),1,dtype=torch.float,device=device)  
            train_pred = modelD(fake.detach()).view(-1)#.detach()不計算此tensor梯度
            batch_loss_false = loss(train_pred, label_false)
            # update model D
            batch_loss_false.backward()
            optimizerD.step()
            batch_loss_modelD = (batch_loss_true + batch_loss_false) / 2

            # train model G fake vector
            modelG.zero_grad()
            output = modelD(fake).view(-1)
            batch_loss_modelG = loss(output,label_true)
            batch_loss_modelG.backward()
            # update G
            optimizerG.step()

        # update pbar
        pbar.set_postfix({'modelD loss':batch_loss_modelD.item(),'modelG loss':batch_loss_modelG.item()})
        #append loss to plot
        modelD_loss_point.append(batch_loss_modelD.item())
        modelG_loss_point.append(batch_loss_modelG.item())

        if (epoch+1)%5 == 0:
            with torch.no_grad():
                # fake image
                noise = torch.randn(1,args.nz,1,1).to(device)
                fake = modelG(noise.detach())
                fake_image = transforms.ToPILImage()((fake[0].cpu()+1)/2).convert('RGB')
                fake_image.save(args.training_data_path + str(epoch+1) + "fake.jpg")
                # save model
                torch.save(modelD.state_dict(), args.modelpath +args.modelD + '.pth')
                torch.save(modelG.state_dict(), args.modelpath +args.modelG + '.pth')
    if show_GANloss_switch:
        show_loss_graph()

"""----------main end----------"""