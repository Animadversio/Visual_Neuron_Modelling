"""
Define a factorized convolution model for the neurons

tensorflow 1*1 convolution and linearized fullly connected layer

"""
mat_path = r"C:\Users\binxu\OneDrive - Washington University in St. Louis\Mat_Statistics"
from scipy.io import loadmat
from os.path import join
#%%
Animal = "Alfa"
# data = loadmat(join(mat_path, "Beto_ManifPopDynamics.mat"), struct_as_record=False, squeeze_me=True)
# ManifDyn = loadmat(join(mat_path, Animal + "_ManifPopDynamics.mat"), struct_as_record=False, squeeze_me=True)['ManifDyn']
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True)['EStats']
#%%
from skimage.io import imread, imread_collection
from torchvision import models
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam

# VGG = models.vgg16(pretrained=True)
# img_dim = 256
# dummy = torch.zeros([1, 3, img_dim, img_dim]).cuda()
# FeatNet = VGG.features[:20]
# FeatNet.requires_grad_(False).cuda()
# feat_tsr = FeatNet(dummy)
# ch = feat_tsr.shape[1]
# H, W = tuple(feat_tsr.shape[-2:])
# feat_trans = nn.Conv2d(ch, 1, (1, 1))
# sp_mask = nn.Linear(H * W, 1)
# feat_trans(feat_tsr).reshape([-1, H * W])
# optimizer = Adam(list(feat_trans.parameters())+list(sp_mask.parameters()), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
#%
layername_dict ={"vgg16":['conv1', 'conv1_relu',
                         'conv2', 'conv2_relu', 'pool1',
                         'conv3', 'conv3_relu',
                         'conv4', 'conv4_relu', 'pool2',
                         'conv5', 'conv5_relu',
                         'conv6', 'conv6_relu',
                         'conv7', 'conv7_relu', 'pool3',
                         'conv8', 'conv8_relu',
                         'conv9', 'conv9_relu',
                         'conv10', 'conv10_relu', 'pool4',
                         'conv11', 'conv11_relu',
                         'conv12', 'conv12_relu',
                         'conv13', 'conv13_relu', 'pool5',
                         'fc1', 'fc1_relu', 'dropout1',
                         'fc2', 'fc2_relu', 'dropout2',
                         'fc3'],
                 "densenet121":['conv1',
                                 'bn1',
                                 'bn1_relu',
                                 'pool1',
                                 'denseblock1', 'transition1',
                                 'denseblock2', 'transition2',
                                 'denseblock3', 'transition3',
                                 'denseblock4',
                                 'bn2',
                                 'fc1']}
class FactModel(nn.Module):
    def __init__(self, basenet="vgg16", layername="conv9", img_dim=256):
        super(FactModel, self).__init__()
        if basenet == "vgg16":
            VGG = models.vgg16(pretrained=True)
            layer_idx = layername_dict[basenet].index(layername)
            self.FeatNet = VGG.features[:layer_idx + 1].cuda()
        dummy = torch.zeros([1, 3, img_dim, img_dim]).cuda()
        feat_tsr = self.FeatNet(dummy)
        self.ch, self.maskH, self.maskW = tuple(feat_tsr.shape[-3:])
        print("Using pretrained mode %s upto layer %s, %d channels, (%d, %d) spatial"%(basenet, layername, self.ch, self.maskH, self.maskW))
        self.feat_trans = nn.Conv2d(self.ch, 1, (1, 1)).cuda()  # feature transform
        self.sp_mask = nn.Linear(self.maskH * self.maskW, 1).cuda()  # assume scaler rate output
        self.optimizer = Adam(list(self.feat_trans.parameters())+list(self.sp_mask.parameters()), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)

    def forward(self, img):
        x = self.FeatNet(img)
        x = self.feat_trans(x).reshape([-1, self.maskH * self.maskW])
        r = F.relu(self.sp_mask(x))
        return r
#%%
# "subst N: E:\Network_Data_Sync"
from glob import glob
# imgnms = os.listdir(MStats[6].meta.stimuli) + os.listdir(r"N:\Stimuli\2019-Manifold\pasupathy-wg-f-4-ori")
Pasupath = r"N:\Stimuli\2019-Manifold\pasupathy-wg-f-4-ori"
Gaborpath = r"N:\Stimuli\2019-Manifold\gabor"
imgnms = glob(MStats[6].meta.stimuli+"\\*") + glob(Pasupath+"\\*") + glob(Gaborpath+"\\*")
import numpy as np
imguniq = np.unique(MStats[6].imageName)
imgpathuniq = [[path for path in imgnms if imgnm in path][0] for imgnm in imguniq]
#%%
FM = FactModel(img_dim=256)
#%% Sort the response to the order of images
fitimg = [fn for fn in imgpathuniq if "PC2" in fn]
Expi = 6
Wdw = 50,200
import re
fitrsp = np.zeros(len(fitimg))
for i in range(len(fitimg)):
    if "norm" in fitimg[i]:
        theta, phi = re.findall("PC2_(.*)_PC3_(.*)\.", fitimg[i])[0]
        theta, phi = int(theta), int(phi)
        theta, phi = round((theta + 90) / 18), round((phi + 90)/ 18)
        if len(MStats[6].manif.psth[theta, phi].shape)==2:
            fitrsp[i] = MStats[6].manif.psth[theta, phi][50:,:].mean()
        else:
            assert False
fitrsptsr = torch.tensor(fitrsp).unsqueeze(1).cuda()
preprocess = transforms.Compose([transforms.ToTensor(),
            #transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
ppimgs = []
for img_path in fitimg:
    # should be taken care of by the CNN part
    curimg = imread(img_path)
    # curimg = resize(curimg, (256, 256))
    x = preprocess(curimg)
    ppimgs.append(x.unsqueeze(0))
input_tsr = torch.cat(tuple(ppimgs), dim=0)
#%% Process images and feed into network
from skimage.transform import resize, rescale
from skimage.color import gray2rgb
from time import time
alpha = 10
Bnum = 11
t0 = time()
optimizer = Adam(list(FM.feat_trans.parameters())+list(FM.sp_mask.parameters()), lr=0.005, betas=(0.9, 0.999), weight_decay=0)
for ep in range(10):
    idx_csr = 0
    BS_num = 0
    rsp_all = torch.tensor([])
    while idx_csr < len(fitimg):
        idx_ub = min(idx_csr + Bnum, len(fitimg))
        ppimgs = []
        for img_path in fitimg[idx_csr:idx_ub]:
            # should be taken care of by the CNN part
            curimg = imread(img_path)
            # curimg = resize(curimg, (256, 256))
            x = preprocess(curimg)
            ppimgs.append(x.unsqueeze(0))
        input_tsr = torch.cat(tuple(ppimgs), dim=0)
        # should be taken care of by the CNN part
        optimizer.zero_grad()
        rsp = FM(input_tsr.cuda())#[idx_csr:idx_ub]
        rsp_all = torch.cat((rsp_all, rsp.detach().cpu()), dim=0)
        loss = (fitrsptsr[idx_csr:idx_ub] - rsp).pow(2).mean()
        if alpha != 0:
            reg = FM.feat_trans.weight.abs().sum() + FM.feat_trans.bias.abs().sum() + FM.sp_mask.weight.abs().sum() + FM.sp_mask.bias.abs().sum()
            loss += reg * alpha
        loss.backward()
        optimizer.step()
        FM.sp_mask.weight.data.clamp_(0)
        idx_csr = idx_ub
        BS_num += 1
        print("Ep%d Finished %d batch, take %.1f s loss %.1f, reg %.1f" % (ep, BS_num, time() - t0, loss.item(), reg.item()))
# 315 sec for 100 epocs. Note preload the data make it much much slower...not sure why
# 88 sec for 10 epocs
#%%
idx_csr = 0
BS_num = 0
rsp_all = torch.tensor([])
while idx_csr < len(fitimg):
    idx_ub = min(idx_csr + Bnum, len(fitimg))
    ppimgs = []
    for img_path in fitimg[idx_csr:idx_ub]:
        # should be taken care of by the CNN part
        curimg = imread(img_path)
        # curimg = resize(curimg, (256, 256))
        x = preprocess(curimg)
        ppimgs.append(x.unsqueeze(0))
    input_tsr = torch.cat(tuple(ppimgs), dim=0)
    with torch.no_grad():
        rsp = FM(input_tsr.cuda()).cpu()
    rsp_all = torch.cat((rsp_all, rsp), dim=0)
#%%
import matplotlib.pylab as plt
plt.imshow(FM.sp_mask.weight.reshape(32,32).detach().cpu())
plt.show()
#%%
plt.imshow(rsp_all.reshape(11, 11).detach().cpu())
plt.show()
#%%
plt.imshow(fitrsptsr.reshape(11, 11).detach().cpu())
plt.show()
