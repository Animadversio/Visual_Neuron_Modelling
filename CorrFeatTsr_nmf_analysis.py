%load_ext autoreload
%autoreload 2
#%%
# load corr feat tsr
import numpy as np
from scipy.io import loadmat
from os.path import join
from sklearn.decomposition import NMF
import matplotlib.pylab as plt
from numpy.linalg import norm as npnorm
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
def show_img(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def rectify_tsr(Ttsr, mode="abs", thr=(-5, 5)):
    if mode is "pos_rect":
        Ttsr_pp = np.clip(Ttsr, 0, None)
    elif mode is "abs":
        Ttsr_pp = np.abs(Ttsr)
    elif mode is "thresh":
        Ttsr_pp = Ttsr.copy()
        Ttsr_pp[(Ttsr<thr[1])*(Ttsr>thr[0])] = 0
        Ttsr_pp = np.abs(Ttsr_pp)
    else:
        raise ValueError
    return Ttsr_pp
#%%
from CorrFeatTsr_visualize import CorrFeatScore, corr_GAN_visualize, preprocess
#%%
from GAN_utils import upconvGAN
import torch
from torchvision import models
VGG = models.vgg16(pretrained=True)
VGG.requires_grad_(False)
VGG.features.cuda()
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%%
Animal = "Beto"
Expi = 15
corrDict = np.load(join("S:\corrFeatTsr","%s_Exp%d_Evol_corrTsr.npz"%(Animal,Expi)), allow_pickle=True) # **featFetcher.make_savedict()
cctsr_dict = corrDict.get("cctsr").item()
Ttsr_dict = corrDict.get("Ttsr").item()