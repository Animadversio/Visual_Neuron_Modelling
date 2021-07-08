"""Output tables for the paper for CorrFeatTsr """
import os
from os.path import join
from glob import glob
import pickle as pkl
from easydict import EasyDict
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import numpy.ma as ma
from scipy.stats import ttest_rel
from featvis_lib import load_featnet, rectify_tsr
from CorrFeatTsr_utils import area_mapping, multichan2rgb, saveallforms
modelroot = r"E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis\models"
#%% Calculate the sparseness ratio and the corresponding correlation value.
# rect_mode = "Tthresh"; thresh = (None, 3)
# netname = "resnet50_linf8";layer = "layer3";exp_suffix = "_nobdr_res-robust"
#%%
modellist = ['resnet50_linf8-layer3_NF1_bdr1_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer3_NF2_bdr1_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer3_NF5_bdr1_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer3_NF7_bdr1_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer3_NF9_bdr1_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer3_Full_bdr1_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer2_NF1_bdr3_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer2_NF2_bdr3_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer2_NF3_bdr3_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer2_NF5_bdr3_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer2_NF7_bdr3_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer2_NF9_bdr3_Tthresh_3__nobdr_res-robust_CV',
             'resnet50_linf8-layer2_Full_bdr3_Tthresh_3__nobdr_res-robust_CV',
             'resnet50-layer3_NF1_bdr1_Tthresh_3__nobdr_resnet_CV',
             'resnet50-layer3_NF2_bdr1_Tthresh_3__nobdr_resnet_CV',
             'resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet_CV',
             'resnet50-layer3_NF5_bdr1_Tthresh_3__nobdr_resnet_CV',
             'resnet50-layer3_NF7_bdr1_Tthresh_3__nobdr_resnet_CV',
             'resnet50-layer3_NF9_bdr1_Tthresh_3__nobdr_resnet_CV',
             'resnet50-layer3_Full_bdr1_Tthresh_3__nobdr_resnet_CV',
             'alexnet-conv4_NF1_bdr1_Tthresh_3__nobdr_alex_CV',
             'alexnet-conv4_NF2_bdr1_Tthresh_3__nobdr_alex_CV',
             'alexnet-conv4_NF3_bdr1_Tthresh_3__nobdr_alex_CV',
             'alexnet-conv4_NF5_bdr1_Tthresh_3__nobdr_alex_CV',
             'alexnet-conv4_NF7_bdr1_Tthresh_3__nobdr_alex_CV',
             'alexnet-conv4_NF9_bdr1_Tthresh_3__nobdr_alex_CV',
             'alexnet-conv4_Full_bdr1_Tthresh_3__nobdr_alex_CV',
             'vgg16-conv4_3_NF1_bdr1_Tthresh_3__nobdr_CV',
             'vgg16-conv4_3_NF2_bdr1_Tthresh_3__nobdr_CV',
             'vgg16-conv4_3_NF3_bdr1_Tthresh_3__nobdr_CV',
             'vgg16-conv4_3_NF5_bdr1_Tthresh_3__nobdr_CV',
             'vgg16-conv4_3_NF7_bdr1_Tthresh_3__nobdr_CV',
             'vgg16-conv4_3_NF9_bdr1_Tthresh_3__nobdr_CV',
             'vgg16-conv4_3_Full_bdr1_Tthresh_3__nobdr_CV',]

S_col = []
for modelstr in modellist:
    csvpath = glob(join(modelroot, modelstr, "*.csv"))[0]
    exptab = pd.read_csv(csvpath)
    PredData = pkl.load(open(join(modelroot, modelstr, "PredictionData.pkl"),'rb'))[0]
    data = pkl.load(open(join(modelroot, modelstr, "Alfa_Exp01_factors.pkl"), 'rb'))
    if "Nfactor" in exptab:
        NF = exptab.Nfactor[0]
    else:
        NF = "Full"#np.nan

    valmsk = ~((exptab.Animal=="Alfa")&(exptab.Expi==10))
    netname = data.netname
    layer = data.layer
    bdr = data.bdr
    thresh = data.thresh
    rect_mode = data.rect_mode
    exp_suffix = data.exp_suffix
    S = EasyDict()
    S.netname = netname
    S.layer = layer
    S.NF = NF
    S.thresh = thresh[1]

    S.corr_bef_manif = exptab.cc_bef_manif[valmsk].mean()
    S.corr_aft_manif = exptab.cc_aft_manif[valmsk].mean()
    S.corr_aft_manif_norm = exptab.cc_aft_norm_manif[valmsk].mean()
    S.corr_bef_all = exptab.cc_bef_all[valmsk].mean()
    S.corr_aft_all = exptab.cc_aft_all[valmsk].mean()
    S.corr_aft_all_norm = exptab.cc_aft_norm_all[valmsk].mean()
    S_col.append(S)

synopstab = pd.DataFrame(S_col)
print(synopstab)
outtab = r"O:\Manuscript_Manifold\TableS1"
synopstab.to_csv(join(outtab,"synopsis_table.csv"))

#%%
['alexnet-conv2_NF3_none__nobdr_alex',
 'alexnet-conv2_NF3_pos__nobdr_alex',
 'alexnet-conv3_NF3_bdr1_Tthresh_3__nobdr_alex',
 'alexnet-conv3_NF3_none__nobdr_alex',
 'alexnet-conv3_NF3_pos__nobdr_alex',
 'alexnet-conv4_NF1_bdr1_Tthresh_3__nobdr_alex',
 'alexnet-conv4_NF1_bdr1_Tthresh_3__nobdr_alex_CV',
 'alexnet-conv4_NF2_bdr1_Tthresh_3__nobdr_alex',
 'alexnet-conv4_NF2_bdr1_Tthresh_3__nobdr_alex_CV',
 'alexnet-conv4_NF3_bdr1_Tthresh_3__nobdr_alex',
 'alexnet-conv4_NF3_bdr1_Tthresh_3__nobdr_alex_CV',
 'alexnet-conv4_NF3_none__nobdr_alex',
 'alexnet-conv4_NF3_pos__nobdr_alex',
 'alexnet-conv4_NF5_bdr1_Tthresh_3__nobdr_alex',
 'alexnet-conv4_NF5_bdr1_Tthresh_3__nobdr_alex_CV',
 'alexnet-conv4_NF7_bdr1_Tthresh_3__nobdr_alex',
 'alexnet-conv4_NF7_bdr1_Tthresh_3__nobdr_alex_CV',
 'alexnet-conv4_NF9_bdr1_Tthresh_3__nobdr_alex_CV',
 'alexnet-conv5_NF3_bdr1_Tthresh_3__nobdr_alex',
 'alexnet-conv5_NF3_none__nobdr_alex',
 'alexnet-conv5_NF3_pos__nobdr_alex',
 'best_models',
 'resnet50-layer2_NF3_bdr1_pos__nobdr_resnet',
 'resnet50-layer2_NF3_bdr3_Tthresh_3__nobdr_resnet',
 'resnet50-layer3_Full_bdr0_Tthresh_3__nobdr_resnet',
 'resnet50-layer3_Full_bdr0_Tthresh_5__nobdr_resnet',
 'resnet50-layer3_NF1_bdr1_Tthresh_3__nobdr_resnet',
 'resnet50-layer3_NF1_bdr1_Tthresh_3__nobdr_resnet_CV',
 'resnet50-layer3_NF2_bdr1_Tthresh_3__nobdr_resnet',
 'resnet50-layer3_NF2_bdr1_Tthresh_3__nobdr_resnet_CV',
 'resnet50-layer3_NF3_bdr1_pos__nobdr_resnet',
 'resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet',
 'resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet_CV',
 'resnet50-layer3_NF5_bdr1_Tthresh_3__nobdr_resnet',
 'resnet50-layer3_NF5_bdr1_Tthresh_3__nobdr_resnet_CV',
 'resnet50-layer3_NF7_bdr1_Tthresh_3__nobdr_resnet',
 'resnet50-layer3_NF7_bdr1_Tthresh_3__nobdr_resnet_CV',
 'resnet50-layer3_NF9_bdr1_Tthresh_3__nobdr_resnet_CV',
 'resnet50-layer4_NF3_bdr1_pos__nobdr_resnet',
 'resnet50_linf8-layer2_NF1_bdr3_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer2_NF2_bdr3_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer2_NF3_bdr1_pos__nobdr_res-robust',
 'resnet50_linf8-layer2_NF3_bdr3_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer2_NF3_bdr3_Tthresh_5__nobdr_res-robust',
 'resnet50_linf8-layer2_NF5_bdr3_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer2_NF7_bdr3_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer2_NF9_bdr3_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer3_Full_bdr0_Tthresh_5__nobdr_res-robust',
 'resnet50_linf8-layer3_NF1_bdr1_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer3_NF2_bdr1_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer3_NF3_bdr1_pos__nobdr_res-robust',
 'resnet50_linf8-layer3_NF3_bdr1_Tthresh_3_sprs1e-02_l15e-01__nobdr_res-robust',
 'resnet50_linf8-layer3_NF3_bdr1_Tthresh_3_sprs1e-02_l18e-01__nobdr_res-robust',
 'resnet50_linf8-layer3_NF3_bdr1_Tthresh_3_sprs5e-03_l15e-01__nobdr_res-robust',
 'resnet50_linf8-layer3_NF3_bdr1_Tthresh_3_sprs5e-03_l18e-01_KL__nobdr_res-robust',
 'resnet50_linf8-layer3_NF3_bdr1_Tthresh_3_sprs5e-03_l18e-01__nobdr_res-robust',
 'resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer3_NF3_bdr1_Tthresh_5__nobdr_res-robust',
 'resnet50_linf8-layer3_NF5_bdr1_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer3_NF7_bdr1_Tthresh_3__nobdr_res-robust',
 'resnet50_linf8-layer4_NF3_bdr1_pos__nobdr_res-robust',
 'vgg16-conv2_2_NF3_bdr5_pos__nobdr',
 'vgg16-conv2_2_NF3_pos__nobdr',
 'vgg16-conv3_3_NF3_bdr1_Tthresh_3__nobdr',
 'vgg16-conv3_3_NF3_bdr1_Tthresh_3__nobdr_dot',
 'vgg16-conv3_3_NF3_bdr2_Tthresh_3__nobdr',
 'vgg16-conv3_3_NF3_pos__nobdr',
 'vgg16-conv4_3_NF1_bdr1_Tthresh_1__nobdr',
 'vgg16-conv4_3_NF1_bdr1_Tthresh_3__nobdr',
 'vgg16-conv4_3_NF1_bdr1_Tthresh_3__nobdr_CV',
 'vgg16-conv4_3_NF2_bdr1_Tthresh_3__nobdr',
 'vgg16-conv4_3_NF2_bdr1_Tthresh_3__nobdr_CV',
 'vgg16-conv4_3_NF3_bdr1_pos__nobdr',
 'vgg16-conv4_3_NF3_bdr1_Tthresh_2__nobdr',
 'vgg16-conv4_3_NF3_bdr1_Tthresh_3__nobdr',
 'vgg16-conv4_3_NF3_bdr1_Tthresh_3__nobdr_CV',
 'vgg16-conv4_3_NF3_bdr1_Tthresh_5__nobdr',
 'vgg16-conv4_3_NF3_none__nobdr',
 'vgg16-conv4_3_NF3_pos__nobdr',
 'vgg16-conv4_3_NF5_bdr1_Tthresh_3__nobdr',
 'vgg16-conv4_3_NF5_bdr1_Tthresh_3__nobdr_CV',
 'vgg16-conv4_3_NF7_bdr1_Tthresh_3__nobdr',
 'vgg16-conv4_3_NF7_bdr1_Tthresh_3__nobdr_CV',
 'vgg16-conv4_3_NF9_bdr1_Tthresh_3__nobdr_CV',
 'vgg16-conv5_3_NF3_bdr1_pos__nobdr',
 'vgg16-conv5_3_NF3_bdr1_Tthresh_3__nobdr',
 'vgg16-conv5_3_NF5_bdr1_pos__nobdr']