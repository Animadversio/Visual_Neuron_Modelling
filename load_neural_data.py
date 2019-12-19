#%%
import h5py
import sys
import os
from os.path import join
from glob import glob
from scipy.io import loadmat
import torch
import numpy as np
import matplotlib.pylab as plt
import csv
from time import time
#%%
stimPath = r"N:\Stimuli\2019-Manifold\beto-191030a\backup_10_30_2019_10_15_31"
allfns = os.listdir(stimPath)
matfns = sorted(glob(join(stimPath, "*.mat")))  # [fn if ".mat" in fn else [] for fn in allfns]
imgfns = sorted(glob(join(stimPath, "*.jpg")))  # [fn if ".mat" in fn else [] for fn in allfns]
#%% Load the Codes mat file
data = loadmat(matfns[1])
codes = data['codes']
img_id = [arr[0] for arr in list(data['ids'][0])] # list of ids
#%%
sys.path.append(r"D:\Github\Activation-Maximization-for-Visual-System")
from torch_net_utils import load_caffenet, load_generator, visualize
# net = load_caffenet()
Generator = load_generator()
#%%
rspPath = r"D:\Network_Data_Sync\Data-Ephys-MAT"#r"N:\Data-Ephys-MAT"#
EphsFN = "Beto64chan-30102019-001"  # "Beto64chan-11112019-006" #

Rspfns = sorted(glob(join(rspPath, EphsFN+"*")))
rspData = h5py.File(Rspfns[1])
spikeID = rspData['meta']['spikeID']
rsp = rspData['rasters']
# Extremely useful code snippet to solve the problem
imgnms_refs = np.array(rspData['Trials']['imageName']).flatten()
imgnms = np.array([''.join(chr(i) for i in rspData[ref]) for ref in imgnms_refs])
#%%
prefchan_idx = np.nonzero(spikeID[0,:]==26)[0] - 1
prefrsp = rsp[:, :, prefchan_idx]  # Dataset reading takes time
scores = prefrsp[:, 50:, :].mean(axis=1) - prefrsp[:, :40, :].mean(axis=1)