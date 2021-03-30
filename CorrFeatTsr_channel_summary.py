""""This script examine the channel wise distribution of features in CorrFeatTsr Analysis. """
import sys
from os.path import join
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import ttest_rel, ttest_ind, ranksums, pearsonr
import torch
from easydict import EasyDict
from skimage.transform import resize
from skimage.io import imread
import numpy as np
def norm_value(tsr):
    normtsr = (tsr - np.nanmin(tsr)) / (np.nanmax(tsr) - np.nanmin(tsr))
    return normtsr

def calc_chanvec_from_tensors(D, ):
    """Summarize CorrFeatTsr along the channel dim """
    cctsr = D["cctsr"].item()
    Ttsr = D["Ttsr"].item()
    layers = list(cctsr.keys())
    chanvec = EasyDict()
    chanvec.mean = EasyDict()
    chanvec.max = EasyDict()
    chanvec.tsig_mean = EasyDict()
    for layer in layers:
        # map of original small size
        spatialN = cctsr[layer].shape[1] * cctsr[layer].shape[2]
        chanvec.max[layer] = np.nanmax(np.abs(cctsr[layer]), axis=(1, 2))
        chanvec.mean[layer] = np.nansum(np.abs(cctsr[layer]), axis=(1, 2)) / spatialN
        cctsr_layer = np.copy(cctsr[layer])
        cctsr_layer[np.abs(Ttsr[layer]) < 7] = 0
        chanvec.tsig_mean[layer] = np.nansum(np.abs(cctsr_layer), axis=(1, 2)) / spatialN
    return chanvec
#%%
import os
ccdir = "S:\corrFeatTsr"
figdir = r"O:\ProtoObjectivenss\channel_summary"
mat_path = r"O:\Mat_Statistics"
os.makedirs(figdir, exist_ok=True)
# Animal = "Alfa"
outlabel = "centRFintp"
cctsr_label = "Evol_nobdr_"
Scol = []
chanvec_col = []
for Animal in ["Alfa", "Beto"]:
    # Load summary stats for each animal
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    for Expi in range(1, len(EStats)+1):#len(EStats)+1):
        D = np.load(join(ccdir, "%s_Exp%d_%scorrTsr.npz" % (Animal, Expi, cctsr_label)),
                    allow_pickle=True)
        cctsr = D["cctsr"].item()
        Ttsr = D["Ttsr"].item()
        layers = list(cctsr.keys())
        figh = plt.figure(figsize=[11,4])
        chanvec = EasyDict()
        chanvec.mean = EasyDict()
        chanvec.max = EasyDict()
        chanvec.tsig_mean = EasyDict()
        for li, layer in enumerate(layers):
            # map of original small size
            spatialN = cctsr[layer].shape[1] * cctsr[layer].shape[2]
            chanvec.max[layer] = np.nanmax(np.abs(cctsr[layer]), axis=(1,2))
            chanvec.mean[layer] = np.nansum(np.abs(cctsr[layer]), axis=(1,2)) / spatialN
            cctsr_layer = np.copy(cctsr[layer])
            cctsr_layer[np.abs(Ttsr[layer]) < 7] = 0
            chanvec.tsig_mean[layer] = np.nansum(np.abs(cctsr_layer), axis=(1,2)) / spatialN
            plt.subplot(1,4,li+1)
            plt.plot(chanvec.max[layer], label="max",alpha=0.7)
            plt.plot(chanvec.mean[layer], label="mean",alpha=0.7)
            plt.plot(chanvec.tsig_mean[layer], label="Signif Mean",alpha=0.7)
            plt.title("Layer %s"%(layer))
            plt.legend()
            plt.xlabel("Channel Id")
        plt.suptitle("%s Exp%d %s"%(Animal, Expi, cctsr_label, ))
        plt.savefig(join(figdir,"%s_Exp%02d_%s_chanvec_sum.png"%(Animal,Expi,cctsr_label)))
        plt.show()
        chanvec_col.append(chanvec)
#%% Summarize the feature vector
for layer in layers:
    for vecname in ["max", "mean", "tsig_mean"]:
        vec_col = np.array([chanvec[vecname][layer] for chanvec in chanvec_col])
        avgvec = np.nanmean(vec_col, axis=0)
        # plt.plot(vec_col.T, label="avg_" + vecname, alpha=0.1)
        plt.plot(avgvec, label="avg_" + vecname, alpha=0.5)
    plt.title("Both All Exp Avg %s Layer %s" % (cctsr_label, layer))
    plt.legend()
    plt.xlabel("Channel Id")
    plt.show()
#%%
def summarize_chanvec_mask(masks, labels, save=True):
    for msk, label in zip(masks, labels):
        if msk is None: msk = slice(None)
        figh = plt.figure(figsize=[11, 4])
        for li, layer in enumerate(layers):
            plt.subplot(1, 4, li + 1)
            for vecname in ["max", "mean", "tsig_mean"]:
                vec_col = np.array([chanvec[vecname][layer] for chanvec in np.array(chanvec_col)[msk]])
                avgvec = np.nanmean(vec_col, axis=0)
                # plt.plot(vec_col.T, label="avg_" + vecname, alpha=0.1)
                plt.plot(avgvec, label="avg_" + vecname, alpha=0.5)
            plt.title("Layer %s" % (layer, ))
            plt.legend()
            plt.xlabel("Channel Id")
        plt.suptitle("Both %s Exp %s Avg (N=%d)" % (label, cctsr_label, len(vec_col)))
        if save: plt.savefig(join(figdir, "Both_sum_%s_%s_chanvec_sum.png" % (label, cctsr_label)))
        plt.show()

#%%
# Load Trajectory Statistics Successfulness of evol and meta info.
BothTrajtab = pd.concat(pd.read_csv(join(mat_path, Animal+"_EvolTrajStats.csv")) \
          for Animal in ["Alfa","Beto"])
BothTrajtab = BothTrajtab.reset_index()
sucsmsk = BothTrajtab.t_p_initmax<1E-3
V1msk = (BothTrajtab.pref_chan < 49) & (BothTrajtab.pref_chan > 32)
V4msk = BothTrajtab.pref_chan > 48
ITmsk = BothTrajtab.pref_chan < 33
BothTrajtab["area"] = ""
BothTrajtab.area[V1msk] = "V1"
BothTrajtab.area[V4msk] = "V4"
BothTrajtab.area[ITmsk] = "IT"
#%%
summarize_chanvec_mask([V1msk*sucsmsk,V4msk*sucsmsk,ITmsk*sucsmsk,], ["V1_sucs","V4_sucs","IT_sucs"])
summarize_chanvec_mask([V1msk, V4msk, ITmsk,], ["V1", "V4", "IT"])
# seems no structure across the channel dimension......
#%%
summarize_chanvec_mask([None,sucsmsk], ["All","All_sucs"])
# Correlated voxel number is very different across exps it depends on SNr