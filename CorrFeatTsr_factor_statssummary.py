import os
from os.path import join
import pickle as pkl
from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from featvis_lib import load_featnet, rectify_tsr
from CorrFeatTsr_utils import area_mapping
from data_loader import mat_path, loadmat, load_score_mat
rect_mode = "Tthresh"; thresh = (None, 5)
netname = "resnet50_linf8";layer = "layer3";exp_suffix = "_nobdr_res-robust"
bdr = 1;
S_col = []
for Animal in ["Alfa", "Beto"]:
    MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    # ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)[
    #     'ReprStats']
    for Expi in tqdm(range(1, len(MStats)+1)):
        imgsize = EStats[Expi - 1].evol.imgsize
        imgpos = EStats[Expi - 1].evol.imgpos
        pref_chan = EStats[Expi - 1].evol.pref_chan
        area = area_mapping(pref_chan)
        imgpix = int(imgsize * 40)
        corrDict = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_Evol%s_corrTsr.npz" % (Animal, Expi, exp_suffix)),
                           allow_pickle=True)
        cctsr_dict = corrDict.get("cctsr").item()
        Ttsr_dict = corrDict.get("Ttsr").item()
        stdtsr_dict = corrDict.get("featStd").item()
        covtsr_dict = {layer: cctsr_dict[layer] * stdtsr_dict[layer] for layer in cctsr_dict}

        # show_img(ReprStats[Expi - 1].Manif.BestImg)
        # figdir = join(figroot, "%s_Exp%02d" % (Animal, Expi))
        # os.makedirs(figdir, exist_ok=True)
        Ttsr = Ttsr_dict[layer]
        cctsr = cctsr_dict[layer]
        covtsr = covtsr_dict[layer]
        Ttsr = np.nan_to_num(Ttsr)
        cctsr = np.nan_to_num(cctsr)
        covtsr = np.nan_to_num(covtsr)
        tsrmsk = np.zeros_like(Ttsr, dtype=bool)
        tsrmsk[:, bdr:-bdr, bdr:-bdr] = Ttsr[:, bdr:-bdr, bdr:-bdr] > thresh[1]
        S = EasyDict()
        S.Animal = Animal
        S.Expi = Expi
        S.pref_chan = pref_chan
        S.sparse_num = (tsrmsk).sum()
        S.sparse_prct = (tsrmsk).mean()
        S.ccmean_thr = cctsr[tsrmsk].mean()
        S.cc95prc_thr = np.percentile(cctsr[tsrmsk], 95)
        S.cc80prc_thr = np.percentile(cctsr[tsrmsk], 80)
        S_col.append(S)

corrtsr_tab = pd.DataFrame(S_col)
#%
# cctsr_sparse = rectify_tsr(cctsr, rect_mode, thresh, Ttsr=Ttsr)
Tnum = corrtsr_tab.sparse_num.describe(percentiles=[0.1,0.9])
Tprc = corrtsr_tab.sparse_prct.describe(percentiles=[0.1,0.9])
Tcc = corrtsr_tab.ccmean_thr.describe(percentiles=[0.1,0.9])
print("remained unit number range from %d(%.1f%% of all units) to %d(%.1f%%) (10,90 percen-tile across experiments),"\
    " with mean correlation from %.3f to %.3f). " % (Tnum["10%"],100*Tprc["10%"],Tnum["90%"],100*Tprc["90%"],
                                                    Tcc["10%"],Tcc["90%"]))

#%%
from glob import glob
modelroot = r"E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis\models"
# modelstr = "resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust"
modelstr = "resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust"
csvpath = glob(join(modelroot, modelstr, "*.csv"))[0]
exptab = pd.read_csv(csvpath)
# valmsk = exptab.Expi>0
valmsk = ~((exptab.Animal=="Alfa") * (exptab.Expi==10))
for space in ["manif", "all"]:
    print("For Images in %s space mean correlation of model prediction and actual response is %.3f+-%.3f, "
          "normalized by the noise ceiling, the mean correlation is %.3f+-%.3f (N=%d)"%\
        (space, exptab[valmsk]["cc_aft_"+space].mean(),
        exptab[valmsk]["cc_aft_"+space].sem(),
        exptab[valmsk]["cc_aft_norm_"+space].mean(),
        exptab[valmsk]["cc_aft_norm_"+space].sem(), sum(valmsk),))


