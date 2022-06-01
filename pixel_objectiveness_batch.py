
import sys
from os.path import join
import matplotlib.pylab as plt
import numpy as np
import numpy.ma as ma
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import ttest_rel, ttest_ind, ranksums, pearsonr
import torch
from easydict import EasyDict
from skimage.transform import resize
from skimage.io import imread
from kornia.filters import gaussian_blur2d
from pixel_objectness import PixObjectiveNet
#%%
PNet = PixObjectiveNet(pretrained=True).eval().cuda()
PNet.requires_grad_(False)
#%%
ccdir = "S:\corrFeatTsr"
figdir = r"O:\ProtoObjectivenss\summary_batch"
mat_path = r"O:\Mat_Statistics"
#%%
outlabel = "centRFintp"
Scol = []
for Animal in ["Alfa", "Beto"]:
    # Load summary stats for each animal
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    for Expi in range(1, len(EStats)+1):
        # Prep meta info
        imgsize = EStats[Expi - 1].evol.imgsize
        imgpos = EStats[Expi - 1].evol.imgpos
        pref_chan = EStats[Expi - 1].evol.pref_chan
        metadict = {"Animal":Animal, "Expi":Expi, "imgsize":imgsize, "imgpos":imgpos, "pref_chan":pref_chan}
        imgpix = int(imgsize * 40)
        titstr = "%s Exp %d Driver Chan %d, %.1f deg [%s]" % (Animal, Expi, pref_chan, imgsize, tuple(imgpos))
        print(titstr)
        # import images and pre-process
        img = ReprStats[Expi-1].Evol.BestBlockAvgImg
        imgtsr = torch.from_numpy(img).float().permute([2,0,1]).unsqueeze(0)
        imgtsr_pp = gaussian_blur2d(imgtsr, (5, 5), (3, 3))
        # get objectivity map
        objmap = PNet(imgtsr_pp.cuda(), fullmap=True).cpu()
        objmsk = (objmap[:, 0, :, :] < objmap[:, 1, :, :]).numpy()[0]
        probmap_rel = (objmap[:, 1, :, :] - objmap[:, 0, :, :]).numpy()[0]
        probmap = objmap[:, 1, :, :].numpy()[0]
        probmap_fg = np.copy(probmap)
        probmap_fg[~objmsk] = 0.0  # thresholded version of probmap

#%%
#%%
from glob import glob
def load_block_images(stimpath, block):
    pass

def load_stim_sets(stimpath, imgnms):
    imgnms_col = glob(stimpath + "\\*")
    imgfullpath_vect = [[path for path in imgnms_col if imgnm in path][0] for imgnm in imgnms]
    img_col = [imread(fp) for fp in imgfullpath_vect]
    return img_col

def preprocess_img_col(img_col, inputscale=255):
    imgtsr = torch.stack([torch.from_numpy(img).float().permute([2, 0, 1]) *255/ inputscale for img in img_col])
    imgtsr_pp = gaussian_blur2d(imgtsr, (5, 5), (3, 3))
    return imgtsr_pp

def score_traj(psth):
    # psth = EStats[Expi - 1].evol.psth
    if psth[0].ndim == 3:
        nunit = psth[0].shape[0]
    else:
        nunit = 1
    # psthmat = np.concatenate(tuple(np.reshape(P, [nunit, 200, -1]) for P in psth), axis=2)
    assert nunit == 1
    psth_col = [np.reshape(P, [nunit, 200, -1]) for P in psth]
    score_col = [P[0,50:200,:].mean(axis=0) for P in psth_col]
    score_mean = np.array([score.mean() for score in score_col])
    score_sem = np.array([np.std(score)/np.sqrt(len(score)) for score in score_col])
    return score_mean, score_sem, score_col

import os
from pixel_objectness import visualize_result
datadir = r"O:\ProtoObjectivenss\batch_data"
figdir = r"O:\ProtoObjectivenss\batch_fig"
os.makedirs(datadir, exist_ok=True)
os.makedirs(figdir, exist_ok=True)
for Animal in ["Alfa", "Beto"]:
    # Load summary stats for each animal
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    for Expi in range(1, len(EStats)+1):
        # Prep meta info
        imgsize = EStats[Expi - 1].evol.imgsize
        imgpos = EStats[Expi - 1].evol.imgpos
        pref_chan = EStats[Expi - 1].evol.pref_chan
        metadict = {"Animal":Animal, "Expi":Expi, "imgsize":imgsize, "imgpos":imgpos, "pref_chan":pref_chan}
        imgpix = int(imgsize * 40)
        titstr = "%s Exp %d Driver Chan %d, %.1f deg [%s]" % (Animal, Expi, pref_chan, imgsize, tuple(imgpos))
        print(titstr)
        Reprimg = ReprStats[Expi - 1].Evol.BestBlockAvgImg
        # Load the scores
        score_mean, score_sem, _ = score_traj(EStats[Expi - 1].evol.psth)
        best_block = score_mean[:-1].argmax()
        best_blk_idx = EStats[Expi - 1].evol.idx_seq[best_block]
        best_blk_imnms = EStats[Expi-1].imageName[best_blk_idx-1] # -1 to shift the matlab indexing convention.
        stimpath = EStats[Expi - 1].meta.stimuli
        stimpath = stimpath.replace(r"\\storage1.ris.wustl.edu\crponce\Active", r"N:")
        stimpath = stimpath.replace(r"N:", r"S:")
        best_blk_imgs = load_stim_sets(stimpath, best_blk_imnms)
        imgtsr_pp = preprocess_img_col(best_blk_imgs)
        with torch.no_grad():
            objmap = PNet(imgtsr_pp.cuda(), fullmap=True).cpu()

        S = EasyDict()
        S.objmap = objmap.numpy()
        S.best_blk_imnms = best_blk_imnms
        S.stimpath = stimpath
        S.score_mean = score_mean
        S.score_sem = score_sem
        S.best_block = best_block
        np.savez(join(datadir, "%s_Exp%02d_Batch_PixObj.npz"%(Animal, Expi)), **S)

        visualize_result(objmap.mean(dim=0, keepdims=True), Reprimg, titstr="%s Exp%02d EvolBlock Best image Batch Avg Mask" % (Animal, Expi), savenm="%s_Exp%02d_EvolBlock_batchavgmsk" % (Animal, Expi), figdir=figdir)
        # plt.imshow((objmap[:,1,:,:]-objmap[:,0,:,:]).mean(dim=0).numpy())
        # plt.colorbar()
        # plt.show()
#%% summary batch compute
def map_corr(corrmap, probmap):
    nas = np.logical_or(np.isnan(corrmap), np.isnan(probmap))
    rval, ccpval = pearsonr(corrmap[~nas].flatten(), probmap[~nas].flatten())
    return rval, ccpval

def compare_maps(corrmap, probmap, thresh=0.5, namestr="", suffix=""):
    """Calculate stats from 2 maps. Tstats, """
    nas = np.logical_or(np.isnan(corrmap), np.isnan(probmap))
    rval, ccpval = pearsonr(corrmap[~nas].flatten(), probmap[~nas].flatten())
    threshval =  thresh * (np.nanmax(corrmap) - np.nanmin(corrmap)) + np.nanmin(corrmap)
    corrmsk = corrmap > threshval
    tval, pval = ttest_ind(probmap[corrmsk], probmap[~corrmsk],nan_policy='omit')
    objin_m, objin_s, objin_MX = np.mean(probmap[corrmsk]), np.std(probmap[corrmsk]), np.max(probmap[corrmsk])
    objout_m, objout_s, objout_MX = np.mean(probmap[~corrmsk]), np.std(probmap[~corrmsk]), np.max(probmap[~corrmsk])
    # More complicated ways of comparing masks
    print("%s Objness Ttest T=%.3f (P=%.1e) corr=%.3f(P=%.1e)\nin msk %.3f(%.3f) out msk %.3f(%.3f)"%\
        (namestr, tval, pval, rval, ccpval, objin_m, objin_s, objout_m, objout_s, ))
    S = EasyDict()
    for varnm in ["rval", "ccpval", "tval", "pval", "rval", "ccpval", "objin_m", "objin_s", "objin_MX", "objout_m", "objout_s", "objout_MX"]:
        S[varnm+suffix] = eval(varnm)
    return S

def dict_union(*args):
    """Util function to pool dictionaries together"""
    S = EasyDict()
    for vdict in args:
        S.update(vdict)
    return S

from scipy.stats import pearsonr, spearmanr
from featvis_lib import load_featnet, rectify_tsr, tsr_factorize, tsr_posneg_factorize
mat_path = r'O:\Mat_Statistics'
sumdir = r"O:\ProtoObjectivenss\summary"
outlabel = "PN_batch-vs-NMF_resnet_robust_map"
# Compare this mask with the NMF mask
netname = "resnet50_linf8";layer = "layer3";bdr = 1;exp_suffix = "_nobdr_res-robust"
rect_mode = "Tthresh"; thresh = (None, 3)
NF = 3
init = "nndsvda"; solver="cd"; l1_ratio=0; alpha=0; beta_loss="frobenius" # default
showfig = False
explabel = ""
Scol = []
Animal, Expi = "Alfa", 3
for Animal in ["Alfa", "Beto"]:
    # Load summary stats for each animal
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    for Expi in range(1, len(EStats)+1):
        imgsize = EStats[Expi - 1].evol.imgsize
        imgpos = EStats[Expi - 1].evol.imgpos
        pref_chan = EStats[Expi - 1].evol.pref_chan
        metadict = {"Animal":Animal, "Expi":Expi, "imgsize":imgsize, "imgpos":imgpos, "pref_chan":pref_chan}

        corrDict = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_Evol%s_corrTsr.npz" % (Animal, Expi, exp_suffix)),
                                   allow_pickle=True)
        cctsr_dict = corrDict.get("cctsr").item()
        Ttsr_dict = corrDict.get("Ttsr").item()
        stdtsr_dict = corrDict.get("featStd").item()
        covtsr_dict = {layer: cctsr_dict[layer] * stdtsr_dict[layer] for layer in cctsr_dict}
        Ttsr = Ttsr_dict[layer]
        cctsr = cctsr_dict[layer]
        covtsr = covtsr_dict[layer]
        Ttsr = np.nan_to_num(Ttsr)
        cctsr = np.nan_to_num(cctsr)
        covtsr = np.nan_to_num(covtsr)
        Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr),
                         bdr=bdr, Nfactor=NF, init=init, solver=solver, l1_ratio=l1_ratio, alpha=alpha, beta_loss=beta_loss,
                         figdir=figdir, savestr="%s-%scov" % (netname, layer), suptit=explabel, show=showfig,)
        S = np.load(join(datadir, "%s_Exp%02d_Batch_PixObj.npz" % (Animal, Expi)))
        objmap_tsr = S.get("objmap")
        obj_msk = np.mean(objmap_tsr[:,0,:,:] < objmap_tsr[:,1,:,:], axis=0)
        Vnorm = np.sqrt((ccfactor**2).sum(axis=0))
        weighted_Hmap = Vnorm * Hmaps
        mergedHmap = np.abs(weighted_Hmap).sum(axis=2)
        mergedHmap_paded = np.pad(mergedHmap, [bdr, bdr], mode="constant", constant_values=np.nan)
        res = 256
        mergedHmap_rsz = resize(mergedHmap_paded, [res, res])
        # cval, ccpval = map_corr(mergedHmap_rsz, obj_msk)
        STS = compare_maps(mergedHmap_rsz, obj_msk)
        Stot = dict_union(metadict, STS)
        Scol.append(Stot)

Both_df = pd.DataFrame(Scol)
Both_df.to_csv(join(sumdir, "%s_cmp.csv"%outlabel))
#%%
V1msk = (Both_df.pref_chan < 49) & (Both_df.pref_chan > 32)
V4msk = Both_df.pref_chan > 48
ITmsk = Both_df.pref_chan < 33
Both_df["area"] = ""
Both_df.area[V1msk] = "V1"
Both_df.area[V4msk] = "V4"
Both_df.area[ITmsk] = "IT"
#%%
summary_df = Both_df.groupby("area").mean()
#%%
model_sum_fn = r"O:\corrFeatTsr_FactorVis\models\resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet" \
           r"\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF3.csv"
model_sum_tab = pd.read_csv(model_sum_fn)
pred_sucs_msk = (model_sum_tab.cc_aft_manif>0.5)
summary_df = Both_df[pred_sucs_msk].groupby("area").median()