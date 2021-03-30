
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
from kornia.filters import gaussian_blur2d
from pixel_objectness import PixObjectiveNet
#%%
PNet = PixObjectiveNet(pretrained=True).eval().cuda()
PNet.requires_grad_(False)
#%%
ccdir = "S:\corrFeatTsr"
figdir = r"O:\ProtoObjectivenss\summary_batch"
mat_path = r"O:\Mat_Statistics"
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
        np.savez(join(datadir, "%s_Exp%02d_Batch_PixObj.npz"), **S)

        visualize_result(objmap.mean(dim=0, keepdims=True), Reprimg, titstr="%s Exp%02d EvolBlock Best image Batch Avg Mask" % (Animal, Expi), savenm="%s_Exp%02d_EvolBlock_batchavgmsk" % (Animal, Expi), figdir=figdir)
        # plt.imshow((objmap[:,1,:,:]-objmap[:,0,:,:]).mean(dim=0).numpy())
        # plt.colorbar()
        # plt.show()
#%% summary batch compute
