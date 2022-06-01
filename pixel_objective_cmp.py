"""Really well structured script for testing pixel objectiveness of the feature xx selects for."""
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
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# Load the pixel Objectness network
from pixel_objectness import PixObjectiveNet
PNet = PixObjectiveNet(pretrained=True).eval().cuda()
PNet.requires_grad_(False)
#%% Evaluate correlation with correlation mask.
from easydict import EasyDict
from skimage.transform import resize
from skimage.io import imread
from kornia.filters import gaussian_blur2d
def norm_value(tsr):
    normtsr = (tsr - np.nanmin(tsr)) / (np.nanmax(tsr) - np.nanmin(tsr))
    return normtsr

def calc_map_from_tensors(D, res=256):
    """
    :Parameter
        D: Structure loaded. Created as in `make_savedict`
        res: Resolution of the full size mask, determines the interpolation process.
    Return
        maps: A EasyDict structure.
    """
    cctsr = D["cctsr"].item()
    Ttsr = D["Ttsr"].item()
    layers = list(cctsr.keys())
    maps = EasyDict()
    maps.mean = EasyDict()
    maps.max = EasyDict()
    maps.tsig_mean = EasyDict()
    maps.mean_rsz = EasyDict()
    maps.max_rsz = EasyDict()
    maps.tsig_mean_rsz = EasyDict()
    for layer in layers:
        # map of original small size
        maps.max[layer] = np.nanmax(np.abs(cctsr[layer]), axis=0)
        maps.mean[layer] = np.nansum(np.abs(cctsr[layer]), axis=0) / cctsr[layer].shape[0]
        cctsr_layer = np.copy(cctsr[layer])
        cctsr_layer[np.abs(Ttsr[layer]) < 8] = 0
        maps.tsig_mean[layer] = np.nansum(np.abs(cctsr_layer), axis=0) / cctsr[layer].shape[0]
        # resize to large image size
        maps.mean_rsz[layer] = norm_value(resize(maps.mean[layer], [res, res]))
        maps.max_rsz[layer] = norm_value(resize(maps.max[layer], [res, res]))
        maps.tsig_mean_rsz[layer] = norm_value(resize(maps.tsig_mean[layer], [res, res]))
    return maps


def visualize_msks(msk_list, label_list=None, titstr=None):
    """ Visualize a list of masks or maps as plt figure
    Parameter
        msk_list: list or generator of np arrays
        label_list: Title for each masks if given
        titstr: Title for the whole figure.
    Return:
        figh: Figure handle
    """
    nmsk = len(msk_list)
    figh, axs = plt.subplots(1, nmsk, figsize=[3*nmsk+1.5, 3])
    if nmsk == 1: axs = np.array([axs])
    else: axs = axs.reshape(-1)
    for i, msk in enumerate(msk_list):
        plt.sca(axs[i])
        plt.imshow(msk)
        if label_list is not None: plt.title(label_list[i])
        plt.axis("off")
        plt.colorbar()
    if titstr is not None:
        plt.suptitle(titstr)
    plt.show()
    return figh

def merge_msks(msk_list, weights=None):
    """Merge a list of masks with same size. """
    nmsk = len(msk_list)
    if weights is None:
        weights = np.ones(nmsk)
    mmsk = None
    for w, msk in zip(weights, msk_list):
        mmsk = (w * msk) if mmsk is None else mmsk + (w * msk)
    mmsk = mmsk / np.sum(weights)
    return mmsk
#%%

mmsk = merge_msks(mean_map_rsz.values(), [map.mean() for map in max_map.values()])
visualize_msks([mmsk])
#%%

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


def IoU(mask1, mask2):
    """Simple stats to compare 2 binary mask Intersect Over Union"""
    maskIntersect = np.logical_and(mask1,mask2)
    maskUnion = np.logical_or(mask1,mask2)
    IoU = np.nansum(maskIntersect) / np.nansum(maskUnion)
    return IoU


def compare_objmsk(corrmap, objectmask, thresh=0.5, namestr="", suffix=""):
    """Compare correlation map and objective masks"""
    threshval = thresh * (np.nanmax(corrmap) - np.nanmin(corrmap)) + np.nanmin(corrmap)
    corrmsk = corrmap > threshval
    maskIoU = IoU(corrmsk, objectmask)
    tval_cc, pval_cc = ttest_ind(corrmap[objectmask], corrmap[~objectmask],nan_policy='omit')
    corrin_m, corrin_s, corrin_MX = np.nanmean(corrmap[objectmask]), np.nanstd(corrmap[objectmask]), np.nanmax(corrmap[objectmask])
    corrout_m, corrout_s, corrout_MX = np.nanmean(corrmap[~objectmask]), np.nanstd(corrmap[~objectmask]), np.nanmax(corrmap[~objectmask])
    print("%s Correlation Ttest T=%.3f (P=%.1e), masks IoU=%.3f\nCorrelation in msk %.3f(%.3f) out msk %.3f(%.3f)"%\
        (namestr, tval_cc, pval_cc, maskIoU, corrin_m, corrin_s, corrout_m, corrout_s, ))
    S = EasyDict()
    for varnm in ["maskIoU", "corrin_m", "corrin_s", "corrin_MX", "corrout_m", "corrout_s", "corrout_MX"]:
        S[varnm + suffix] = eval(varnm)
    return S


def dict_union(*args):
    """Util function to pool dictionaries together"""
    S = EasyDict()
    for vdict in args:
        S.update(vdict)
    return S
#%% More refined mask using RF mapping of CNN units
sys.path.append("E:\Github_Projects\pytorch-receptive-field")
from torch_receptive_field import receptive_field, receptive_field_for_unit
from torchvision.models import alexnet, vgg16
from scipy.interpolate import griddata
net = vgg16()
RF_dict = receptive_field(net.features, (3, 224, 224), device="cpu")
# RF_for_unit = receptive_field_for_unit(RF_dict, (3, 224, 224), "8", (6,6))
layeridmap = {"conv2_2":"8",
            "conv3_3":"15",
            "conv4_3":"22",
            "conv5_3":"29"}
bdrmap = {"conv2_2": 8,
          "conv3_3": 4,
          "conv4_3": 2,
          "conv5_3": 1}

def RFinterp_map(maps_dict, res=256, RFfilter=False):
    maps_RFinterp = EasyDict()
    imgX, imgY = np.meshgrid(np.arange(224), np.arange(224))
    for layernm, map_curlayer in maps_dict.items():
        Ldict = RF_dict[layeridmap[layernm]]
        H, W = Ldict["output_shape"][-2:]
        jump = Ldict["j"]
        rsize = Ldict['r']
        start = Ldict["start"]
        Xvec = np.arange(W) * jump + start
        Yvec = np.arange(H) * jump + start
        XX, YY = np.meshgrid(Xvec, Yvec)
        bdr = bdrmap[layernm]
        borderMsk = np.ones([H-2*bdr, W-2*bdr], dtype=np.bool)
        borderMsk = np.pad(borderMsk, [(bdr, bdr), (bdr, bdr)], constant_values=0)
        RFinterpMap = griddata(np.array([XX[borderMsk],YY[borderMsk]]).T, map_curlayer[borderMsk], (imgX, imgY), fill_value=np.nan, method='linear') # fill in nan for out of border value
        maps_RFinterp[layernm] = RFinterpMap

    RFinterpMap_merge = merge_msks(list(maps_RFinterp.values()))
    maps_RFinterp_rsz = {layer: resize(map, [res, res]) for layer, map in maps_RFinterp.items()}
    RFinterpMap_merge_rsz = resize(RFinterpMap_merge, [res, res])

    return maps_RFinterp, RFinterpMap_merge, maps_RFinterp_rsz, RFinterpMap_merge_rsz

# maps_RFintp, RFintpMap_merge, maps_RFintp_rsz, RFintpMap_merge_rsz = RFinterp_map(maps.mean)
#%% Compare the Correlated Feature Mask with the
ccdir = "S:\corrFeatTsr"
figdir = r"O:\ProtoObjectivenss\summary"
mat_path = r"O:\Mat_Statistics"
# Animal = "Alfa"
outlabel = "centRFintp"
Scol = []
for Animal in ["Alfa", "Beto"]:
    # Load summary stats for each animal
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    for Expi in range(1, len(EStats)+1):
        imgsize = EStats[Expi - 1].evol.imgsize
        imgpos = EStats[Expi - 1].evol.imgpos
        pref_chan = EStats[Expi - 1].evol.pref_chan
        metadict = {"Animal":Animal, "Expi":Expi, "imgsize":imgsize, "imgpos":imgpos, "pref_chan":pref_chan}
        imgpix = int(imgsize * 40)
        titstr = "%s Exp %d Driver Chan %d, %.1f deg [%s]" % (Animal, Expi, pref_chan, imgsize, tuple(imgpos))
        print(titstr)
        img = ReprStats[Expi-1].Evol.BestBlockAvgImg
        imgtsr = torch.from_numpy(img).float().permute([2,0,1]).unsqueeze(0)
        imgtsr_pp = gaussian_blur2d(imgtsr, (5, 5), (3,3))
        objmap = PNet(imgtsr_pp.cuda(), fullmap=True).cpu()
        objmsk = (objmap[:, 0, :, :] < objmap[:, 1, :, :]).numpy()[0]
        probmap_rel = (objmap[:, 1, :, :] - objmap[:, 0, :, :]).numpy()[0]
        probmap = objmap[:, 1, :, :].numpy()[0]
        probmap_fg = np.copy(probmap)
        probmap_fg[~objmsk] = 0.0
        D = np.load(join(ccdir, "%s_Exp%d_Evol_nobdr_corrTsr.npz" % (Animal, Expi)),
                    allow_pickle=True)
        maps = calc_map_from_tensors(D)
        # layernm = "conv5_3"
        # corrmap = maps.mean_rsz[layernm]
        # layernm = "conv_merged"
        # corrmap = merge_msks(list(maps.mean_rsz.values()), weights=None)
        layernm = "conv_RfIntpMerged"
        maps_RFintp, RFintpMap_merge, maps_RFintp_rsz, RFintpMap_merge_rsz = RFinterp_map(maps.mean)
        corrmap = RFintpMap_merge_rsz
        S_IoU = compare_objmsk(corrmap, objmsk, thresh=0.5, namestr="%s vs object mask"%layernm)
        S = compare_maps(corrmap, probmap, namestr="%s vs prob"%layernm)
        S_fg = compare_maps(corrmap, probmap_rel, namestr="%s vs fg"%layernm, suffix="_fg")#probmap)
        S_cnt = compare_maps(corrmap[16:-16, 16:-16], probmap[16:-16, 16:-16], namestr="%s vs prob"%layernm, suffix="_cnt")
        figh = visualize_msks([img, corrmap, probmap], label_list=["img", "corrmap", "probmap"], titstr=titstr+" cent corr%.3f"%S_cnt.rval_cnt)
        figh.savefig(join(figdir, "%s_Exp%02d_objcorrmask_%s_cmp.png"%(Animal, Expi,outlabel)))
        Stot = dict_union(metadict, S_IoU, S, S_fg, S_cnt)
        Scol.append(Stot)
#%% Summary statistics
Both_df = pd.DataFrame(Scol)
Both_df.to_csv(join(figdir, "obj_corrmsk_%s_cmp.csv"%outlabel))
#%%
V1msk = (Both_df.pref_chan < 49) & (Both_df.pref_chan > 32)
V4msk = Both_df.pref_chan > 48
ITmsk = Both_df.pref_chan < 33
Both_df["area"] = ""
Both_df.area[V1msk] = "V1"
Both_df.area[V4msk] = "V4"
Both_df.area[ITmsk] = "IT"
#%%
BothTrajtab = pd.concat(pd.read_csv(join(mat_path, Animal+"_EvolTrajStats.csv")) \
          for Animal in ["Alfa","Beto"])
BothTrajtab = BothTrajtab.reset_index()
sucsmsk = BothTrajtab.t_p_initmax<1E-3
#%%
sns.stripplot(x=Both_df.area[sucsmsk], y=Both_df.rval_cnt[sucsmsk])
plt.show()
ttest_ind(Both_df.rval_cnt[ITmsk&sucsmsk], Both_df.rval_cnt[V1msk&sucsmsk],)
# ttest_ind(Both_df.rval_fg[ITmsk&sucsmsk], Both_df.rval_fg[V1msk&sucsmsk],)
#%%
sns.stripplot(x=Both_df.area[sucsmsk], y=Both_df.maskIoU[sucsmsk])
plt.show()
ttest_ind(Both_df.maskIoU[ITmsk&sucsmsk], Both_df.maskIoU[V1msk&sucsmsk],)
#%%
varnm = "objin_m_fg"
sns.stripplot(x=Both_df.area[sucsmsk], y=Both_df[varnm][sucsmsk])
plt.show()
ttest_ind(Both_df[varnm][ITmsk&sucsmsk], Both_df[varnm][V1msk&sucsmsk],)
#%%
visualize_msks(maps.max_rsz.values())
#%%
sns.regplot(BothTrajtab.t_initmax, Both_df.objin_m)
plt.show()
#%%
visualize_msks([RFinterpMap_merge_rsz]+list(maps_RFinterp_rsz.values()))
#%%

#%% Compare the Correlated Feature Mask with the
ccdir = "S:\corrFeatTsr"
figdir = r"O:\ProtoObjectivenss\summary"
mat_path = r"O:\Mat_Statistics"
outlabel = "NMF_Wmsk"
Scol = []
for Animal in ["Alfa", "Beto"]:
    # Load summary stats for each animal
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    for Expi in range(1, len(EStats)+1):
        imgsize = EStats[Expi - 1].evol.imgsize
        imgpos = EStats[Expi - 1].evol.imgpos
        pref_chan = EStats[Expi - 1].evol.pref_chan
        metadict = {"Animal":Animal, "Expi":Expi, "imgsize":imgsize, "imgpos":imgpos, "pref_chan":pref_chan}
        imgpix = int(imgsize * 40)
        titstr = "%s Exp %d Driver Chan %d, %.1f deg [%s]" % (Animal, Expi, pref_chan, imgsize, tuple(imgpos))
        print(titstr)
        img = ReprStats[Expi-1].Evol.BestBlockAvgImg
        imgtsr = torch.from_numpy(img).float().permute([2,0,1]).unsqueeze(0)
        imgtsr_pp = gaussian_blur2d(imgtsr, (5, 5), (3,3))
        objmap = PNet(imgtsr_pp.cuda(), fullmap=True).cpu()
        objmsk = (objmap[:, 0, :, :] < objmap[:, 1, :, :]).numpy()[0]
        probmap_rel = (objmap[:, 1, :, :] - objmap[:, 0, :, :]).numpy()[0]
        probmap = objmap[:, 1, :, :].numpy()[0]
        probmap_fg = np.copy(probmap)
        probmap_fg[~objmsk] = 0.0
        D = np.load(join(ccdir, "%s_Exp%d_Evol_nobdr_corrTsr.npz" % (Animal, Expi)),
                    allow_pickle=True)
        maps = calc_map_from_tensors(D)
        # layernm = "conv5_3"
        # corrmap = maps.mean_rsz[layernm]
        # layernm = "conv_merged"
        # corrmap = merge_msks(list(maps.mean_rsz.values()), weights=None)
        layernm = "conv_RfIntpMerged"
        maps_RFintp, RFintpMap_merge, maps_RFintp_rsz, RFintpMap_merge_rsz = RFinterp_map(maps.mean)
        corrmap = RFintpMap_merge_rsz
        S_IoU = compare_objmsk(corrmap, objmsk, thresh=0.5, namestr="%s vs object mask"%layernm)
        S = compare_maps(corrmap, probmap, namestr="%s vs prob"%layernm)
        S_fg = compare_maps(corrmap, probmap_rel, namestr="%s vs fg"%layernm, suffix="_fg")#probmap)
        S_cnt = compare_maps(corrmap[16:-16, 16:-16], probmap[16:-16, 16:-16], namestr="%s vs prob"%layernm, suffix="_cnt")
        figh = visualize_msks([img, corrmap, probmap], label_list=["img", "corrmap", "probmap"], titstr=titstr+" cent corr%.3f"%S_cnt.rval_cnt)
        figh.savefig(join(figdir, "%s_Exp%02d_objcorrmask_%s_cmp.png"%(Animal, Expi,outlabel)))
        Stot = dict_union(metadict, S_IoU, S, S_fg, S_cnt)
        Scol.append(Stot)
#%% Summary statistics
Both_df = pd.DataFrame(Scol)
Both_df.to_csv(join(figdir, "obj_corrmsk_%s_cmp.csv"%outlabel))