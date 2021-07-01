import os
from os.path import join
import pickle as pkl
from easydict import EasyDict
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
import numpy.ma as ma
from scipy.stats import ttest_rel
from featvis_lib import load_featnet, rectify_tsr
from CorrFeatTsr_utils import area_mapping, multichan2rgb
from data_loader import mat_path, loadmat, load_score_mat
from glob import glob
import pickle as pkl

modelroot = r"E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis\models"

#%% Calculate the sparseness ratio and the corresponding correlation value.
rect_mode = "Tthresh"; thresh = (None, 3)
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

#%% Print stats string for models.
modelstr = "resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust"
# modelstr = "resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust"
# modelstr = "resnet50_linf8-layer3_Full_bdr1_Tthresh_3__nobdr_res-robust_CV"
# modelstr = "resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust_CV"
modelstrs = ["resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust",
             "resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust",
             "resnet50_linf8-layer3_Full_bdr1_Tthresh_3__nobdr_res-robust_CV",
             "resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust_CV",]
for modelstr in modelstrs:
    csvpath = glob(join(modelroot, modelstr, "*.csv"))[0]
    exptab = pd.read_csv(csvpath)
    # valmsk = exptab.Expi>0
    valmsk = ~((exptab.Animal=="Alfa") * (exptab.Expi==10))
    print("For %s"%modelstr)
    for space in ["manif", "all"]:
        print("For Images in %s space mean correlation of model prediction and actual response is %.3f+-%.3f, "
              "normalized by the noise ceiling, the mean correlation is %.3f+-%.3f (N=%d)"%\
            (space, exptab[valmsk]["cc_aft_"+space].mean(),
            exptab[valmsk]["cc_aft_"+space].sem(),
            exptab[valmsk]["cc_aft_norm_"+space].mean(),
            exptab[valmsk]["cc_aft_norm_"+space].sem(), sum(valmsk),))
#%%
modelstr = "resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust_CV"
exptab = pd.read_csv(glob(join(modelroot, modelstr, "*.csv"))[0])
print("Factorized Weights account for %.3f+-%.3f variance of original weights"%(exptab[valmsk].exp_var.mean(),
                                                                          exptab[valmsk].exp_var.sem()))

#%% Generate Supplementary figure that compares models with each other.
def showimg(ax, imgarr, cbar=False, ylabel=None):
    pcm = ax.imshow(imgarr)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if cbar:
        plt.colorbar(pcm, ax=ax)
    return pcm

def off_axes(axs):
    for ax in axs:
        ax.axis("off")

ReprStats_col = EasyDict()
EStats_col = EasyDict()
MStats_col = EasyDict()
for Animal in ["Alfa", "Beto"]:
    ReprStats_col[Animal] = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)[
                'ReprStats']

modelstrs = ["resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust",
            "resnet50_linf8-layer2_NF3_bdr3_Tthresh_3__nobdr_res-robust",
            "resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet",
            "vgg16-conv4_3_NF3_bdr1_Tthresh_3__nobdr",
            "alexnet-conv4_NF3_bdr1_Tthresh_3__nobdr_alex",]
modellabels = ["ResNet50_Robust-layer3",
            "ResNet50_Robust-layer2",
            "ResNet50-layer3",
            "VGG16-conv4_3",
            "AlexNet-conv4",]
Explist = [
        ("Alfa", 4),
        ("Alfa", 15),
        ("Alfa", 20),
        ("Beto", 4),
        ("Beto", 7),
        ("Beto", 26),
        ("Beto", 29),]
outputdir = r"O:\Manuscript_Manifold\FigureS4A\Example"
NF = 3
plotpred = False # plot prediction comparison
for Animal, Expi in Explist[:]:
    nrow = 1+2*len(modelstrs)
    ncol = NF+1+plotpred
    if plotpred:
        figh, axs = plt.subplots(nrow, ncol, figsize=[ncol * 2.1, nrow * 2.13 + 0.5, ])  # ,
    else:
        figh, axs = plt.subplots(nrow, ncol, figsize=[ncol * 2.1, nrow * 2.1+0.5,])#,
    # constrained_layout=True
    for mi, modelstr in enumerate(modelstrs):
        exptab = pd.read_csv(glob(join(modelroot, modelstr, "*.csv"))[0])
        data = pkl.load(open(join(modelroot, modelstr, "%s_Exp%02d_factors.pkl"%(Animal, Expi)),'rb'))
        imgsize = data.AllStat.imgsize
        imgpos = data.AllStat.imgpos
        pref_chan = data.AllStat.pref_chan
        area = area_mapping(pref_chan)
        imgpix = int(imgsize * 40)
        explabel = "%s Exp%02d Driver Chan %d, %.1f deg [%s]" % (Animal, Expi, pref_chan, imgsize, tuple(imgpos))
        bdr = data.bdr
        manif_proto = ReprStats_col[Animal][Expi - 1].Manif.BestImg
        tsr_proto = data.tsr_proto
        fact_protos = data.fact_protos
        featvec_norm = np.linalg.norm(data.ccfactor, axis=0)
        Hmaps_norm = multichan2rgb(data.Hmaps * featvec_norm)
        # Hmaps_norm = multichan2rgb(data.Hmaps)
        Hmaps_pad = np.pad(Hmaps_norm, [(bdr, bdr), (bdr, bdr), (0, 0)],
                         mode="constant", constant_values=np.nan)
        PD = data.PredData
        AllStat = data.AllStat
        showimg(axs[0, 0], manif_proto, ylabel="manif proto")
        off_axes(axs[0, 1:])
        showimg(axs[2*mi+1, 0], Hmaps_pad)#, ylabel=modellabels[mi])
        showimg(axs[2*mi+2, 0], data.tsr_proto, ylabel=modellabels[mi])
        for fi in range(NF):
            showimg(axs[2*mi+1, fi+1], Hmaps_pad[:,:,fi])
            showimg(axs[2*mi+2, fi+1], fact_protos[fi])
        if plotpred:
            imglabel = AllStat.Nimg_manif*["manif"] + AllStat.Nimg_gabor*["gabor"] + \
                   AllStat.Nimg_pasu*["pasu"] + AllStat.Nimg_evoref*["evoref"]
            # axs[2*mi+1, -1].scatter(PD.pred_scr_manif, PD.nlpred_scr_manif, alpha=0.3, color='k', s=9)
            sns.scatterplot(x=PD.nlpred_scr_manif, y=PD.score_vect_manif, alpha=0.5, ax=axs[2*mi+1, -1])
            axs[2*mi+1, -1].annotate("cc %.3f (%.3f)"%(AllStat.cc_aft_manif, AllStat.cc_aft_norm_manif),
                            xy=(0.1, 0.8), xycoords="axes fraction")
            # axs[2*mi+1, -1].set_title()
            # axs[2*mi+2, -1].scatter(PD.pred_scr_all, PD.nlpred_scr_all, alpha=0.3, color='k', s=9)
            # axs[2*mi+2, -1].set_title()
            sns.scatterplot(x=PD.nlpred_scr_all, y=PD.score_vect_all, hue=imglabel, alpha=0.5, ax=axs[2*mi+2, -1])
            axs[2*mi+2, -1].annotate("cc %.3f (%.3f)"%(AllStat.cc_aft_all, AllStat.cc_aft_norm_all),
                            xy=(0.1, 0.8), xycoords="axes fraction")
    plt.suptitle(explabel)
    figh.tight_layout()
    plt.savefig(join(outputdir, "%s_Exp%02d_factor_cmp%s.png" % (Animal, Expi, "_pred" if plotpred else "")),
                bbox_inches='tight')
    plt.savefig(join(outputdir, "%s_Exp%02d_factor_cmp%s.pdf" % (Animal, Expi, "_pred" if plotpred else "")),
                bbox_inches='tight')
    figh.show()

#%%

#%% Comparing the CorrFeatTsr mean, max, num, and the NMF factors
exp_suffix = "_nobdr_res-robust"
layer = "layer2"
for Animal, Expi in Explist[:]:
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






#%%
figroot = "O:\corrFeatTsr_FactorVis"
sumdir = join(figroot, "summary")
def pred_cmp_scatter(tab1, tab2, explab1, explab2, varnm="cc_bef", colorvar="area", stylevar="Animal", masktab=None,
                     mask=None):
    """Compare prediction score for 2 models
    Plotting the scatter of prediction accuracy, separated by area and animal.
    """
    if masktab is None:
        masktab = tab1
    if mask is None:
        mask = np.ones(tab1.shape[0], dtype=bool)
    masktab["area"] = ""
    masktab["area"][masktab.pref_chan <= 32] = "IT"
    masktab["area"][(masktab.pref_chan <= 48) & (masktab.pref_chan >= 33)] = "V1"
    masktab["area"][masktab.pref_chan >= 49] = "V4"

    figh = plt.figure(figsize=(5.5,5))
    sns.scatterplot(x=tab1[varnm][mask], y=tab2[varnm][mask], hue=masktab[colorvar][mask], style=masktab[stylevar][mask])
    plt.plot([0,1],[0,1],linestyle=":", c='k', lw=1)
    plt.ylabel(explab2);plt.xlabel(explab1)
    plt.gca().set_aspect('equal', adjustable='box')  # datalim
    cc = ma.corrcoef(ma.masked_invalid(tab1[varnm][mask]), ma.masked_invalid(tab2[varnm][mask]))[0, 1]
    # cc = np.corrcoef(tab1[varnm], tab2[varnm])
    tval, pval = ttest_rel(np.arctanh(tab1[varnm][mask]), np.arctanh(tab2[varnm][mask]), nan_policy='omit') # ttest: exp1 - exp2
    plt.title("Linear model prediction %s comparison\ncc %.3f t test(Fisher z) %.2f (%.1e)"%(varnm, cc, tval, pval))
    plt.savefig(join(sumdir, "models_pred_cmp_%s_%s_%s.png"%(varnm, explab1, explab2)))
    plt.savefig(join(sumdir, "models_pred_cmp_%s_%s_%s.pdf"%(varnm, explab1, explab2)))
    plt.show()
    return figh

#%%
modelstr1 = "resnet50_linf8-layer3_Full_bdr1_Tthresh_3__nobdr_res-robust_CV"
exptab1 = pd.read_csv(glob(join(modelroot, modelstr1, "*.csv"))[0])
modelstr2 = "resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust_CV"
exptab2 = pd.read_csv(glob(join(modelroot, modelstr2, "*.csv"))[0])
valmsk = ~((exptab1.Animal=="Alfa")&(exptab1.Expi==10))
pred_cmp_scatter(exptab1, exptab2, "ResNet-rbst-l3-Full", "ResNet-rbst-l3-NF3", mask=valmsk,
                 varnm="cc_aft_norm_all", colorvar="area", stylevar="Animal", masktab=None)
