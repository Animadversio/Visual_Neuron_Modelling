"""Plotting Routine for CorrFeatTsr """
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
from CorrFeatTsr_utils import area_mapping, multichan2rgb, saveallforms
from data_loader import mat_path, loadmat, load_score_mat
from glob import glob
import pickle as pkl
def showimg(ax, imgarr, cbar=False, ylabel=None, title=None, clim=None):
    if clim is None:
        pcm = ax.imshow(imgarr)
    else:
        pcm = ax.imshow(imgarr, vmin=clim[0], vmax=clim[1],)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title(title)
    if cbar:
        plt.colorbar(pcm, ax=ax)
    return pcm


def off_axes(axs):
    for ax in axs:
        ax.axis("off")
#%
ReprStats_col = EasyDict()
EStats_col = EasyDict()
MStats_col = EasyDict()
for Animal in ["Alfa", "Beto"]:
    ReprStats_col[Animal] = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), \
        struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    MStats_col[Animal] = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), \
        struct_as_record=False, squeeze_me=True)['Stats']
    EStats_col[Animal] = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), \
        struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']

modelroot = r"E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis\models"

#%% Text report: Calculate the sparseness ratio and the corresponding correlation value.
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

#%% Summarize Performance Print stats string for models.
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

#%% Generate Supplementary figure (S4A) that compares the factors extracted by from models with each other.
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
Explist =  [("Alfa", 4),
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


#%% Supplementary Figure S4C Comparing the CorrFeatTsr mean, max, num, and the NMF factors
Explist = [("Alfa", 4),
        ("Alfa", 15),
        ("Alfa", 20),
        ("Beto", 4),
        ("Beto", 7),
        ("Beto", 26),
        ("Beto", 29),]
outputdir = r"O:\Manuscript_Manifold\FigureS4C\Example"
modelstr = "resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust"
exp_suffix = "_nobdr_res-robust"
modellabel = "resnet50_robust_layer3"
layer = "layer3"
rect_mode = "Tthresh"; thresh = (None, 3)
for Animal, Expi in Explist[:]:
    data = pkl.load(open(join(modelroot, modelstr, "%s_Exp%02d_factors.pkl" % (Animal, Expi)), 'rb'))
    imgsize = data.AllStat.imgsize
    imgpos = data.AllStat.imgpos
    pref_chan = data.AllStat.pref_chan
    area = area_mapping(pref_chan)
    imgpix = int(imgsize * 40)
    explabel = "%s Exp%02d Driver Chan %d, %.1f deg [%s]" % (Animal, Expi, pref_chan, imgsize, tuple(imgpos))
    bdr = data.bdr
    NF = data.Hmaps.shape[2]
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
    covtsr_rect = rectify_tsr(covtsr, mode="Tthresh", thr=thresh, Ttsr=Ttsr)
    cctsr_rect = rectify_tsr(cctsr, mode="Tthresh", thr=thresh, Ttsr=Ttsr)
    manif_proto = ReprStats_col[Animal][Expi - 1].Manif.BestImg

    nrow, ncol = 3, 5
    figh, axs = plt.subplots(nrow, ncol, figsize=[ncol * 2.25, nrow * 2.1 + 0.5,])#,
    showimg(axs[0, 0], manif_proto, title="")
    showimg(axs[0, 1], covtsr_rect.max(axis=0), title="thresh cov max")
    showimg(axs[0, 2], covtsr_rect.mean(axis=0), title="thresh cov mean")
    showimg(axs[0, 3], np.abs(covtsr).max(axis=0), title="abs cov max")
    showimg(axs[0, 4], np.abs(covtsr).mean(axis=0), title="abs cov mean")
    # showimg(axs[0, 5], (covtsr_rect>0).sum(axis=0), title="thresh cov num")
    showimg(axs[1, 1], cctsr_rect.max(axis=0), title="thresh corr max")
    showimg(axs[1, 2], cctsr_rect.mean(axis=0), title="thresh corr mean")
    showimg(axs[1, 3], np.abs(cctsr).max(axis=0), title="abs corr max")
    showimg(axs[1, 4], np.abs(cctsr).mean(axis=0), title="abs corr mean")
    # showimg(axs[1, 5], (cctsr_rect>0).sum(axis=0), title="thresh corr num")
    featvec_norm = np.linalg.norm(data.ccfactor, axis=0)
    Hmaps_norm = multichan2rgb(data.Hmaps * featvec_norm)
    # Hmaps_norm = multichan2rgb(data.Hmaps)
    Hmaps_pad = np.pad(Hmaps_norm, [(bdr, bdr), (bdr, bdr), (0, 0)],
                     mode="constant", constant_values=np.nan)
    # PD = data.PredData
    # AllStat = data.AllStat
    showimg(axs[2, 0], Hmaps_pad, title="NMF 3")
    for fi in range(NF):
        showimg(axs[2, fi+1], Hmaps_pad[:, :, fi], title="Factor %d"%(fi+1), cbar=False, clim=[0, 1])
        # showimg(axs[2, fi+1], fact_protos[fi])
    off_axes([axs[1, 0], axs[2, 4]])
    figh.suptitle(explabel)
    figh.tight_layout()
    figh.savefig(join(outputdir, "%s_Exp%02d_tsr_summary_cmp_%s.png" % (Animal, Expi, modellabel)),
                bbox_inches='tight')
    figh.savefig(join(outputdir, "%s_Exp%02d_tsr_summary_cmp_%s.pdf" % (Animal, Expi, modellabel)),
                bbox_inches='tight')
    figh.show()

#%% Supplementary Figure S4O
from featvis_lib import vis_feattsr, vis_feattsr_factor, vis_featvec_wmaps
from CorrFeatTsr_predict_lib import visualize_fulltsrModel, visualize_factorModel
from GAN_utils import upconvGAN
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%
def get_cctsrs(Animal, Expi, exp_suffix, layer, nan2num=True):
    corrDict = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_Evol%s_corrTsr.npz" % (Animal, Expi, exp_suffix)),
                       allow_pickle=True)
    cctsr_dict = corrDict.get("cctsr").item()
    Ttsr_dict = corrDict.get("Ttsr").item()
    stdtsr_dict = corrDict.get("featStd").item()
    covtsr_dict = {layer: cctsr_dict[layer] * stdtsr_dict[layer] for layer in cctsr_dict}
    Ttsr = Ttsr_dict[layer]
    cctsr = cctsr_dict[layer]
    covtsr = covtsr_dict[layer]
    if nan2num:
        Ttsr = np.nan_to_num(Ttsr)
        cctsr = np.nan_to_num(cctsr)
        covtsr = np.nan_to_num(covtsr)
    return cctsr, Ttsr, covtsr


#%% Add visualization to CV prediction examples
ExpAll = [("Alfa", Expi) for Expi in range(1, 47)] + [("Beto", Expi) for Expi in range(1, 46)]
NFmodstr = "resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust_CV"
Fullmodstr = "resnet50_linf8-layer3_Full_bdr1_Tthresh_3__nobdr_res-robust_CV"
NFdir = join(modelroot, NFmodstr)
Fulldir = join(modelroot, Fullmodstr)
netname = "resnet50_linf8"; layer = "layer3"; exp_suffix="_nobdr_res-robust"; rect_mode = "Tthresh"; thr = (None, 3); bdr = 1
featvis_mode = "corr"; 
featnet, net = load_featnet(netname)
for Animal, Expi in ExpAll[:]:#Explist[:]:
    ReprStats = ReprStats_col[Animal]
    EStats = EStats_col[Animal]
    imgsize = EStats[Expi - 1].evol.imgsize
    imgpos = EStats[Expi - 1].evol.imgpos
    pref_chan = EStats[Expi - 1].evol.pref_chan
    area = area_mapping(pref_chan)
    imgpix = int(imgsize * 40)
    explabel = "%s Exp%02d Driver Chan %d, %.1f deg [%s]\nCCtsr %s-%s sfx:%s bdr%d rect %s Fact" % (Animal, Expi,\
         pref_chan, imgsize, tuple(imgpos), netname, layer, exp_suffix, bdr, rect_mode, )

    print("Processing "+explabel)
    corrDict = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_Evol%s_corrTsr.npz" % (Animal, Expi, exp_suffix)),
                                   allow_pickle=True)
    cctsr_dict = corrDict.get("cctsr").item()
    Ttsr_dict = corrDict.get("Ttsr").item()
    stdtsr_dict = corrDict.get("featStd").item()
    covtsr_dict = {layer: cctsr_dict[layer] * stdtsr_dict[layer] for layer in cctsr_dict}
    Ttsr = np.nan_to_num(Ttsr_dict[layer])
    cctsr = np.nan_to_num(cctsr_dict[layer])
    covtsr = np.nan_to_num(covtsr_dict[layer])
    manif_proto = ReprStats[Expi - 1].Manif.BestImg
    # Remake visualization for Full Tensor model 
    DR_Wtsr = rectify_tsr(covtsr, rect_mode, thr, Ttsr=Ttsr)

    data = pkl.load(open(join(Fulldir, "%s_Exp%02d_factors.pkl"%(Animal, Expi)), 'rb'))
    tsrimgs, mtg, score_traj = vis_feattsr(DR_Wtsr, net, G, layer, netname=netname, score_mode=featvis_mode,
            featnet=featnet, Bsize=5, saveImgN=1, bdr=bdr, figdir=Fulldir, savestr=featvis_mode,
            saveimg=False, imshow=False)
    tsr_proto = tsrimgs[0, :, :, :].permute([1, 2, 0]).numpy()  # shape [256, 256, 3] numpy array
    AllStat, PredData = data.AllStat, data.PredData
    figh = visualize_fulltsrModel(AllStat, PredData, manif_proto, DR_Wtsr, explabel+" Full", \
        savestr="%s_Exp%02d"%(Animal, Expi), figdir=Fulldir, tsr_proto=tsr_proto, bdr=bdr)
    data.tsr_proto = tsr_proto
    pkl.dump(data, open(join(Fulldir, "%s_Exp%02d_factors.pkl" % (Animal, Expi)), 'wb'))

    data = pkl.load(open(join(NFdir, "%s_Exp%02d_factors.pkl"%(Animal, Expi)), 'rb'))
    AllStat, PredData = data.AllStat, data.PredData
    ccfactor, Hmaps = data.ccfactor, data.Hmaps
    NF = ccfactor.shape[1]
    factimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, layer, netname=netname,
                 score_mode=featvis_mode, featnet=featnet, bdr=bdr, Bsize=5, saveImgN=1,
                 figdir=NFdir, savestr="corr", imshow=False, saveimg=False, show_featmap=False)
    tsrimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname,
                  score_mode=featvis_mode, featnet=featnet, Bsize=5, saveImgN=1, bdr=bdr, figdir=NFdir, savestr="corr",
                  saveimg=False, imshow=False)
    fact_protos = [factimgs[0, :, :, :].permute([1, 2, 0]).numpy() for factimgs in factimgs_col]
    tsr_proto = tsrimgs[0, :, :, :].permute([1, 2, 0]).numpy()
    figh = visualize_factorModel(AllStat, PredData, manif_proto, Hmaps, ccfactor, explabel+" %d" % NF, \
        savestr="%s_Exp%02d"%(Animal, Expi), figdir=NFdir, fact_protos=fact_protos, tsr_proto=tsr_proto, bdr=bdr)
    data.fact_protos = fact_protos
    data.tsr_proto = tsr_proto
    pkl.dump(data, open(join(NFdir, "%s_Exp%02d_factors.pkl" % (Animal, Expi)), 'wb'))

#%% Visualize Experiments and Show Examples (Factor model and full model)
outdir = "O:\Manuscript_Manifold\FigureS4\Examples"
Explist = [# ("Beto", 11),
           # ("Alfa", 45),
           # ("Beto", 30),
           # ("Alfa", 36),
           ("Alfa", 20),
           ]  #
plotpred = True
NFmodstr = "resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust_CV"
Fullmodstr = "resnet50_linf8-layer3_Full_bdr1_Tthresh_3__nobdr_res-robust_CV"
NFdir = join(modelroot, NFmodstr)
Fulldir = join(modelroot, Fullmodstr)
netname = "resnet50_linf8"; layer = "layer3"; exp_suffix="_nobdr_res-robust"; rect_mode = "Tthresh"; thr = (None, 3); bdr = 1
for Animal, Expi in Explist[:]:
    ReprStats = ReprStats_col[Animal]
    EStats = EStats_col[Animal]
    manif_proto = ReprStats[Expi - 1].Manif.BestImg
    imgsize = EStats[Expi - 1].evol.imgsize
    imgpos = EStats[Expi - 1].evol.imgpos
    pref_chan = EStats[Expi - 1].evol.pref_chan
    area = area_mapping(pref_chan)
    imgpix = int(imgsize * 40)
    explabel = "%s Exp%02d Driver Chan %d, %.1f deg [%s]\nCCtsr %s-%s sfx:%s bdr%d rect %s Full and NF3" % \
           (Animal, Expi, pref_chan, imgsize, tuple(imgpos), netname, layer, exp_suffix, bdr, rect_mode,)
    NFdata = pkl.load(open(join(NFdir, "%s_Exp%02d_factors.pkl" % (Animal, Expi)), 'rb'))
    Fulldata = pkl.load(open(join(Fulldir, "%s_Exp%02d_factors.pkl" % (Animal, Expi)), 'rb'))
    AS_NF, PD_NF = NFdata.AllStat, NFdata.PredData
    AS_Fu, PD_Fu = Fulldata.AllStat, Fulldata.PredData
    ccfactor, Hmaps = NFdata.ccfactor, NFdata.Hmaps
    facttsr_proto, fact_protos = NFdata.tsr_proto, NFdata.fact_protos
    tsr_proto = Fulldata.tsr_proto
    NF = ccfactor.shape[1]
    cctsr, Ttsr, covtsr = get_cctsrs(Animal, Expi, exp_suffix, layer, nan2num=True)
    DR_Wtsr = rectify_tsr(covtsr, rect_mode, thr, Ttsr=Ttsr)
    Fulltsr_msk = DR_Wtsr.mean(axis=0)

    nrow, ncol = 3, 6
    figh, axs = plt.subplots(nrow, ncol, figsize=[ncol * 2.1, nrow * 2.1 + 0.5,])#,
    showimg(axs[0, 0], manif_proto, title="Manif Proto")
    showimg(axs[0, 1], tsr_proto, title="FullTsr Proto")
    showimg(axs[0, 2], facttsr_proto, title="FactorTsr Proto")
    showimg(axs[1, 1], Fulltsr_msk, title="FullTsr (chan mean)")
    featvec_norm = np.linalg.norm(ccfactor, axis=0)
    Hmaps_norm = multichan2rgb(Hmaps * featvec_norm)
    # Hmaps_norm = multichan2rgb(data.Hmaps)
    Hmaps_pad = np.pad(Hmaps_norm, [(bdr, bdr), (bdr, bdr), (0, 0)],
                       mode="constant", constant_values=np.nan)
    showimg(axs[1, 2], Hmaps_pad, title="Factor Msk")
    for fi in range(NF):
        showimg(axs[fi, 3], fact_protos[fi], )#title="Factor %d" % (fi + 1),
        showimg(axs[fi, 4], Hmaps_pad[:, :, fi], cbar=False, clim=[0, 1])#title="Factor %d" % (fi + 1),
        axs[fi, 5].plot(ccfactor[:, fi], lw=0.5, alpha=0.5)
        axs[fi, 5].plot(sorted(ccfactor[:, fi]), lw=0.5, alpha=0.5)

    if plotpred:
        imglabel = AS_NF.Nimg_manif * ["manif"] + AS_NF.Nimg_gabor * ["gabor"] + \
                   AS_NF.Nimg_pasu * ["pasu"] + AS_NF.Nimg_evoref * ["evoref"]

        sns.scatterplot(x=PD_Fu.pred_scr_all, y=PD_Fu.score_vect_all, hue=imglabel, alpha=0.5, ax=axs[2, 1])
        axs[2, 1].annotate("cc %.3f\n(%.3f)" % (AS_Fu.cc_aft_all, AS_Fu.cc_aft_norm_all),
                                     xy=(0.1, 0.8), xycoords="axes fraction")
        axs[2, 1].scatter(PD_Fu.pred_scr_all, PD_Fu.nlpred_scr_all, alpha=0.3, color='k', s=9)
        # axs[2, 1].set_ylabel("Observed\nResp")
        sns.scatterplot(x=PD_NF.pred_scr_all, y=PD_NF.score_vect_all, hue=imglabel, alpha=0.5, ax=axs[2, 2])
        axs[2, 2].annotate("cc %.3f\n(%.3f)" % (AS_NF.cc_aft_all, AS_NF.cc_aft_norm_all),
                                     xy=(0.1, 0.8), xycoords="axes fraction")
        axs[2, 2].scatter(PD_NF.pred_scr_all, PD_NF.nlpred_scr_all, alpha=0.3, color='k', s=9)

    off_axes(axs[1:, 0])
    figh.suptitle(explabel)
    figh.tight_layout()
    saveallforms(outdir, "%s_Exp%02d_FullFact_cmp"%(Animal, Expi))
    figh.show()

#%% FigureS4B Factor number comparison
from CorrFeatTsr_utils import area_mapping
from scipy.stats import spearmanr, f_oneway
outdir = r"O:\Manuscript_Manifold\FigureS4B\Examples"


def pred_perform_cmp(nf_csv_dict, label2num, statname, modelstr="net-layer", figdir=outdir, expmsk=None):
    """Well formed plot function for model performance as a function of factor number
    Separated by area and animal
    Report the mean, sem of factor number per area
    Report the progression statistics
    """
    sumtab = pd.DataFrame()
    xticks = []
    for lab, tabfn in nf_csv_dict.items():
        tab = pd.read_csv(tabfn)
        sumtab[lab] = tab[statname]
        xticks.append(label2num[lab])
    if expmsk is None:
        expmsk = np.ones(tab.shape[0], dtype=bool)
    xticks = np.array(xticks).reshape([-1,1])
    summarymat = np.array(sumtab)
    summarymat_norm = (summarymat / np.abs(summarymat).max(axis=1, keepdims=True))
    # if expmsk is not None:
    sumtab = sumtab[expmsk] # Note sumtab alread exclude the rows as masked out in sumtab
    summarymat = summarymat[expmsk]
    summarymat_norm = summarymat_norm[expmsk, :]

    figh, ax = plt.subplots(1, 3, figsize=[9, 6])
    area_str = ["V1","V4","IT"]
    msks = [tab[expmsk].area=="V1", tab[expmsk].area=="V4", tab[expmsk].area=="IT"]
    minormsks = [tab[expmsk].Animal=="Alfa", tab[expmsk].Animal=="Beto"]
    clrs = ["red", "green", "blue"]
    styles = ["-", "-."]
    for i, (clr, msk) in enumerate(zip(clrs, msks)):
        for mi, mmsk in enumerate(minormsks):
            xjit = np.random.randn(1,sum(msk&mmsk))*0.1
            ax[i].plot(xticks+xjit, summarymat_norm[msk&mmsk,:].T,
                       alpha=0.5, color=clr, linestyle=styles[mi])
        ax[i].set_title(area_str[i])
        ax[i].set_xlabel("Factor Number")
        ax[i].set_xticks(xticks[:,-1])
    ax[0].set_ylabel(statname+" norm to max")

    best_NFnum = sumtab.idxmax(axis=1).apply(lambda lab: label2num[lab])
    # best_NFnum["area"] = tab.area
    sumtab["best_NFnum"] = best_NFnum
    for colnm in ["Animal", "Expi", "area", "pref_chan"]:
        sumtab[colnm] = tab[expmsk][colnm]
    # Quick way to get summary statistics for each group
    summary = sumtab.groupby("area", sort=False).mean()
    summary_sem = sumtab.groupby("area", sort=False).sem()
    summary_cnt = sumtab.groupby("area", sort=False).size()
    print("Summarize statistics %s vs Number of factors"%statname)
    print(summary)
    # summary2 = sumtab.groupby(["Animal", "area"], sort=False).mean()
    # print(summary2)
    valmsk = ~sumtab.best_NFnum.isna()
    Fval, F_pval = f_oneway(sumtab.best_NFnum[(sumtab.area=="V1")&valmsk], sumtab.best_NFnum[(sumtab.area=="V4")&valmsk], \
                    sumtab.best_NFnum[(sumtab.area=="IT")&valmsk])
    
    areanummap = lambda A: {"V1": 1, "V4": 2, "IT": 3}[A]
    area_num = sumtab.area.apply(areanummap)
    rval, rpval = spearmanr(area_num[valmsk], best_NFnum[valmsk])
    print("Best NF number ~ area, ANOVA F %.2f (p=%.1e)"%(Fval, F_pval))
    print("Best NF number ~ area, Spearman R %.3f (p=%.1e)"%(rval, rpval))
    figh.suptitle("%s model prediction as function of NF\nbest factor N ~ area: ANOVA F %.2f (p=%.1e) Spearman R %.3f ("
                  "p=%.1e)" % (modelstr, Fval, F_pval, rval, rpval))
    for i, area in enumerate(area_str):
        ax[i].axvline(summary.best_NFnum[area], C='k', ls=":")
        ax[i].text(summary.best_NFnum[area], 1.0, '%.2f+-%.2f (N=%d)'%(summary.best_NFnum[area], summary_sem.best_NFnum[area], summary_cnt[area]), rotation=0)
        # ax[i].vlines(sumtab.best_NFnum[msks[i]].to_list(), 0, 1, color='k', ls="-.", alpha=0.1)
    plt.tight_layout()
    figh.savefig(join(figdir, "NF-modelAccur_area_sep_%s_%s.png"%(modelstr, statname)),
                bbox_inches='tight')
    figh.savefig(join(figdir, "NF-modelAccur_area_sep_%s_%s.pdf"%(modelstr, statname)),
                bbox_inches='tight')
    plt.show()
    return sumtab, summary, figh


label2num = {"NF1":1, "NF2":2, "NF3":3, "NF5":5, "NF7":7,  "NF9":9, "Full":1024, np.nan:np.nan}
nf_csv_list = {
 "NF%d"%(NF):join(modelroot,'resnet50_linf8-layer3_NF%d_bdr1_Tthresh_3__nobdr_res-robust_CV\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF%d_CV.csv')%(NF,NF)
     for NF in [1,2,3,5,7,9]
}
tmptab = pd.read_csv(nf_csv_list["NF1"])
valmsk = ~((tmptab.Animal=="Alfa") & (tmptab.Expi==10))

sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_manif", modelstr="resnet50_robust-layer3_CV", expmsk=valmsk)
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_manif", modelstr="resnet50_robust-layer3_CV", expmsk=valmsk)

#%% ResNet50-Robust layer2
nf_csv_list = {
 "NF%d"%(NF):join(modelroot,'resnet50_linf8-layer2_NF%d_bdr3_Tthresh_3__nobdr_res-robust_CV'
                            '\\Both_pred_stats_resnet50_linf8-layer2_Tthresh_bdr3_NF%d_CV.csv')%(NF,NF)
     for NF in [1,2,3,5,7,9]
}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_manif", modelstr="resnet50_robust-layer2_CV", expmsk=valmsk)
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_manif", modelstr="resnet50_robust-layer2_CV", expmsk=valmsk)
#%% VGG16
nf_csv_list = {
 'NF%d'%(NF): join(modelroot,'vgg16-conv4_3_NF%d_bdr1_Tthresh_3__nobdr_CV\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF%d_CV.csv')%(NF,NF)
    for NF in [1,2,3,5,7,9]
}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_manif", modelstr="VGG16-conv4_3_CV", expmsk=valmsk)
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_manif", modelstr="VGG16-conv4_3_CV", expmsk=valmsk)
#% Alexnet
nf_csv_list = {
"NF%d"%(NF): join(modelroot,'alexnet-conv4_NF%d_bdr1_Tthresh_3__nobdr_alex_CV\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF%d_CV.csv')%(NF,NF)
    for NF in [1,2,3,5,7,9]
}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_manif", modelstr="alexnet-conv4_CV", expmsk=valmsk)
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_manif", modelstr="alexnet-conv4_CV", expmsk=valmsk)
#% Resnet50
nf_csv_list = {
"NF%d"%(NF): join(modelroot,'resnet50-layer3_NF%d_bdr1_Tthresh_3__nobdr_resnet_CV\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF%d_CV.csv')%(NF,NF)
    for NF in [1,2,3,5,7,9]
}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_manif", modelstr="resnet50-layer3_CV", expmsk=valmsk)
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_manif", modelstr="resnet50-layer3_CV", expmsk=valmsk)



#%% Model comparison panel
# figroot = "O:\corrFeatTsr_FactorVis"
# sumdir = join(figroot, "summary")
outdir = r"O:\Manuscript_Manifold\Figure4\Final_Elements"
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

    figh = plt.figure(figsize=(4.25,4))
    sns.scatterplot(x=tab1[varnm][mask], y=tab2[varnm][mask], hue=masktab[colorvar][mask], style=masktab[stylevar][mask])
    plt.plot([0,1],[0,1],linestyle=":", c='k', lw=1)
    plt.ylabel(explab2);plt.xlabel(explab1)
    plt.gca().set_aspect('equal', adjustable='box')  # datalim
    cc = ma.corrcoef(ma.masked_invalid(tab1[varnm][mask]), ma.masked_invalid(tab2[varnm][mask]))[0, 1]
    # cc = np.corrcoef(tab1[varnm], tab2[varnm])
    tval, pval = ttest_rel(np.arctanh(tab1[varnm][mask]), np.arctanh(tab2[varnm][mask]), nan_policy='omit') # ttest: exp1 - exp2
    plt.title("Linear model prediction %s comparison\ncc %.3f t test(Fisher z) %.2f (%.1e)"%(varnm, cc, tval, pval))
    plt.savefig(join(outdir, "models_pred_cmp_%s_%s_%s.png"%(varnm, explab1, explab2)))
    plt.savefig(join(outdir, "models_pred_cmp_%s_%s_%s.pdf"%(varnm, explab1, explab2)))
    plt.show()
    return figh

modelstr1 = "resnet50_linf8-layer3_Full_bdr1_Tthresh_3__nobdr_res-robust_CV"
exptab1 = pd.read_csv(glob(join(modelroot, modelstr1, "*.csv"))[0])
modelstr2 = "resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust_CV"
exptab2 = pd.read_csv(glob(join(modelroot, modelstr2, "*.csv"))[0])
valmsk = ~((exptab1.Animal=="Alfa")&(exptab1.Expi==10))
pred_cmp_scatter(exptab1, exptab2, "ResNet-rbst-l3-Full", "ResNet-rbst-l3-NF3", mask=valmsk,
                 varnm="cc_aft_norm_manif", colorvar="area", stylevar="Animal", masktab=None)
#%% Compare the performance of full model vs small model.
varnm = "cc_aft_manif"
print("Statistics %s"%varnm)
for area in ["V1", "V4", "IT"]:
    msk = (exptab2.area == area)&valmsk
    tval, pval = ttest_rel(np.arctanh(exptab1[varnm][msk]), np.arctanh(exptab2[varnm][msk]),nan_policy="omit")
    print("%s cmp Full - NF3: %.3f (P=%.1e, df=%d)"%(area, tval, pval,sum(msk)-1))
