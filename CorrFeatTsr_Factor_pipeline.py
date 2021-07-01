"""
SUPER DUPER well written formulated pipeline for factorizations of Monkey A,B manifold experiments
# Super useful for hyper parameter tuning. Comfortable! SHUFU
"""
from featvis_lib import load_featnet, rectify_tsr, tsr_factorize, tsr_posneg_factorize, vis_feattsr, vis_featvec, \
    vis_feattsr_factor, vis_featvec_point, vis_featvec_wmaps, \
    CorrFeatScore, preprocess, show_img, pad_factor_prod
from CorrFeatTsr_predict_lib import fitnl_predscore, score_images, loadimg_preprocess
from CorrFeatTsr_utils import area_mapping, add_suffix, merge_dicts, multichan2rgb
import os
from os.path import join
import pickle as pkl
from easydict import EasyDict
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pylab as plt
from data_loader import mat_path, loadmat, load_score_mat
from GAN_utils import upconvGAN
import pandas as pd
import seaborn as sns
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
# os.system(r'subst N: E:\Network_Data_Sync') # do this if at home.
os.system(r'subst S: E:\Network_Data_Sync')
os.system(r'subst O: "E:\OneDrive - Washington University in St. Louis"')
os.system(r"subst N: \\storage1.ris.wustl.edu\crponce\Active")

def resample_correlation(scorecol, trial=100):
    """ Compute noise ceiling for correlating with a collection of noisy data"""
    resamp_scores_col = []
    for tri in range(trial):
        resamp_scores = np.array([np.random.choice(A, len(A), replace=True).mean() for A in scorecol])
        resamp_scores_col.append(resamp_scores)
    resamp_scores_arr = np.array(resamp_scores_col)
    resamp_ccmat = np.corrcoef(resamp_scores_arr)
    resamp_ccmat += np.diag(np.nan*np.ones(trial))
    split_cc_mean = np.nanmean(resamp_ccmat)
    split_cc_std = np.nanstd(resamp_ccmat)
    return split_cc_mean, split_cc_std


def predict_fit_dataset(DR_Wtsr, imgfullpath_vect, score_vect, scorecol, net, layer, netname="", featnet=None,\
                imgloader=loadimg_preprocess, batchsize=62, show=True, figdir="", savenm="pred", suptit=""):
    """ Use the weight tensor DR_Wtsr to do a linear model over 
        DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)

    :param imgfullpath_vect:
    :param score_vect:
    :param scorecol:
    :param net:
    :param layer:
    :param featnet:
    :param netname:
    :return:
    """
    if len(imgfullpath_vect)==0:
        return np.array([]), np.array([]), lambda x:x, EasyDict({"Nimg":0, "cc_bef":np.nan, "cc_aft":np.nan})
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    scorer.register_weights({layer: DR_Wtsr})
    pred_score = score_images(featnet, scorer, layer, imgfullpath_vect, imgloader=imgloader,
                                  batchsize=batchsize, ).numpy()
    scorer.clear_hook()
    nlfunc, popt, pcov, scaling, nlpred_score, PredStat = fitnl_predscore(pred_score, score_vect,
                      savedir=figdir, savenm=savenm, suptit=suptit, show=show)
    # Record stats and form population statistics
    if scorecol is not None: # compute the noise ceiling by resample from the trials. 
        corr_ceil_mean, corr_ceil_std = resample_correlation(scorecol, trial=100)
        for varnm in ["corr_ceil_mean", "corr_ceil_std"]:
            PredStat[varnm] = eval(varnm)
        PredStat.cc_aft_norm = PredStat.cc_aft / corr_ceil_mean  # prediction normalized by noise ceiling.
        PredStat.cc_bef_norm = PredStat.cc_bef / corr_ceil_mean
    return pred_score, nlpred_score, nlfunc, PredStat


def nlfit_merged_dataset(pred_score_col:list, score_vect_col:list, scorecol_col:list, figdir="", savenm="pred_all", suptit="",
                         show=True):
    """ Use the weight tensor DR_Wtsr to do a linear model over
        DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)

    :param imgfullpath_vect:
    :param score_vect:
    :param scorecol:
    :param net:
    :param layer:
    :param featnet:
    :param netname:
    :return:
    """
    pred_score_all = np.concatenate(tuple(pred_score_col), axis=0)
    score_vect_all = np.concatenate(tuple(score_vect_col), axis=0)
    nlfunc, popt, pcov, scaling, nlpred_score_all, PredStat = fitnl_predscore(pred_score_all, score_vect_all,
                      savedir=figdir, savenm=savenm, suptit=suptit, show=show)
    # Record stats and form population statistics
    if scorecol_col is not None:
        scorecol_all = [scores  for scorecol in scorecol_col for scores in scorecol]
        corr_ceil_mean, corr_ceil_std = resample_correlation(scorecol_all, trial=100)
        for varnm in ["corr_ceil_mean", "corr_ceil_std"]:
            PredStat[varnm] = eval(varnm)
        PredStat.cc_aft_norm = PredStat.cc_aft / corr_ceil_mean  # prediction normalized by noise ceiling.
        PredStat.cc_bef_norm = PredStat.cc_bef / corr_ceil_mean
    return pred_score_all, nlpred_score_all, score_vect_all, nlfunc, PredStat


def visualize_factorModel(AllStat, PD, protoimg, Hmaps, ccfactor, explabel, savestr="", figdir="", show=True,
                          fact_protos=None, tsr_proto=None, bdr=0):
    """Summarize the evaluations of a factor model."""
    if Hmaps is None: NF = 0
    else: NF = Hmaps.shape[2]
    ncol = max(4, NF+1)
    if fact_protos is not None and tsr_proto is not None:
        show_proto = True
        r_pad = 1
    else:
        show_proto = False
        r_pad = 0
    nrow = r_pad+3
    figh, axs = plt.subplots(nrow, ncol, squeeze=False, figsize=[ncol*4, 3*nrow])
    axs[0, 0].imshow(protoimg)
    axs[0, 0].axis("off")

    axs[1, 0].imshow(multichan2rgb(Hmaps))
    axs[1, 0].axis("off")
    for ci in range(NF):
        plt.sca(axs[0, 1+ci])
        Hmap_pad = np.pad(Hmaps[:, :, ci], [bdr, bdr],mode="constant",constant_values=np.nan)
        im = axs[0, 1+ci].imshow(Hmap_pad)
        axs[0, 1+ci].axis("off")
        figh.colorbar(im)
        axs[1, 1+ci].plot(ccfactor[:, ci], alpha=0.5)
        axs[1, 1+ci].plot([0, ccfactor.shape[0]], [0, 0], 'k-.')
        axs[1, 1+ci].plot(sorted(ccfactor[:, ci]), alpha=0.3) # this is smart, plot the sorted weight to show the distribution on yaxis
        axs[1, 1+ci].set_xlim([0, ccfactor.shape[0]+1])
    
    if show_proto:
        axs[2, 0].imshow(tsr_proto)
        axs[2, 0].axis("off")
        for ci in range(NF):
            axs[2, 1+ci].imshow(fact_protos[ci])
            axs[2, 1+ci].axis("off")

    
    axs[r_pad+2, 0].scatter(PD.pred_scr_manif, PD.nlpred_scr_manif, alpha=0.3, color='k', s=9)
    axs[r_pad+2, 0].scatter(PD.pred_scr_manif, PD.score_vect_manif, alpha=0.5, label="manif", s=25)
    axs[r_pad+2, 1].scatter(PD.nlpred_scr_manif, PD.score_vect_manif, alpha=0.5, s=25)
    axs[r_pad+2, 1].set_aspect(1, adjustable='datalim')
    axs[r_pad+2, 0].set_ylabel("Observed Scores")
    axs[r_pad+2, 0].set_xlabel("Factor Linear Pred")
    axs[r_pad+2, 1].set_xlabel("Factor Pred + nl")
    axs[r_pad+2, 0].set_title("Manif: Before NL Fit corr %.3f"%(AllStat.cc_bef_manif))
    axs[r_pad+2, 1].set_title("Manif: After NL Fit corr %.3f"%(AllStat.cc_aft_manif))
    axs[r_pad+2, 0].legend()

    imglabel = AllStat.Nimg_manif*["manif"] + AllStat.Nimg_gabor*["gabor"] + \
               AllStat.Nimg_pasu*["pasu"] + AllStat.Nimg_evoref*["evoref"]
    axs[r_pad+2, 2].scatter(PD.pred_scr_all, PD.nlpred_scr_all, alpha=0.3, color='k', s=9)
    sns.scatterplot(x=PD.pred_scr_all, y=PD.score_vect_all, hue=imglabel, alpha=0.5, ax=axs[r_pad+2, 2])
    sns.scatterplot(x=PD.nlpred_scr_all, y=PD.score_vect_all, hue=imglabel, alpha=0.5, ax=axs[r_pad+2, 3])
    axs[r_pad+2, 3].set_aspect(1, adjustable='datalim')
    axs[r_pad+2, 2].set_ylabel("Observed Scores")
    axs[r_pad+2, 2].set_xlabel("Factor Linear Pred")
    axs[r_pad+2, 3].set_xlabel("Factor Pred + nl")
    axs[r_pad+2, 2].set_title("All: Before NL Fit corr %.3f"%(AllStat.cc_bef_all))
    axs[r_pad+2, 3].set_title("All: After NL Fit corr %.3f"%(AllStat.cc_aft_all))
    axs[r_pad+2, 2].legend()
    figh.suptitle(explabel, fontsize=14)
    figh.savefig(join(figdir, "%s_summary.png"%savestr))
    figh.savefig(join(figdir, "%s_summary.pdf"%savestr))
    if show:
        figh.show()
        return figh
    else:
        figh.close()
        return None


def visualize_fulltsrModel(AllStat, PD, protoimg, Wtsr, explabel, savestr="", figdir="", show=True,
                          tsr_proto=None, bdr=0):
    """Summarize the evaluations of a full tsr linear model."""
    ncol = 4#max(4, NF+1)
    if tsr_proto is not None:
        show_proto = True
        r_pad = 1
    else:
        show_proto = False
        r_pad = 0
    nrow = r_pad+2
    figh, axs = plt.subplots(nrow, ncol, squeeze=False, figsize=[ncol*4, nrow*3])
    axs[0, 0].imshow(protoimg)
    axs[0, 0].axis("off")

    meanMap = np.abs(Wtsr).mean(axis=0)
    maxMap = np.abs(Wtsr).max(axis=0)
    for ci, (Wmap, name) in enumerate(\
        zip([meanMap,maxMap],["mean","max"])):
        plt.sca(axs[0, 1+ci])
        im = axs[0, 1+ci].imshow(Wmap)
        axs[0, 1+ci].axis("off")
        figh.colorbar(im)
        axs[0, 1+ci].set_title("W map: %s sum"%name)
    
    if show_proto:
        axs[1, 0].imshow(tsr_proto)
        axs[1, 0].axis("off")
    
    axs[r_pad+1, 0].scatter(PD.pred_scr_manif, PD.nlpred_scr_manif, alpha=0.3, color='k', s=9)
    axs[r_pad+1, 0].scatter(PD.pred_scr_manif, PD.score_vect_manif, alpha=0.5, label="manif", s=25)
    axs[r_pad+1, 1].scatter(PD.nlpred_scr_manif, PD.score_vect_manif, alpha=0.5, s=25)
    axs[r_pad+1, 1].set_aspect(1, adjustable='datalim')
    axs[r_pad+1, 0].set_ylabel("Observed Scores")
    axs[r_pad+1, 0].set_xlabel("Factor Linear Pred")
    axs[r_pad+1, 1].set_xlabel("Factor Pred + nl")
    axs[r_pad+1, 0].set_title("Manif: Before NL Fit corr %.3f"%(AllStat.cc_bef_manif))
    axs[r_pad+1, 1].set_title("Manif: After NL Fit corr %.3f"%(AllStat.cc_aft_manif))
    axs[r_pad+1, 0].legend()

    imglabel = AllStat.Nimg_manif*["manif"] + AllStat.Nimg_gabor*["gabor"] + \
               AllStat.Nimg_pasu*["pasu"] + AllStat.Nimg_evoref*["evoref"]
    axs[r_pad+1, 2].scatter(PD.pred_scr_all, PD.nlpred_scr_all, alpha=0.3, color='k', s=9)
    sns.scatterplot(x=PD.pred_scr_all, y=PD.score_vect_all, hue=imglabel, alpha=0.5, ax=axs[r_pad+1, 2])
    sns.scatterplot(x=PD.nlpred_scr_all, y=PD.score_vect_all, hue=imglabel, alpha=0.5, ax=axs[r_pad+1, 3])
    axs[r_pad+1, 3].set_aspect(1, adjustable='datalim')
    axs[r_pad+1, 2].set_ylabel("Observed Scores")
    axs[r_pad+1, 2].set_xlabel("Factor Linear Pred")
    axs[r_pad+1, 3].set_xlabel("Factor Pred + nl")
    axs[r_pad+1, 2].set_title("All: Before NL Fit corr %.3f"%(AllStat.cc_bef_all))
    axs[r_pad+1, 3].set_title("All: After NL Fit corr %.3f"%(AllStat.cc_aft_all))
    axs[r_pad+1, 2].legend()
    figh.suptitle(explabel, fontsize=14)
    figh.savefig(join(figdir, "%s_summary.png"%savestr))
    figh.savefig(join(figdir, "%s_summary.pdf"%savestr))
    if show:
        figh.show()
        return figh
    else:
        figh.close()
        return None


def summarize_tab(tab, verbose=False):
    validmsk = ~((tab.Animal == "Alfa") & (tab.Expi == 10))
    if "reg_cc" in tab and "exp_var" in tab:
        print("FactTsr cc: %.3f  ExpVar: %.3f" % (tab[validmsk].reg_cc.mean(), tab[validmsk].exp_var.mean()))
    if verbose:
        for sfx in ["_manif", "_evoref", "_pasu", "_all", "_allref"]:
            print("For %s: cc before fit %.3f cc after fit %.3f cc norm after fit %.3f"%(sfx[1:],
                tab[validmsk]["cc_bef"+sfx].mean(), tab[validmsk]["cc_aft"+sfx].mean(), tab[validmsk]["cc_aft_norm"+sfx].mean()))
    else:
        for sfx in ["_manif", "_evoref", "_pasu", "_all", "_allref", "_gabor"]:
            print("For %s: %.3f \t%.3f \t%.3f" % (sfx[1:], tab[validmsk]["cc_bef" + sfx].mean(),
                tab[validmsk]["cc_aft" + sfx].mean(),tab[validmsk]["cc_aft_norm" + sfx].mean()))


# Set up path for experiments
figroot = "E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis"
sumdir = join(figroot, "summary")
exproot = join(figroot, "models")
#%% population level analysis
# netname = "resnet50_linf8"
# netname = "alexnet"
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
# featnet, net = load_featnet(netname)
#%% Factorized model pipeline 
featvis_mode = "corr"
visualize_proto = False
save_data = True
showfig = False
# netname = "vgg16"; layer = "conv4_3"; bdr = 1;exp_suffix = "_nobdr"
# netname = "alexnet"; layer = "conv4"; bdr = 1;exp_suffix = "_nobdr_alex"
# netname = "resnet50_linf8";layer = "layer3";bdr = 1;exp_suffix = "_nobdr_res-robust"
netname = "resnet50_linf8";layer = "layer2";bdr = 3;exp_suffix = "_nobdr_res-robust"
# netname = "resnet50";layer = "layer2";bdr = 3;exp_suffix = "_nobdr_resnet"
# netname = "resnet50";layer = "layer3";bdr = 1;exp_suffix = "_nobdr_resnet"
for NF in [1, 2, 5, 7, 9]:
    init = "nndsvda"; solver="cd"; l1_ratio=0; alpha=0; beta_loss="frobenius" # default
    # init="nndsvd"; solver="mu"; l1_ratio=0.8; alpha=0.005; beta_loss="kullback-leibler"#"frobenius"##
    # rect_mode = "pos"; thresh = (None, None)
    rect_mode = "Tthresh"; thresh = (None, 3)

    batchsize = 41
    # Record hyper parameters in name string
    rectstr = rect_mode
    if "thresh" in rect_mode:
        if thresh[0] is not None: rectstr += "_%d"%thresh[0]
        if thresh[1] is not None: rectstr += "_%d"%thresh[1]

    if alpha > 0:
        rectstr += "_sprs%.e_l1%.e" % (alpha, l1_ratio)

    if beta_loss=="kullback-leibler":
        rectstr += "_KL"

    featvis_str = ""
    if visualize_proto and featvis_mode!="corr":
        featvis_str = "_"+featvis_mode

    expdir = join(exproot, "%s-%s_NF%d_bdr%d_%s_%s%s" % (netname, layer, NF, bdr, rectstr, exp_suffix, featvis_str))
    os.makedirs(expdir, exist_ok=True)
    featnet, net = load_featnet(netname)
    AllStat_col = []
    PredData_col = []
    for Animal in ["Alfa", "Beto"]:
        MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
        EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
        ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)[
            'ReprStats']
        for Expi in range(1, len(MStats)+1):
            imgsize = EStats[Expi - 1].evol.imgsize
            imgpos = EStats[Expi - 1].evol.imgpos
            pref_chan = EStats[Expi - 1].evol.pref_chan
            area = area_mapping(pref_chan)
            imgpix = int(imgsize * 40)
            explabel = "%s Exp%02d Driver Chan %d, %.1f deg [%s]\nCCtsr %s-%s sfx:%s bdr%d rect %s Fact %d" % (Animal, Expi, pref_chan, imgsize, tuple(imgpos), \
                netname, layer, exp_suffix, bdr, rect_mode, NF)
            print("Processing "+explabel)
            corrDict = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_Evol%s_corrTsr.npz" % (Animal, Expi, exp_suffix)),
                               allow_pickle=True)
            cctsr_dict = corrDict.get("cctsr").item()
            Ttsr_dict = corrDict.get("Ttsr").item()
            stdtsr_dict = corrDict.get("featStd").item()
            covtsr_dict = {layer: cctsr_dict[layer] * stdtsr_dict[layer] for layer in cctsr_dict}

            # show_img(ReprStats[Expi - 1].Manif.BestImg)
            figdir = join(figroot, "%s_Exp%02d" % (Animal, Expi))
            os.makedirs(figdir, exist_ok=True)
            Ttsr = Ttsr_dict[layer]
            cctsr = cctsr_dict[layer]
            covtsr = covtsr_dict[layer]
            Ttsr = np.nan_to_num(Ttsr)
            cctsr = np.nan_to_num(cctsr)
            covtsr = np.nan_to_num(covtsr)
            # Indirect factorize
            # Ttsr_pp = rectify_tsr(Ttsr, "pos")  # mode="thresh", thr=(-5, 5))  #  #
            # Hmat, Hmaps, Tcomponents, ccfactor, Stat = tsr_factorize(Ttsr_pp, covtsr, bdr=bdr, Nfactor=NF,
            #                                 figdir=figdir, savestr="%s-%scov" % (netname, layer))
            # Direct factorize
            Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr),
                     bdr=bdr, Nfactor=NF, init=init, solver=solver, l1_ratio=l1_ratio, alpha=alpha, beta_loss=beta_loss,
                     figdir=figdir, savestr="%s-%scov" % (netname, layer), suptit=explabel, show=showfig,)

            DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)
            # prediction for different image sets.
            score_vect_manif, imgfp_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="S")
            scorecol_manif  , _           = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50, 200)], stimdrive="S")
            pred_scr_manif, nlpred_scr_manif, nlfunc, PredStat_manif = predict_fit_dataset(DR_Wtsr, imgfp_manif, score_vect_manif, scorecol_manif, net, layer, \
                    netname, featnet, imgloader=loadimg_preprocess, batchsize=batchsize, figdir=figdir, savenm="manif_pred_cov", suptit=explabel+" manif", show=showfig)

            score_vect_gab, imgfp_gab = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)], stimdrive="S")
            scorecol_gab  , _         = load_score_mat(EStats, MStats, Expi, "Gabor_sgtr", wdws=[(50, 200)], stimdrive="S")
            pred_scr_gab, nlpred_scr_gab, nlfunc, PredStat_gab = predict_fit_dataset(DR_Wtsr, imgfp_gab, score_vect_gab, scorecol_gab, net, layer, \
                    netname, featnet, imgloader=loadimg_preprocess, batchsize=batchsize, figdir=figdir, savenm="pasu_pred_cov", suptit=explabel+" pasu", show=showfig)

            score_vect_pasu, imgfp_pasu = load_score_mat(EStats, MStats, Expi, "Pasu_avg", wdws=[(50, 200)], stimdrive="S")
            scorecol_pasu  , _          = load_score_mat(EStats, MStats, Expi, "Pasu_sgtr", wdws=[(50, 200)], stimdrive="S")
            pred_scr_pasu, nlpred_scr_pasu, nlfunc, PredStat_pasu = predict_fit_dataset(DR_Wtsr, imgfp_pasu, score_vect_pasu, scorecol_pasu, net, layer, \
                    netname, featnet, imgloader=loadimg_preprocess, batchsize=batchsize, figdir=figdir, savenm="gabor_pred_cov", suptit=explabel+" gabor", show=showfig)

            score_vect_evoref, imgfp_evoref = load_score_mat(EStats, MStats, Expi, "EvolRef_avg", wdws=[(50, 200)], stimdrive="S")
            scorecol_evoref  , _            = load_score_mat(EStats, MStats, Expi, "EvolRef_sgtr", wdws=[(50, 200)], stimdrive="S")
            pred_scr_evoref, nlpred_scr_evoref, nlfunc, PredStat_evoref = predict_fit_dataset(DR_Wtsr, imgfp_evoref, score_vect_evoref, scorecol_evoref, net, layer, \
                    netname, featnet, imgloader=loadimg_preprocess, batchsize=batchsize, figdir=figdir, savenm="evoref_pred_cov", suptit=explabel+" evoref", show=showfig)
            # Do nl fit on more than one image set predicted by linear model.
            [pred_scr_all, nlpred_scr_all, score_vect_all, nlfunc_all, PredStat_all] = nlfit_merged_dataset([pred_scr_manif, pred_scr_gab, pred_scr_pasu, pred_scr_evoref],
                                 [score_vect_manif, score_vect_gab, score_vect_pasu, score_vect_evoref],
                                 [scorecol_manif, scorecol_gab, scorecol_pasu, scorecol_evoref],
                                 figdir=figdir, savenm="all_pred_cov", suptit=explabel+" all", show=showfig)
            [pred_scr_allref, nlpred_scr_allref, score_vect_allref, nlfunc_allref, PredStat_allref] = nlfit_merged_dataset([pred_scr_gab, pred_scr_pasu, pred_scr_evoref],
                                 [score_vect_gab, score_vect_pasu, score_vect_evoref],
                                 [scorecol_gab, scorecol_pasu, scorecol_evoref],
                                 figdir=figdir, savenm="allref_pred_cov", suptit=explabel+" allref", show=showfig)
            [pred_scr_allnat, nlpred_scr_allnat, score_vect_allnat, nlfunc_allnat, PredStat_allnat] = nlfit_merged_dataset([pred_scr_manif, pred_scr_evoref],
                                 [score_vect_manif, score_vect_evoref],
                                 [scorecol_manif, scorecol_evoref],
                                 figdir=figdir, savenm="allnat_pred_cov", suptit=explabel+" allnat", show=showfig)
            # Meta info and
            ExpStat = EasyDict()
            for varnm in ["Animal", "Expi", "pref_chan", "area", "imgsize", "imgpos"]:
                ExpStat[varnm] = eval(varnm)

            PredData = EasyDict() # linear, nonlinear predictions and real scores for these.
            for varnm in ["pred_scr_manif", "nlpred_scr_manif", "score_vect_manif",   "pred_scr_gab", "nlpred_scr_gab", "score_vect_gab",
                "pred_scr_pasu", "nlpred_scr_pasu", "score_vect_pasu",   "pred_scr_evoref", "nlpred_scr_evoref", "score_vect_evoref",
                "pred_scr_all", "nlpred_scr_all", "score_vect_all",  "pred_scr_allref", "nlpred_scr_allref", "score_vect_allref",
                "pred_scr_allnat", "nlpred_scr_allnat", "score_vect_allnat",]:
                PredData[varnm] = eval(varnm)

            AllStat = merge_dicts([ExpStat, FactStat, add_suffix(PredStat_manif, "_manif"),
                            add_suffix(PredStat_gab, "_gabor"),
                            add_suffix(PredStat_pasu, "_pasu"),
                            add_suffix(PredStat_evoref, "_evoref"),
                            add_suffix(PredStat_all, "_all"),
                            add_suffix(PredStat_allnat, "_allnat"),
                            add_suffix(PredStat_allref, "_allref"),])
            AllStat_col.append(AllStat)
            PredData_col.append(PredData)
            # visualize prototypes
            if visualize_proto:
                factimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, layer, netname=netname,
                             score_mode=featvis_mode, featnet=featnet, bdr=bdr, Bsize=5, saveImgN=1,
                             figdir=expdir, savestr="corr", imshow=False, saveimg=False, show_featmap=False)
                tsrimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname,
                              score_mode=featvis_mode, featnet=featnet, Bsize=5, saveImgN=1, bdr=bdr, figdir=expdir, savestr="corr",
                              saveimg=False, imshow=False)
                fact_protos = [factimgs[0, :, :, :].permute([1, 2, 0]).numpy() for factimgs in factimgs_col]
                tsr_proto = tsrimgs[0, :, :, :].permute([1, 2, 0]).numpy()
                figh = visualize_factorModel(AllStat, PredData, ReprStats[Expi - 1].Manif.BestImg, Hmaps, ccfactor, explabel, \
                    savestr="%s_Exp%02d"%(Animal, Expi), figdir=expdir, fact_protos=fact_protos, tsr_proto=tsr_proto, bdr=bdr)
            else:
                fact_protos, tsr_proto = None, None
                figh = visualize_factorModel(AllStat, PredData, ReprStats[Expi - 1].Manif.BestImg, Hmaps, ccfactor, explabel, \
                    savestr="%s_Exp%02d"%(Animal, Expi), figdir=expdir, fact_protos=None, tsr_proto=None, bdr=bdr)
            if save_data:
                saveDict = EasyDict(netname=netname, layer=layer, exp_suffix=exp_suffix, bdr=bdr, explabel=explabel,
                                    rect_mode=rect_mode, thresh=thresh, featvis_mode=featvis_mode,
                                    Hmaps=Hmaps, ccfactor=ccfactor, tsr_proto=tsr_proto, fact_protos=fact_protos,
                                    BestImg=ReprStats[Expi - 1].Manif.BestImg, PredData=PredData, AllStat=AllStat,)
                pkl.dump(saveDict, open(join(expdir, "%s_Exp%02d_factors.pkl"%(Animal, Expi)), "wb"))

    tab = pd.DataFrame(AllStat_col)
    tab.to_csv(join(sumdir, "Both_pred_stats_%s-%s_%s_bdr%d_NF%d.csv"%(netname, layer, rect_mode, bdr, NF)))
    tab.to_csv(join(expdir, "Both_pred_stats_%s-%s_%s_bdr%d_NF%d.csv"%(netname, layer, rect_mode, bdr, NF)))
    pkl.dump(PredData_col, open(join(expdir, "PredictionData.pkl"), "wb"))
    with open(join(expdir, "summarytext.txt"), "w") as text_file:
        print("%s Layer %s rectify_mode %s border %d Nfact %d (cc exp suffix %s)"%(netname, layer, rectstr, bdr, NF, exp_suffix))
        print("NMF setting: init %s solver %s l1_ratio %.1e alpha %.1e beta_loss %s"%(init, solver, l1_ratio, alpha, beta_loss))
        summarize_tab(tab)

#%% Full tensor prediction
def tsr_crop_border(tsr, bdr=0):
    if bdr==0:
        return tsr
    elif bdr > 0:
        bdrtsr = np.zeros_like(tsr)
        bdrtsr[:, bdr:-bdr, bdr:-bdr] = tsr[:, bdr:-bdr, bdr:-bdr]
        return bdrtsr

featvis_mode = "corr"
visualize_proto = True
save_data = True
showfig = False
# netname = "vgg16"; layer = "conv4_3"; exp_suffix = "_nobdr"
# netname = "alexnet"; layer = "conv5"; exp_suffix = "_nobdr_alex"
# netname = "resnet50_linf8";layer = "layer3";bdr = 0;exp_suffix = "_nobdr_res-robust"
# netname = "resnet50_linf8";layer = "layer2";bdr = 3;exp_suffix = "_nobdr_res-robust"
# netname = "resnet50";layer = "layer2";bdr = 3;exp_suffix = "_nobdr_resnet"
netname = "resnet50";layer = "layer3";bdr = 0;exp_suffix = "_nobdr_resnet"
# rect_mode = "pos"; thresh = (None, None)
rect_mode = "Tthresh"; thresh = (None, 3)

# Record hyper parameters in name string
rectstr = rect_mode
if "thresh" in rect_mode:
    if thresh[0] is not None: rectstr += "_%d"%thresh[0]
    if thresh[1] is not None: rectstr += "_%d"%thresh[1]

featvis_str = ""
if visualize_proto and featvis_mode!="corr":
    featvis_str = "_"+featvis_mode

expdir = join(exproot, "%s-%s_Full_bdr%d_%s_%s%s" % (netname, layer, bdr, rectstr, exp_suffix, featvis_str))
os.makedirs(expdir, exist_ok=True)
featnet, net = load_featnet(netname)
AllStat_col = []
PredData_col = []
for Animal in ["Alfa", "Beto"]:
    MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)[
        'ReprStats']
    for Expi in range(1, len(MStats)+1):
        imgsize = EStats[Expi - 1].evol.imgsize
        imgpos = EStats[Expi - 1].evol.imgpos
        pref_chan = EStats[Expi - 1].evol.pref_chan
        area = area_mapping(pref_chan)
        imgpix = int(imgsize * 40)
        explabel = "%s Exp%02d Driver Chan %d, %.1f deg [%s]\nCCtsr %s-%s sfx:%s bdr%d rect %s" % (Animal, Expi, pref_chan, imgsize, tuple(imgpos), \
            netname, layer, exp_suffix, bdr, rect_mode)
        print("Processing "+explabel)
        corrDict = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_Evol%s_corrTsr.npz" % (Animal, Expi, exp_suffix)),
                           allow_pickle=True)
        cctsr_dict = corrDict.get("cctsr").item()
        Ttsr_dict = corrDict.get("Ttsr").item()
        stdtsr_dict = corrDict.get("featStd").item()
        covtsr_dict = {layer: cctsr_dict[layer] * stdtsr_dict[layer] for layer in cctsr_dict}

        # show_img(ReprStats[Expi - 1].Manif.BestImg)
        figdir = join(figroot, "%s_Exp%02d" % (Animal, Expi))
        os.makedirs(figdir, exist_ok=True)
        Ttsr = Ttsr_dict[layer]
        cctsr = cctsr_dict[layer]
        covtsr = covtsr_dict[layer]
        Ttsr = np.nan_to_num(Ttsr)
        cctsr = np.nan_to_num(cctsr)
        covtsr = np.nan_to_num(covtsr)
        # Just rectify
        Wtsr = rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr)
        # prediction for different image sets.
        DR_Wtsr = tsr_crop_border(Wtsr, bdr=bdr)
        score_vect_manif, imgfp_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="S")
        scorecol_manif  , _           = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50, 200)], stimdrive="S")
        pred_scr_manif, nlpred_scr_manif, nlfunc, PredStat_manif = predict_fit_dataset(DR_Wtsr, imgfp_manif, score_vect_manif, scorecol_manif, net, layer, \
                netname, featnet, imgloader=loadimg_preprocess, batchsize=62, figdir=figdir, savenm="manif_pred_cov", suptit=explabel+" manif", show=showfig)

        score_vect_gab, imgfp_gab = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)], stimdrive="S")
        scorecol_gab  , _         = load_score_mat(EStats, MStats, Expi, "Gabor_sgtr", wdws=[(50, 200)], stimdrive="S")
        pred_scr_gab, nlpred_scr_gab, nlfunc, PredStat_gab = predict_fit_dataset(DR_Wtsr, imgfp_gab, score_vect_gab, scorecol_gab, net, layer, \
                netname, featnet, imgloader=loadimg_preprocess, batchsize=62, figdir=figdir, savenm="pasu_pred_cov", suptit=explabel+" pasu", show=showfig)

        score_vect_pasu, imgfp_pasu = load_score_mat(EStats, MStats, Expi, "Pasu_avg", wdws=[(50, 200)], stimdrive="S")
        scorecol_pasu  , _          = load_score_mat(EStats, MStats, Expi, "Pasu_sgtr", wdws=[(50, 200)], stimdrive="S")
        pred_scr_pasu, nlpred_scr_pasu, nlfunc, PredStat_pasu = predict_fit_dataset(DR_Wtsr, imgfp_pasu, score_vect_pasu, scorecol_pasu, net, layer, \
                netname, featnet, imgloader=loadimg_preprocess, batchsize=62, figdir=figdir, savenm="gabor_pred_cov", suptit=explabel+" gabor", show=showfig)

        score_vect_evoref, imgfp_evoref = load_score_mat(EStats, MStats, Expi, "EvolRef_avg", wdws=[(50, 200)], stimdrive="S")
        scorecol_evoref  , _            = load_score_mat(EStats, MStats, Expi, "EvolRef_sgtr", wdws=[(50, 200)], stimdrive="S")
        pred_scr_evoref, nlpred_scr_evoref, nlfunc, PredStat_evoref = predict_fit_dataset(DR_Wtsr, imgfp_evoref, score_vect_evoref, scorecol_evoref, net, layer, \
                netname, featnet, imgloader=loadimg_preprocess, batchsize=62, figdir=figdir, savenm="evoref_pred_cov", suptit=explabel+" evoref", show=showfig)
        # Do nl fit on more than one image set predicted by linear model.
        [pred_scr_all, nlpred_scr_all, score_vect_all, nlfunc_all, PredStat_all] = nlfit_merged_dataset([pred_scr_manif, pred_scr_gab, pred_scr_pasu, pred_scr_evoref], 
                             [score_vect_manif, score_vect_gab, score_vect_pasu, score_vect_evoref],
                             [scorecol_manif, scorecol_gab, scorecol_pasu, scorecol_evoref], 
                             figdir=figdir, savenm="all_pred_cov", suptit=explabel+" all", show=showfig)
        [pred_scr_allref, nlpred_scr_allref, score_vect_allref, nlfunc_allref, PredStat_allref] = nlfit_merged_dataset([pred_scr_gab, pred_scr_pasu, pred_scr_evoref], 
                             [score_vect_gab, score_vect_pasu, score_vect_evoref],
                             [scorecol_gab, scorecol_pasu, scorecol_evoref], 
                             figdir=figdir, savenm="allref_pred_cov", suptit=explabel+" allref", show=showfig)
        [pred_scr_allnat, nlpred_scr_allnat, score_vect_allnat, nlfunc_allnat, PredStat_allnat] = nlfit_merged_dataset([pred_scr_manif, pred_scr_evoref], 
                             [score_vect_manif, score_vect_evoref],
                             [scorecol_manif, scorecol_evoref], 
                             figdir=figdir, savenm="allnat_pred_cov", suptit=explabel+" allnat", show=showfig)
        # Meta info and
        ExpStat = EasyDict()
        for varnm in ["Animal", "Expi", "pref_chan", "area", "imgsize", "imgpos"]:
            ExpStat[varnm] = eval(varnm)

        PredData = EasyDict() # linear, nonlinear predictions and real scores for these. 
        for varnm in ["pred_scr_manif", "nlpred_scr_manif", "score_vect_manif",   "pred_scr_gab", "nlpred_scr_gab", "score_vect_gab", 
            "pred_scr_pasu", "nlpred_scr_pasu", "score_vect_pasu",   "pred_scr_evoref", "nlpred_scr_evoref", "score_vect_evoref", 
            "pred_scr_all", "nlpred_scr_all", "score_vect_all",  "pred_scr_allref", "nlpred_scr_allref", "score_vect_allref",
            "pred_scr_allnat", "nlpred_scr_allnat", "score_vect_allnat",]:
            PredData[varnm] = eval(varnm)

        AllStat = merge_dicts([ExpStat, add_suffix(PredStat_manif, "_manif"),
                        add_suffix(PredStat_gab, "_gabor"),
                        add_suffix(PredStat_pasu, "_pasu"),
                        add_suffix(PredStat_evoref, "_evoref"),
                        add_suffix(PredStat_all, "_all"),
                        add_suffix(PredStat_allnat, "_allnat"),
                        add_suffix(PredStat_allref, "_allref"),])
        AllStat_col.append(AllStat)
        PredData_col.append(PredData)
        # visualize prototypes
        if visualize_proto:
            tsrimgs, mtg, score_traj = vis_feattsr(DR_Wtsr, net, G, layer, netname=netname,
                          score_mode=featvis_mode, featnet=featnet, Bsize=5, saveImgN=1, bdr=bdr, figdir=expdir, savestr="corr",
                          saveimg=False, imshow=False)
            tsr_proto = tsrimgs[0, :, :, :].permute([1, 2, 0]).numpy()  # shape [256, 256, 3] numpy array
            figh = visualize_fulltsrModel(AllStat, PredData, ReprStats[Expi - 1].Manif.BestImg, DR_Wtsr, explabel, \
                savestr="%s_Exp%02d"%(Animal, Expi), figdir=expdir, tsr_proto=tsr_proto, bdr=bdr)
        else:
            tsr_proto = None
            figh = visualize_fulltsrModel(AllStat, PredData, ReprStats[Expi - 1].Manif.BestImg, DR_Wtsr, explabel, \
                savestr="%s_Exp%02d"%(Animal, Expi), figdir=expdir, tsr_proto=None, bdr=bdr)
        if save_data:
            saveDict = EasyDict(netname=netname, layer=layer, exp_suffix=exp_suffix, bdr=bdr, explabel=explabel,
                                rect_mode=rect_mode, thresh=thresh, featvis_mode=featvis_mode,
                                tsr_proto=tsr_proto, BestImg=ReprStats[Expi - 1].Manif.BestImg, PredData=PredData, AllStat=AllStat,)
            pkl.dump(saveDict, open(join(expdir, "%s_Exp%02d_factors.pkl"%(Animal, Expi)), "wb"))

tab = pd.DataFrame(AllStat_col)
tab.to_csv(join(sumdir, "Both_pred_stats_%s-%s_%s_bdr%d_full.csv"%(netname, layer, rect_mode, bdr)))
tab.to_csv(join(expdir, "Both_pred_stats_%s-%s_%s_bdr%d_full.csv"%(netname, layer, rect_mode, bdr)))
pkl.dump(PredData_col, open(join(expdir, "PredictionData.pkl"), "wb"))

print("%s Layer %s Full tensor rectify_mode %s border %d (cc exp suffix %s)"%(netname, layer, rectstr, bdr, exp_suffix))
summarize_tab(tab)
# exptab = pd.DataFrame(ExpStat_col)
# predtab = pd.DataFrame(PredStat_col)
# facttab = pd.DataFrame(FactStat_col)
# tab = pd.concat((exptab, predtab, facttab), axis=1)
#%% 


#%% Test transform robustness to feature visualization. 
from lucent.optvis.transform import standard_transforms, pad, jitter, random_scale, random_rotate, jitter
tfms = [
    jitter(8),
    # random_scale([1 + (i - 5) / 50.0 for i in range(11)]),
    random_rotate(list(range(-10, 11)) + 5 * [0]),
    jitter(4),
]
#%%
saveDict = pkl.load(open(join(expdir,"Alfa_Exp03_factors.pkl"),'rb'))
ccfactor = saveDict.ccfactor
Hmaps = saveDict.Hmaps
netname = saveDict.netname
layer = saveDict.layer
featvis_mode = saveDict.featvis_mode
bdr = saveDict.bdr
featnet, net = load_featnet(netname)
tsrimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname, tfms=tfms,#[],
              score_mode=featvis_mode, featnet=featnet, Bsize=5, saveImgN=1, bdr=bdr, figdir=expdir, savestr="corr",
              saveimg=False, imshow=True)



# varnm = "cc_bef"; colorvar = "area"; stylevar = "Animal"
# explab1 = "vgg16-conv4_3"; explab2 = "alexnet-conv3"#"vgg16-conv3_3"
# tab1 = pd.read_csv(join(sumdir, "Both_pred_stats_vgg16-conv4_3_none_bdr1_NF3.csv"))
# tab2 = pd.read_csv(join(sumdir, 'Both_pred_stats_alexnet-conv3_none_bdr1_NF3.csv'))
# # tab2 = pd.read_csv(join(sumdir, 'Both_pred_stats_vgg16-conv3_3_none_bdr3_NF3.csv'))

#%% #################################################################
#%% Visualization development zone
#%% #################################################################
visualize_factorModel(AllStat, ReprStats[Expi - 1].Manif.BestImg, Hmaps, ccfactor)
#%% Visualization Development zone
figh, axs = plt.subplots(3, NF+1, squeeze=False, figsize=[13.5, 9])
axs[0, 0].imshow(ReprStats[Expi - 1].Manif.BestImg)
axs[0, 0].axis("off")

axs[1, 0].imshow(multichan2rgb(Hmaps))
axs[1, 0].axis("off")
for ci in range(NF):
    plt.sca(axs[0, 1+ci])
    im = axs[0, 1+ci].imshow(Hmaps[:, :, ci])
    axs[0, 1+ci].axis("off")
    figh.colorbar(im)
    axs[1, 1+ci].plot(ccfactor[:, ci], alpha=0.5)
    axs[1, 1+ci].plot([0, ccfactor.shape[0]], [0, 0], 'k-.')
    axs[1, 1+ci].plot(sorted(ccfactor[:, ci]), alpha=0.3)
    axs[1, 1+ci].set_xlim([0, ccfactor.shape[0]+1])

axs[2, 0].scatter(pred_scr_manif, nlpred_scr_manif, alpha=0.3, color='k', s=9)
axs[2, 0].scatter(pred_scr_manif, score_vect_manif, alpha=0.5, label="manif", s=25)
axs[2, 0].set_aspect(1, adjustable='datalim')
axs[2, 1].scatter(nlpred_scr_manif, score_vect_manif, alpha=0.5, s=25)
axs[2, 1].set_aspect(1, adjustable='datalim')
axs[2, 0].set_ylabel("Observed Scores")
axs[2, 0].set_xlabel("Factor Linear Pred")
axs[2, 1].set_xlabel("Factor Pred + nl")
axs[2, 0].set_title("Manif: Before NL Fit corr %.3f"%(AllStat.cc_bef_manif))
axs[2, 1].set_title("Manif: After NL Fit corr %.3f"%(AllStat.cc_aft_manif))
axs[2, 0].legend()

imglabel = AllStat.Nimg_manif*["manif"] + AllStat.Nimg_gabor*["gabor"] + \
           AllStat.Nimg_pasu*["pasu"] + AllStat.Nimg_evoref*["evoref"]
axs[2, 2].scatter(pred_scr_all, nlpred_scr_all, alpha=0.3, color='k', s=9)
sns.scatterplot(x=pred_scr_all, y=score_vect_all, hue=imglabel, alpha=0.5, ax=axs[2, 2])
# axs[2, 2].scatter(pred_scr_all, score_vect_all, alpha=0.5, label="all")
axs[2, 2].set_aspect(1, adjustable='datalim')
sns.scatterplot(x=nlpred_scr_all, y=score_vect_all, hue=imglabel, alpha=0.5, ax=axs[2, 3])
# axs[2, 3].scatter(nlpred_scr_all, score_vect_all, alpha=0.5)
axs[2, 3].set_aspect(1, adjustable='datalim')
axs[2, 2].set_ylabel("Observed Scores")
axs[2, 2].set_xlabel("Factor Linear Pred")
axs[2, 3].set_xlabel("Factor Pred + nl")
axs[2, 2].set_title("All: Before NL Fit corr %.3f"%(AllStat.cc_bef_all))
axs[2, 3].set_title("All: After NL Fit corr %.3f"%(AllStat.cc_aft_all))
axs[2, 2].legend()
figh.suptitle(explabel, fontsize=14)
figh.show()






#%% #################################################################
#%  Development Zone
#%  #################################################################
# os.listdir(sumdir)

#%%
exp_suffix = ""  # "_nobdr"
netname = "vgg16"
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
featnet, net = load_featnet(netname)
# %%
Animal = "Alfa"; Expi = 3
corrDict = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_Evol%s_corrTsr.npz" % (Animal, Expi, exp_suffix)),
                   allow_pickle=True)  #
cctsr_dict = corrDict.get("cctsr").item()
Ttsr_dict = corrDict.get("Ttsr").item()
stdtsr_dict = corrDict.get("featStd").item()
covtsr_dict = {layer: cctsr_dict[layer] * stdtsr_dict[layer] for layer in cctsr_dict}

MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)[
    'ReprStats']
score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="S")

show_img(ReprStats[Expi - 1].Manif.BestImg)
figdir = join(figroot, "%s_Exp%02d" % (Animal, Expi))
os.makedirs(figdir, exist_ok=True)

# %%
layer = "conv4_3"
Ttsr = Ttsr_dict[layer]
cctsr = cctsr_dict[layer]
covtsr = covtsr_dict[layer]
Ttsr = np.nan_to_num(Ttsr)
cctsr = np.nan_to_num(cctsr)
bdr = 1; NF = 3
Ttsr_pp = rectify_tsr(Ttsr, "abs")  # mode="thresh", thr=(-5, 5))  #  #
Hmat, Hmaps, Tcomponents, ccfactor, Stat = tsr_factorize(Ttsr_pp, covtsr, bdr=bdr, Nfactor=NF, figdir=figdir,
                                                   savestr="%s-%s" % (netname, layer))
# Hmat, Hmaps, Tcomponents, ccfactor = tsr_factorize(Ttsr_pp, cctsr, bdr=bdr, Nfactor=NF, figdir=figdir,
#                                                    savestr="%s-%s" % (netname, layer))
finimgs, mtg, score_traj = vis_feattsr(cctsr, net, G, layer, netname=netname, Bsize=5, figdir=figdir, savestr="corr",
                                       score_mode="corr")
#%%
finimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname, Bsize=5,
                  bdr=bdr, figdir=figdir, savestr="corr", MAXSTEP=100, langevin_eps=0.01, score_mode="corr")
#%
finimgs_col, mtg_col, score_traj_col = vis_featvec(ccfactor, net, G, layer, netname=netname, featnet=featnet,
                 Bsize=5, figdir=figdir, savestr="corr", imshow=False, score_mode="corr")
#%
finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, layer, netname=netname,
                 featnet=featnet, bdr=bdr, Bsize=5, figdir=figdir, savestr="corr", imshow=False, score_mode="corr")
#%%
finimgs_col, mtg_col, score_traj_col = vis_featvec_point(ccfactor, Hmaps, net, G, layer, netname=netname, pntsize=4,
                 featnet=featnet, bdr=bdr, Bsize=5, figdir=figdir, savestr="corr", imshow=False, score_mode="corr")
#%%
padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
DR_Wtsr = np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask) # torch.from_numpy()
scorer = CorrFeatScore()
scorer.register_hooks(net, layer, netname=netname)
scorer.register_weights({layer: DR_Wtsr})
with torch.no_grad():
    pred_score = score_images(featnet, scorer, layer, imgfullpath_vect, imgloader=loadimg_preprocess, batchsize=40,)
scorer.clear_hook()
nlfunc, popt, pcov, scaling, nlpred_score, Stat = fitnl_predscore(pred_score.numpy(), score_vect, savedir=figdir,
                                                            savenm="manif_pred")
#%%
# %% Covariance version of factorization
layer = "conv4_3"
Ttsr = Ttsr_dict[layer]
cctsr = cctsr_dict[layer]
covtsr = covtsr_dict[layer]
Ttsr = np.nan_to_num(Ttsr)
cctsr = np.nan_to_num(cctsr)
bdr = 1; NF = 3
Ttsr_pp = rectify_tsr(Ttsr, "abs")  # mode="thresh", thr=(-5, 5))  #  #
Hmat, Hmaps, Tcomponents, ccfactor, Stat = tsr_factorize(Ttsr_pp, covtsr, bdr=bdr, Nfactor=NF, figdir=figdir,
                                                   savestr="%s-%scov" % (netname, layer))
finimgs, mtg, score_traj = vis_feattsr(cctsr, net, G, layer, netname=netname, Bsize=5, figdir=figdir, savestr="cov")
finimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname, Bsize=5,
                                      bdr=bdr, figdir=figdir, savestr="cov", MAXSTEP=100, langevin_eps=0.01)
finimgs_col, mtg_col, score_traj_col = vis_featvec(ccfactor, net, G, layer, netname=netname, featnet=featnet,
                                   Bsize=5, figdir=figdir, savestr="cov", imshow=False)
finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, layer, netname=netname,
                                     featnet=featnet, bdr=bdr, Bsize=5, figdir=figdir, savestr="cov", imshow=False)
finimgs_col, mtg_col, score_traj_col = vis_featvec_point(ccfactor, Hmaps, net, G, layer, netname=netname,
                                     featnet=featnet, bdr=bdr, Bsize=5, figdir=figdir, savestr="cov", imshow=False)
#%%
padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
DR_Wtsr = np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask) # torch.from_numpy()
scorer = CorrFeatScore()
scorer.register_hooks(net, layer, netname=netname)
scorer.register_weights({layer: DR_Wtsr})
with torch.no_grad():
    pred_score = score_images(featnet, scorer, layer, imgfullpath_vect, imgloader=loadimg_preprocess, batchsize=40,)
scorer.clear_hook()
nlfunc, popt, pcov, scaling, nlpred_score, Stat = fitnl_predscore(pred_score.numpy(), score_vect, savedir=figdir,
                                                            savenm="manif_pred_cov")
#%% Prediction developmentr
# corr_ceil_mean, corr_ceil_std = resample_correlation(scorecol, trial=100)
# DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)
# scorer = CorrFeatScore()
# scorer.register_hooks(net, layer, netname=netname)
# scorer.register_weights({layer: DR_Wtsr})
# with torch.no_grad():
#     pred_score = score_images(featnet, scorer, layer, imgfullpath_vect, imgloader=loadimg_preprocess,
#                               batchsize=62,)
# scorer.clear_hook()
# nlfunc, popt, pcov, scaling, nlpred_score, PredStat = fitnl_predscore(pred_score.numpy(), score_vect, savedir=figdir,
#                                                     savenm="manif_pred_cov", suptit=explabel)
# # Record stats and form population statistics
# for varnm in ["corr_ceil_mean", "corr_ceil_std"]:
#     PredStat[varnm] = eval(varnm)
# PredStat.cc_aft_norm = PredStat.cc_aft / corr_ceil_mean  # prediction normalized by noise ceiling.
# PredStat.cc_bef_norm = PredStat.cc_bef / corr_ceil_mean