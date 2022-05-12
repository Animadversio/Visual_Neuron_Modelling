from featvis_lib import load_featnet, rectify_tsr, tsr_factorize, tsr_posneg_factorize, vis_feattsr, vis_featvec, \
    vis_feattsr_factor, vis_featvec_point, vis_featvec_wmaps, \
    CorrFeatScore, preprocess, show_img, pad_factor_prod
from CorrFeatTsr_predict_lib import loadimg_preprocess, fitnl_predscore, score_images,  predict_fit_dataset, \
    predict_dataset, nlfit_merged_dataset, visualize_fulltsrModel, visualize_factorModel
from CorrFeatTsr_utils import area_mapping, add_suffix, merge_dicts, multichan2rgb
from data_loader import mat_path, loadmat, load_score_mat

import os
from os.path import join
import pickle as pkl
from easydict import EasyDict
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pylab as plt
from GAN_utils import upconvGAN
import pandas as pd
import seaborn as sns
# def visualize_fulltsrModel(AllStat, PD, protoimg, Wtsr, explabel, savestr="", figdir="", show=True,
#                            tsr_proto=None, bdr=0):
#     """Summarize the evaluations of a full tsr linear model."""
#     ncol = 4  # max(4, NF+1)
#     if tsr_proto is not None:
#         show_proto = True
#         r_pad = 1
#     else:
#         show_proto = False
#         r_pad = 0
#     nrow = r_pad + 2
#     figh, axs = plt.subplots(nrow, ncol, squeeze=False, figsize=[ncol * 4, nrow * 3])
#     axs[0, 0].imshow(protoimg)
#     axs[0, 0].axis("off")
#
#     meanMap = np.abs(Wtsr).mean(axis=0)
#     maxMap = np.abs(Wtsr).max(axis=0)
#     for ci, (Wmap, name) in enumerate( \
#             zip([meanMap, maxMap], ["mean", "max"])):
#         plt.sca(axs[0, 1 + ci])
#         im = axs[0, 1 + ci].imshow(Wmap)
#         axs[0, 1 + ci].axis("off")
#         figh.colorbar(im)
#         axs[0, 1 + ci].set_title("W map: %s sum" % name)
#
#     if show_proto:
#         axs[1, 0].imshow(tsr_proto)
#         axs[1, 0].axis("off")
#
#     axs[r_pad + 1, 0].scatter(PD.pred_scr_manif, PD.nlpred_scr_manif, alpha=0.3, color='k', s=9)
#     axs[r_pad + 1, 0].scatter(PD.pred_scr_manif, PD.score_vect_manif, alpha=0.5, label="manif", s=25)
#     axs[r_pad + 1, 1].scatter(PD.nlpred_scr_manif, PD.score_vect_manif, alpha=0.5, s=25)
#     axs[r_pad + 1, 1].set_aspect(1, adjustable='datalim')
#     axs[r_pad + 1, 0].set_ylabel("Observed Scores")
#     axs[r_pad + 1, 0].set_xlabel("Factor Linear Pred")
#     axs[r_pad + 1, 1].set_xlabel("Factor Pred + nl")
#     axs[r_pad + 1, 0].set_title("Manif: Before NL Fit corr %.3f" % (AllStat.cc_bef_manif))
#     axs[r_pad + 1, 1].set_title("Manif: After NL Fit corr %.3f" % (AllStat.cc_aft_manif))
#     axs[r_pad + 1, 0].legend()
#
#     imglabel = AllStat.Nimg_manif * ["manif"] + AllStat.Nimg_gabor * ["gabor"] + \
#                AllStat.Nimg_pasu * ["pasu"] + AllStat.Nimg_evoref * ["evoref"]
#     axs[r_pad + 1, 2].scatter(PD.pred_scr_all, PD.nlpred_scr_all, alpha=0.3, color='k', s=9)
#     sns.scatterplot(x=PD.pred_scr_all, y=PD.score_vect_all, hue=imglabel, alpha=0.5, ax=axs[r_pad + 1, 2])
#     sns.scatterplot(x=PD.nlpred_scr_all, y=PD.score_vect_all, hue=imglabel, alpha=0.5, ax=axs[r_pad + 1, 3])
#     axs[r_pad + 1, 3].set_aspect(1, adjustable='datalim')
#     axs[r_pad + 1, 2].set_ylabel("Observed Scores")
#     axs[r_pad + 1, 2].set_xlabel("Factor Linear Pred")
#     axs[r_pad + 1, 3].set_xlabel("Factor Pred + nl")
#     axs[r_pad + 1, 2].set_title("All: Before NL Fit corr %.3f" % (AllStat.cc_bef_all))
#     axs[r_pad + 1, 3].set_title("All: After NL Fit corr %.3f" % (AllStat.cc_aft_all))
#     axs[r_pad + 1, 2].legend()
#     figh.suptitle(explabel, fontsize=14)
#     figh.savefig(join(figdir, "%s_summary.png" % savestr))
#     figh.savefig(join(figdir, "%s_summary.pdf" % savestr))
#     if show:
#         figh.show()
#         return figh
#     else:
#         figh.close()
#         return None
import sys
def summarize_tab(tab, verbose=False, file=sys.stdout):
    validmsk = ~((tab.Animal == "Alfa") & (tab.Expi == 10))
    if "reg_cc" in tab and "exp_var" in tab:
        print("FactTsr cc: %.3f  ExpVar: %.3f" % (tab[validmsk].reg_cc.mean(), tab[validmsk].exp_var.mean()),file=file)
    if verbose:
        for sfx in ["_manif", "_evoref", "_pasu", "_all", "_allref"]:
            print("For %s: cc before fit %.3f cc after fit %.3f cc norm after fit %.3f"%(sfx[1:],
                tab[validmsk]["cc_bef"+sfx].mean(), tab[validmsk]["cc_aft"+sfx].mean(), tab[validmsk]["cc_aft_norm"+sfx].mean()),file=file)
    else:
        for sfx in ["_manif", "_evoref", "_pasu", "_all", "_allref", "_gabor",]:
            print("For %s: %.3f \t%.3f \t%.3f" % (sfx[1:], tab[validmsk]["cc_bef" + sfx].mean(),
                tab[validmsk]["cc_aft" + sfx].mean(),tab[validmsk]["cc_aft_norm" + sfx].mean()),file=file)
        for sfx in ["_evol", ]:
            print("For %s: %.3f \t%.3f" % (sfx[1:], tab[validmsk]["cc_bef" + sfx].mean(),
                tab[validmsk]["cc_aft" + sfx].mean(),),file=file)


#%%
figroot = r"E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis"
sumdir = join(figroot, "summary")
exproot = join(figroot, "models")
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)

#%% Factorized model for all evol manif 
# netname = "resnet50_linf8"
# netname = "alexnet"
featvis_mode = "corr"
visualize_proto = False
save_data = True
showfig = False
# netname = "resnet50_linf8";layer = "layer3";bdr = 1;exp_suffix = "_nobdr_res-robust"
# netname = "vgg16"; layer = "conv4_3"; bdr = 1;exp_suffix = "_nobdr";batchsize = 41
# netname = "alexnet"; layer = "conv4"; bdr = 1;exp_suffix = "_nobdr_alex";batchsize = 62
# netname = "resnet50";layer = "layer3";bdr = 1;exp_suffix = "_nobdr_resnet";batchsize = 62
settings_col = [#("resnet50_linf8","layer3",1,"_nobdr_res-robust",62),#]
                ("alexnet","conv4",1,"_nobdr_alex",62),
                ("resnet50","layer3",1,"_nobdr_resnet",62),
                ("resnet50_linf8","layer2",3,"_nobdr_res-robust",62),
                #("resnet50_linf8","layer3",1,"_nobdr_res-robust",62),
                ("vgg16","conv4_3",1,"_nobdr",41),]

for setting in settings_col:
    netname, layer, bdr, exp_suffix, batchsize = setting
    init = "nndsvda"; solver="cd"; l1_ratio=0; alpha=0; beta_loss="frobenius" # default
    for NF in [1, 2, 5, 7, 9]:#
        # init="nndsvd"; solver="mu"; l1_ratio=0.8; alpha=0.005; beta_loss="kullback-leibler"#"frobenius"##
        # rect_mode = "pos"; thresh = (None, None)
        rect_mode = "Tthresh"; thresh = (None, 3)
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

        expdir = join(exproot, "%s-%s_NF%d_bdr%d_%s_%s%s_CV" % (netname, layer, NF, bdr, rectstr, exp_suffix, featvis_str))
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
                # Direct factorize
                Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr),
                         bdr=bdr, Nfactor=NF, init=init, solver=solver, l1_ratio=l1_ratio, alpha=alpha, beta_loss=beta_loss,
                         figdir=figdir, savestr="%s-%scov" % (netname, layer), suptit=explabel, show=showfig,)
                DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)

                # fit nonlinearity on the Evol dataset.
                score_vect_evol, imgfp_evol = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50,200)], stimdrive="S")
                pred_scr_evol, nlpred_scr_evol, nlfunc, PredStat_evol = predict_fit_dataset(DR_Wtsr, imgfp_evol, score_vect_evol, None, net, layer, \
                        netname, featnet, nlfunc=None, imgloader=loadimg_preprocess, batchsize=batchsize, figdir=figdir, savenm="evol_pred_cov", suptit=explabel+" evol", show=showfig)

                # prediction for different image sets.
                score_vect_manif, imgfp_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="S")
                scorecol_manif  , _           = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50, 200)], stimdrive="S")
                pred_scr_manif, nlpred_scr_manif, _, PredStat_manif = predict_dataset(DR_Wtsr, imgfp_manif, score_vect_manif, scorecol_manif, net, layer, \
                        netname, featnet, nlfunc=nlfunc, imgloader=loadimg_preprocess, batchsize=batchsize, figdir=figdir, savenm="manif_pred_cov", suptit=explabel+" manif", show=showfig)

                score_vect_gab, imgfp_gab = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)], stimdrive="S")
                scorecol_gab  , _         = load_score_mat(EStats, MStats, Expi, "Gabor_sgtr", wdws=[(50, 200)], stimdrive="S")
                pred_scr_gab, nlpred_scr_gab, _, PredStat_gab = predict_dataset(DR_Wtsr, imgfp_gab, score_vect_gab, scorecol_gab, net, layer, \
                        netname, featnet, nlfunc=nlfunc, imgloader=loadimg_preprocess, batchsize=batchsize, figdir=figdir, savenm="pasu_pred_cov", suptit=explabel+" pasu", show=showfig)

                score_vect_pasu, imgfp_pasu = load_score_mat(EStats, MStats, Expi, "Pasu_avg", wdws=[(50, 200)], stimdrive="S")
                scorecol_pasu  , _          = load_score_mat(EStats, MStats, Expi, "Pasu_sgtr", wdws=[(50, 200)], stimdrive="S")
                pred_scr_pasu, nlpred_scr_pasu, _, PredStat_pasu = predict_dataset(DR_Wtsr, imgfp_pasu, score_vect_pasu, scorecol_pasu, net, layer, \
                        netname, featnet, nlfunc=nlfunc, imgloader=loadimg_preprocess, batchsize=batchsize, figdir=figdir, savenm="gabor_pred_cov", suptit=explabel+" gabor", show=showfig)

                score_vect_evoref, imgfp_evoref = load_score_mat(EStats, MStats, Expi, "EvolRef_avg", wdws=[(50, 200)], stimdrive="S")
                scorecol_evoref  , _            = load_score_mat(EStats, MStats, Expi, "EvolRef_sgtr", wdws=[(50, 200)], stimdrive="S")
                pred_scr_evoref, nlpred_scr_evoref, _, PredStat_evoref = predict_dataset(DR_Wtsr, imgfp_evoref, score_vect_evoref, scorecol_evoref, net, layer, \
                        netname, featnet, nlfunc=nlfunc, imgloader=loadimg_preprocess, batchsize=batchsize, figdir=figdir, savenm="evoref_pred_cov", suptit=explabel+" evoref", show=showfig)
                # Do nl fit on more than one image set predicted by linear model.
                [pred_scr_all, nlpred_scr_all, score_vect_all, _, PredStat_all] = nlfit_merged_dataset([pred_scr_manif, pred_scr_gab, pred_scr_pasu, pred_scr_evoref],
                                     [score_vect_manif, score_vect_gab, score_vect_pasu, score_vect_evoref],
                                     [scorecol_manif, scorecol_gab, scorecol_pasu, scorecol_evoref],
                                     nlfunc=nlfunc, figdir=figdir, savenm="all_pred_cov", suptit=explabel+" all", show=showfig)
                [pred_scr_allref, nlpred_scr_allref, score_vect_allref, _, PredStat_allref] = nlfit_merged_dataset([pred_scr_gab, pred_scr_pasu, pred_scr_evoref],
                                     [score_vect_gab, score_vect_pasu, score_vect_evoref],
                                     [scorecol_gab, scorecol_pasu, scorecol_evoref],
                                     nlfunc=nlfunc, figdir=figdir, savenm="allref_pred_cov", suptit=explabel+" allref", show=showfig)
                [pred_scr_allnat, nlpred_scr_allnat, score_vect_allnat, _, PredStat_allnat] = nlfit_merged_dataset([pred_scr_manif, pred_scr_evoref],
                                     [score_vect_manif, score_vect_evoref],
                                     [scorecol_manif, scorecol_evoref],
                                     nlfunc=nlfunc, figdir=figdir, savenm="allnat_pred_cov", suptit=explabel+" allnat", show=showfig)
                # Meta info and
                ExpStat = EasyDict()
                for varnm in ["Animal", "Expi", "pref_chan", "area", "imgsize", "imgpos"]:
                    ExpStat[varnm] = eval(varnm)

                PredData = EasyDict() # linear, nonlinear predictions and real scores for these.
                for varnm in ["pred_scr_evol", "nlpred_scr_evol", "score_vect_evol",
                    "pred_scr_manif", "nlpred_scr_manif", "score_vect_manif",   "pred_scr_gab", "nlpred_scr_gab", "score_vect_gab",
                    "pred_scr_pasu", "nlpred_scr_pasu", "score_vect_pasu",   "pred_scr_evoref", "nlpred_scr_evoref", "score_vect_evoref",
                    "pred_scr_all", "nlpred_scr_all", "score_vect_all",  "pred_scr_allref", "nlpred_scr_allref", "score_vect_allref",
                    "pred_scr_allnat", "nlpred_scr_allnat", "score_vect_allnat",]:
                    PredData[varnm] = eval(varnm)

                AllStat = merge_dicts([ExpStat, FactStat, add_suffix(PredStat_evol, "_evol"),
                                add_suffix(PredStat_manif, "_manif"),
                                add_suffix(PredStat_gab, "_gabor"),
                                add_suffix(PredStat_pasu, "_pasu"),
                                add_suffix(PredStat_evoref, "_evoref"),
                                add_suffix(PredStat_all, "_all"),
                                add_suffix(PredStat_allnat, "_allnat"),
                                add_suffix(PredStat_allref, "_allref"),])
                AllStat_col.append(AllStat)
                PredData_col.append(PredData)
                tsr_proto, fact_protos = None, None
                figh = visualize_factorModel(AllStat, PredData, ReprStats[Expi - 1].Manif.BestImg, Hmaps, ccfactor,
                                 explabel, savestr="%s_Exp%02d" % (Animal, Expi), figdir=expdir, fact_protos=None,
                                 tsr_proto=None, bdr=bdr, show=showfig)
                if save_data:
                    saveDict = EasyDict(netname=netname, layer=layer, exp_suffix=exp_suffix, bdr=bdr, explabel=explabel,
                                        rect_mode=rect_mode, thresh=thresh, featvis_mode=featvis_mode,
                                        Hmaps=Hmaps, ccfactor=ccfactor, tsr_proto=tsr_proto, fact_protos=fact_protos,
                                        BestImg=ReprStats[Expi - 1].Manif.BestImg, PredData=PredData, AllStat=AllStat,)
                    pkl.dump(saveDict, open(join(expdir, "%s_Exp%02d_factors.pkl"%(Animal, Expi)), "wb"))
                plt.close("all")
        # Wrap up summarize, save statistics
        tab = pd.DataFrame(AllStat_col)
        tab.to_csv(join(sumdir, "Both_pred_stats_%s-%s_%s_bdr%d_NF%d_CV.csv"%(netname, layer, rect_mode, bdr, NF)))
        tab.to_csv(join(expdir, "Both_pred_stats_%s-%s_%s_bdr%d_NF%d_CV.csv"%(netname, layer, rect_mode, bdr, NF)))
        pkl.dump(PredData_col, open(join(expdir, "PredictionData.pkl"), "wb"))
        print("%s Layer %s rectify_mode %s border %d Nfact %d (cc exp suffix %s)" % (
        netname, layer, rectstr, bdr, NF, exp_suffix))
        print("NMF setting: init %s solver %s l1_ratio %.1e alpha %.1e beta_loss %s" % (
        init, solver, l1_ratio, alpha, beta_loss))
        summarize_tab(tab)
        with open(join(expdir, "summarytext.txt"), "w") as text_file:
            print("%s Layer %s rectify_mode %s border %d Nfact %d (cc exp suffix %s)"%(netname, layer, rectstr, bdr, NF,exp_suffix),
                    file=text_file)
            print("NMF setting: init %s solver %s l1_ratio %.1e alpha %.1e beta_loss %s"%(init, solver, l1_ratio, alpha, beta_loss),
                  file=text_file)
            summarize_tab(tab, file=text_file)


#%%
#%% Full tensor model predict on all exps
def tsr_crop_border(tsr, bdr=0):
    if bdr==0:
        return tsr
    elif bdr > 0:
        bdrtsr = np.zeros_like(tsr)
        bdrtsr[:, bdr:-bdr, bdr:-bdr] = tsr[:, bdr:-bdr, bdr:-bdr]
        return bdrtsr

visualize_proto = True
save_data = True
showfig = False
# netname = "resnet50_linf8";layer = "layer3";bdr = 1;exp_suffix = "_nobdr_res-robust";batchsize = 61
settings_col = [#("alexnet","conv4",1,"_nobdr_alex",62),
                #("resnet50","layer3",1,"_nobdr_resnet",62),
                ("vgg16","conv4_3",1,"_nobdr",41),
                ("resnet50_linf8","layer2",3,"_nobdr_res-robust",62),
                #("resnet50_linf8","layer3",1,"_nobdr_res-robust",62),
                ]
for setting in settings_col:
    netname, layer, bdr, exp_suffix, batchsize = setting
    featvis_mode = "corr"
    init = "nndsvda"; solver="cd"; l1_ratio=0; alpha=0; beta_loss="frobenius" # default
    # init="nndsvd"; solver="mu"; l1_ratio=0.8; alpha=0.005; beta_loss="kullback-leibler"#"frobenius"##
    # rect_mode = "pos"; thresh = (None, None)
    rect_mode = "Tthresh"; thresh = (None, 3)
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
    expdir = join(exproot, "%s-%s_Full_bdr%d_%s_%s%s_CV" % (netname, layer, bdr, rectstr, exp_suffix, featvis_str))
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
            # fit nonlinearity on the Evol dataset.
            score_vect_evol, imgfp_evol = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50,200)], stimdrive="S")
            pred_scr_evol, nlpred_scr_evol, nlfunc, PredStat_evol = predict_fit_dataset(DR_Wtsr, imgfp_evol, score_vect_evol, None, net, layer, \
                    netname, featnet, nlfunc=None, imgloader=loadimg_preprocess, batchsize=batchsize, figdir=figdir, savenm="evol_pred_cov", suptit=explabel+" evol", show=showfig)
            # 
            score_vect_manif, imgfp_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="S")
            scorecol_manif  , _           = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50, 200)], stimdrive="S")
            pred_scr_manif, nlpred_scr_manif, _, PredStat_manif = predict_dataset(DR_Wtsr, imgfp_manif, score_vect_manif, scorecol_manif, net, layer, \
                    netname, featnet, nlfunc=nlfunc, imgloader=loadimg_preprocess, batchsize=62, figdir=figdir, savenm="manif_pred_cov", suptit=explabel+" manif", show=showfig)

            score_vect_gab, imgfp_gab = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)], stimdrive="S")
            scorecol_gab  , _         = load_score_mat(EStats, MStats, Expi, "Gabor_sgtr", wdws=[(50, 200)], stimdrive="S")
            pred_scr_gab, nlpred_scr_gab, _, PredStat_gab = predict_dataset(DR_Wtsr, imgfp_gab, score_vect_gab, scorecol_gab, net, layer, \
                    netname, featnet, nlfunc=nlfunc, imgloader=loadimg_preprocess, batchsize=62, figdir=figdir, savenm="pasu_pred_cov", suptit=explabel+" pasu", show=showfig)

            score_vect_pasu, imgfp_pasu = load_score_mat(EStats, MStats, Expi, "Pasu_avg", wdws=[(50, 200)], stimdrive="S")
            scorecol_pasu  , _          = load_score_mat(EStats, MStats, Expi, "Pasu_sgtr", wdws=[(50, 200)], stimdrive="S")
            pred_scr_pasu, nlpred_scr_pasu, _, PredStat_pasu = predict_dataset(DR_Wtsr, imgfp_pasu, score_vect_pasu, scorecol_pasu, net, layer, \
                    netname, featnet, nlfunc=nlfunc, imgloader=loadimg_preprocess, batchsize=62, figdir=figdir, savenm="gabor_pred_cov", suptit=explabel+" gabor", show=showfig)

            score_vect_evoref, imgfp_evoref = load_score_mat(EStats, MStats, Expi, "EvolRef_avg", wdws=[(50, 200)], stimdrive="S")
            scorecol_evoref  , _            = load_score_mat(EStats, MStats, Expi, "EvolRef_sgtr", wdws=[(50, 200)], stimdrive="S")
            pred_scr_evoref, nlpred_scr_evoref, _, PredStat_evoref = predict_dataset(DR_Wtsr, imgfp_evoref, score_vect_evoref, scorecol_evoref, net, layer, \
                    netname, featnet, nlfunc=nlfunc, imgloader=loadimg_preprocess, batchsize=62, figdir=figdir, savenm="evoref_pred_cov", suptit=explabel+" evoref", show=showfig)
            # Do nl fit on more than one image set predicted by linear model.
            [pred_scr_all, nlpred_scr_all, score_vect_all, _, PredStat_all] = nlfit_merged_dataset([pred_scr_manif, pred_scr_gab, pred_scr_pasu, pred_scr_evoref], 
                                 [score_vect_manif, score_vect_gab, score_vect_pasu, score_vect_evoref],
                                 [scorecol_manif, scorecol_gab, scorecol_pasu, scorecol_evoref], 
                                 nlfunc=nlfunc, figdir=figdir, savenm="all_pred_cov", suptit=explabel+" all", show=showfig)
            [pred_scr_allref, nlpred_scr_allref, score_vect_allref, _, PredStat_allref] = nlfit_merged_dataset([pred_scr_gab, pred_scr_pasu, pred_scr_evoref], 
                                 [score_vect_gab, score_vect_pasu, score_vect_evoref],
                                 [scorecol_gab, scorecol_pasu, scorecol_evoref], 
                                 nlfunc=nlfunc, figdir=figdir, savenm="allref_pred_cov", suptit=explabel+" allref", show=showfig)
            [pred_scr_allnat, nlpred_scr_allnat, score_vect_allnat, _, PredStat_allnat] = nlfit_merged_dataset([pred_scr_manif, pred_scr_evoref], 
                                 [score_vect_manif, score_vect_evoref],
                                 [scorecol_manif, scorecol_evoref], 
                                 nlfunc=nlfunc, figdir=figdir, savenm="allnat_pred_cov", suptit=explabel+" allnat", show=showfig)
            # Meta info and
            ExpStat = EasyDict()
            for varnm in ["Animal", "Expi", "pref_chan", "area", "imgsize", "imgpos"]:
                ExpStat[varnm] = eval(varnm)

            PredData = EasyDict() # linear, nonlinear predictions and real scores for these. 
            for varnm in ["pred_scr_evol", "nlpred_scr_evol", "score_vect_evol", 
                "pred_scr_manif", "nlpred_scr_manif", "score_vect_manif", "pred_scr_gab", "nlpred_scr_gab", "score_vect_gab", 
                "pred_scr_pasu", "nlpred_scr_pasu", "score_vect_pasu",    "pred_scr_evoref", "nlpred_scr_evoref", "score_vect_evoref", 
                "pred_scr_all", "nlpred_scr_all", "score_vect_all",  "pred_scr_allref", "nlpred_scr_allref", "score_vect_allref",
                "pred_scr_allnat", "nlpred_scr_allnat", "score_vect_allnat",]:
                PredData[varnm] = eval(varnm)

            AllStat = merge_dicts([ExpStat, add_suffix(PredStat_evol, "_evol"), 
                            add_suffix(PredStat_manif, "_manif"),
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
                    savestr="%s_Exp%02d"%(Animal, Expi), figdir=expdir, tsr_proto=tsr_proto, bdr=bdr, show=showfig)
            else:
                tsr_proto = None
                figh = visualize_fulltsrModel(AllStat, PredData, ReprStats[Expi - 1].Manif.BestImg, DR_Wtsr, explabel, \
                    savestr="%s_Exp%02d"%(Animal, Expi), figdir=expdir, tsr_proto=None, bdr=bdr, show=showfig)
            if save_data:
                saveDict = EasyDict(netname=netname, layer=layer, exp_suffix=exp_suffix, bdr=bdr, explabel=explabel,
                                    rect_mode=rect_mode, thresh=thresh, featvis_mode=featvis_mode,
                                    tsr_proto=tsr_proto, BestImg=ReprStats[Expi - 1].Manif.BestImg, PredData=PredData, AllStat=AllStat,)
                pkl.dump(saveDict, open(join(expdir, "%s_Exp%02d_factors.pkl"%(Animal, Expi)), "wb"))
            plt.close("all")

    tab = pd.DataFrame(AllStat_col)
    tab.to_csv(join(sumdir, "Both_pred_stats_%s-%s_%s_bdr%d_full_CV.csv"%(netname, layer, rect_mode, bdr)))
    tab.to_csv(join(expdir, "Both_pred_stats_%s-%s_%s_bdr%d_full_CV.csv"%(netname, layer, rect_mode, bdr)))
    pkl.dump(PredData_col, open(join(expdir, "PredictionData.pkl"), "wb"))
    try:
        print("%s Layer %s Full tensor rectify_mode %s border %d (cc exp suffix %s)"%(netname, layer, rectstr, bdr, exp_suffix))
        summarize_tab(tab)
        with open(join(expdir, "summarytext.txt"), "w") as text_file:
            print("%s Layer %s Full tensor rectify_mode %s border %d (cc exp suffix %s)"%(netname, layer, rectstr, bdr, exp_suffix),
                    file=text_file)
            print("NMF setting: init %s solver %s l1_ratio %.1e alpha %.1e beta_loss %s"%(init, solver, l1_ratio, alpha, beta_loss),
                  file=text_file)
            summarize_tab(tab, file=text_file)
    except:
        continue