from featvis_lib import load_featnet, rectify_tsr, tsr_factorize, tsr_posneg_factorize, vis_feattsr, vis_featvec, \
    vis_feattsr_factor, vis_featvec_point, vis_featvec_wmaps, \
    CorrFeatScore, preprocess, show_img, pad_factor_prod
from CorrFeatTsr_predict_lib import loadimg_preprocess, fitnl_predscore, score_images,  predict_fit_dataset, \
    predict_dataset, nlfit_merged_dataset
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

figroot = "E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis"
sumdir = join(figroot, "summary")
exproot = join(figroot, "models")
#%% population level analysis
# netname = "resnet50_linf8"
# netname = "alexnet"
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
featvis_mode = "corr"
visualize_proto = False
save_data = True
showfig = False
netname = "resnet50_linf8";layer = "layer3";bdr = 1;exp_suffix = "_nobdr_res-robust"

NF = 3; init = "nndsvda"; solver="cd"; l1_ratio=0; alpha=0; beta_loss="frobenius" # default
# init="nndsvd"; solver="mu"; l1_ratio=0.8; alpha=0.005; beta_loss="kullback-leibler"#"frobenius"##
# rect_mode = "pos"; thresh = (None, None)
rect_mode = "Tthresh"; thresh = (None, 3)

batchsize = 61
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
        tsr_proto, fact_protos = None, None
        if save_data:
            saveDict = EasyDict(netname=netname, layer=layer, exp_suffix=exp_suffix, bdr=bdr, explabel=explabel,
                                rect_mode=rect_mode, thresh=thresh, featvis_mode=featvis_mode,
                                Hmaps=Hmaps, ccfactor=ccfactor, tsr_proto=tsr_proto, fact_protos=fact_protos,
                                BestImg=ReprStats[Expi - 1].Manif.BestImg, PredData=PredData, AllStat=AllStat,)
            pkl.dump(saveDict, open(join(expdir, "%s_Exp%02d_factors.pkl"%(Animal, Expi)), "wb"))