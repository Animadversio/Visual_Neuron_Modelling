from featvis_lib import load_featnet, rectify_tsr, tsr_factorize, tsr_posneg_factorize, vis_feattsr, vis_featvec, \
    vis_feattsr_factor, vis_featvec_point, vis_featvec_wmaps, \
    fitnl_predscore, score_images, CorrFeatScore, preprocess, loadimg_preprocess, show_img, pad_factor_prod
import os
from os.path import join
from easydict import EasyDict
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pylab as plt
from data_loader import mat_path, loadmat, load_score_mat
from GAN_utils import upconvGAN
import pandas as pd
import seaborn as sns
figroot = "E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis"
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def resample_correlation(scorecol, trial=100):
    """ Compute noise ceiling """
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

#%% population level analysis
#"_nobdr"
netname = "vgg16"
netname = "alexnet"
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
featnet, net = load_featnet(netname)
#%%
ExpStat_col = []
PredStat_col = []
FactStat_col = []
# netname = "vgg16";layer = "conv4_3";exp_suffix = "_nobdr"
netname = "alexnet"; layer = "conv2"; exp_suffix = "_nobdr_alex"
bdr = 1; NF = 3
rect_mode = "none"
thresh = (None, None)
for Animal in ["Alfa", "Beto"]:
    MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)[
        'ReprStats']
    for Expi in range(1, len(MStats)+1):
        imgsize = EStats[Expi - 1].evol.imgsize
        imgpos = EStats[Expi - 1].evol.imgpos
        pref_chan = EStats[Expi - 1].evol.pref_chan
        imgpix = int(imgsize * 40)
        explabel = "%s Exp%02d Driver Chan %d, %.1f deg [%s]" % (Animal, Expi, pref_chan, imgsize, tuple(imgpos))
        corrDict = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_Evol%s_corrTsr.npz" % (Animal, Expi, exp_suffix)),
                           allow_pickle=True)
        cctsr_dict = corrDict.get("cctsr").item()
        Ttsr_dict = corrDict.get("Ttsr").item()
        stdtsr_dict = corrDict.get("featStd").item()
        covtsr_dict = {layer: cctsr_dict[layer] * stdtsr_dict[layer] for layer in cctsr_dict}

        show_img(ReprStats[Expi - 1].Manif.BestImg)
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
        Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rectify_tsr(covtsr, rect_mode, thresh), bdr=bdr,
                           Nfactor=NF, figdir=figdir, savestr="%s-%scov" % (netname, layer), suptit=explabel)
        # prediction
        score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="S")
        scorecol, _ = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50, 200)], stimdrive="S")
        corr_ceil_mean, corr_ceil_std = resample_correlation(scorecol, trial=100)
        DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)
        scorer = CorrFeatScore()
        scorer.register_hooks(net, layer, netname=netname)
        scorer.register_weights({layer: DR_Wtsr})
        with torch.no_grad():
            pred_score = score_images(featnet, scorer, layer, imgfullpath_vect, imgloader=loadimg_preprocess,
                                      batchsize=62,)
        scorer.clear_hook()
        nlfunc, popt, pcov, scaling, nlpred_score, PredStat = fitnl_predscore(pred_score.numpy(), score_vect, savedir=figdir,
                                                            savenm="manif_pred_cov", suptit=explabel)
        # Record stats and form population statistics
        for varnm in ["corr_ceil_mean", "corr_ceil_std"]:
            PredStat[varnm] = eval(varnm)
        PredStat.cc_aft_norm = PredStat.cc_aft / corr_ceil_mean  # prediction normalized by noise ceiling.
        PredStat.cc_bef_norm = PredStat.cc_bef / corr_ceil_mean
        ExpStat = EasyDict()
        for varnm in ["Animal", "Expi", "pref_chan", "imgsize", "imgpos"]:
            ExpStat[varnm] = eval(varnm)
        PredStat_col.append(PredStat)
        FactStat_col.append(FactStat)
        ExpStat_col.append(ExpStat)
#%%
def summarize_tab(tab):
    validmsk = ~((tab.Animal=="Alfa")&(tab.Expi==10))
    print("cc before fit %.3f cc after fit %.3f cc norm after fit %.3f"%(tab[validmsk].cc_bef.mean(),
                       tab[validmsk].cc_aft.mean(), tab[validmsk].cc_aft_norm.mean()))
    print("FactTsr cc: %.3f" % (tab[validmsk].reg_cc.mean()))

exptab = pd.DataFrame(ExpStat_col)
predtab = pd.DataFrame(PredStat_col)
facttab = pd.DataFrame(FactStat_col)
sumdir = join(figroot, "summary")
tab = pd.concat((exptab, predtab, facttab), axis=1)
tab.to_csv(join(sumdir, "Both_pred_stats_%s-%s_%s_bdr%d_NF%d.csv"%(netname, layer, rect_mode, bdr, NF)))
print("%s Layer %s rectify_mode %s border %d Nfact %d"%(netname, layer, rect_mode, bdr, NF))
summarize_tab(tab)
#%%
os.listdir(sumdir)

tab1 = pd.read_csv(join(sumdir, "Both_pred_stats_vgg16-conv3_3_none_bdr3_NF3.csv"))
tab2 = pd.read_csv(join(sumdir, 'Both_pred_stats_alexnet-conv3_none_bdr1_NF3.csv'))
plt.figure(figsize=[6,5])
plt.scatter(tab1.cc_bef, tab2.cc_bef)
plt.gca().set_aspect('equal', adjustable='datalim')
plt.ylabel("alexnet-conv3")
plt.xlabel("vgg16-conv3_3")
plt.title("Linear model prediction comparison")
plt.savefig(join(sumdir, "models_pred_cmp.png"))
plt.savefig(join(sumdir, "models_pred_cmp.pdf"))
plt.show()
#%%
from scipy.stats import ttest_rel, ttest_ind


varnm = "cc_bef"; colorvar = "area"
explab1 = "vgg16-conv4_3"; explab2 = "alexnet-conv3"#"vgg16-conv3_3"
tab1 = pd.read_csv(join(sumdir, "Both_pred_stats_vgg16-conv4_3_none_bdr1_NF3.csv"))
tab2 = pd.read_csv(join(sumdir, 'Both_pred_stats_alexnet-conv3_none_bdr1_NF3.csv'))
# tab2 = pd.read_csv(join(sumdir, 'Both_pred_stats_vgg16-conv3_3_none_bdr3_NF3.csv'))

tab1["area"] = ""
tab1["area"][tab1.pref_chan <= 32] = "IT"
tab1["area"][(tab1.pref_chan <= 48) & (tab1.pref_chan >= 33)] = "V1"
tab1["area"][tab1.pref_chan >= 49] = "V4"
sns.scatterplot(x=tab1[varnm], y=tab2[varnm], hue=tab1[colorvar], style=tab1["Animal"])
plt.ylabel(explab2);plt.xlabel(explab1)
plt.gca().set_aspect('equal', adjustable='box')  # datalim
cc = np.corrcoef(tab1[varnm], tab2[varnm])[0, 1]
tval, pval = ttest_rel(np.arctanh(tab1[varnm]), np.arctanh(tab2[varnm]))
plt.title("Linear model prediction comparison\ncc %.3f t test(Fisher z) %.2f (%.1e)"%(cc, tval, pval))
plt.savefig(join(sumdir, "models_pred_cmp_%s_%s.png"%(explab1, explab2)))
plt.savefig(join(sumdir, "models_pred_cmp_%s_%s.pdf"%(explab1, explab2)))
plt.show()
#%% #################################################################
#%  Development Zone
#%  #################################################################
# os.listdir(sumdir)
tab = pd.read_csv(join(sumdir, 'Both_pred_stats_vgg16-conv3_3_bdr3_NF3.csv'))
summarize_tab(tab)
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
#%%
