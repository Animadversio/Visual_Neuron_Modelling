from featvis_lib import load_featnet, rectify_tsr, tsr_factorize, vis_feattsr, vis_featvec, vis_feattsr_factor, vis_featvec_point, \
    vis_featvec_wmaps, fitnl_predscore, score_images, CorrFeatScore, preprocess, loadimg_preprocess, show_img
import os
from os.path import join
import numpy as np
import torch
import matplotlib
import matplotlib.pylab as plt
from data_loader import mat_path, loadmat, load_score_mat
from GAN_utils import upconvGAN
figroot = "E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis"
#%%
exp_suffix = ""#"_nobdr"
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
Ttsr = np.nan_to_num(Ttsr)
cctsr = np.nan_to_num(cctsr)
bdr = 1; NF = 3
Ttsr_pp = rectify_tsr(Ttsr, "abs")# mode="thresh", thr=(-5, 5))  #  #
Hmat, Hmaps, Tcomponents, ccfactor = tsr_factorize(Ttsr_pp, cctsr, bdr=bdr, Nfactor=NF, figdir=figdir,
                                                   savestr="%s-%s" % (netname, layer))
finimgs, mtg, score_traj = vis_feattsr(cctsr, net, G, layer, netname=netname, Bsize=5, figdir=figdir, savestr="")
finimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname, Bsize=5,
                                      bdr=bdr, figdir=figdir, savestr="", MAXSTEP=100, langevin_eps=0.01)
finimgs_col, mtg_col, score_traj_col = vis_featvec(ccfactor, net, G, layer, netname=netname, featnet=featnet,
                                   Bsize=5, figdir=figdir, savestr="", imshow=False)
finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, layer, netname=netname,
                                     featnet=featnet, bdr=bdr, Bsize=5, figdir=figdir, savestr="", imshow=False)
finimgs_col, mtg_col, score_traj_col = vis_featvec_point(ccfactor, Hmaps, net, G, layer, netname=netname,
                                     featnet=featnet, bdr=bdr, Bsize=5, figdir=figdir, savestr="", imshow=False)
#%%
padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
DR_Wtsr = np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask) # torch.from_numpy()
scorer = CorrFeatScore()
scorer.register_hooks(net, layer, netname=netname)
scorer.register_weights({layer: DR_Wtsr})
with torch.no_grad():
    pred_score = score_images(featnet, scorer, layer, imgfullpath_vect, imgloader=loadimg_preprocess, batchsize=40,)
scorer.clear_hook()
nlfunc, popt, pcov, scaling, nlpred_score = fitnl_predscore(pred_score.numpy(), score_vect, savedir=figdir,
                                                            savenm="manif_pred")