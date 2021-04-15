"""This library provides higher level wrapper over CorrFeatTsr_visualize
Specifically, it provides functions that visualize a feature vector / tensor in a given layer 
"""
#%%
# %load_ext autoreload
# %autoreload 2
#%%
# !subst N: E:\Network_Data_Sync
# !subst S: E:\Network_Data_Sync
# !subst O: "E:\OneDrive - Washington University in St. Louis"

#%%
import os
from os.path import join
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import NMF
import matplotlib 
import matplotlib.pylab as plt
from numpy.linalg import norm as npnorm

from CorrFeatTsr_visualize import CorrFeatScore, corr_GAN_visualize, corr_visualize, preprocess
from GAN_utils import upconvGAN
import torch
from torch import nn
from torchvision import models
from data_loader import mat_path, load_score_mat, loadmat
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
ckpt_dir = r"E:\Cluster_Backup\torch"

#%
def align_clim(Mcol: matplotlib.image.AxesImage):
    """Util function to align the color axes of a bunch of imshow maps."""
    cmin = np.inf
    cmax = -np.inf
    for M in Mcol:
        MIN, MAX = M.get_clim()
        cmin = min(MIN, cmin)
        cmax = max(MAX, cmax)
    for M in Mcol:
        M.set_clim((cmin, cmax))
    return cmin, cmax


def show_img(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def rectify_tsr(Ttsr: np.ndarray, mode="abs", thr=(-5, 5)):
    """ Rectify tensor to prep for NMF """
    if mode is "pos_rect":
        Ttsr_pp = np.clip(Ttsr, 0, None)
    elif mode is "abs":
        Ttsr_pp = np.abs(Ttsr)
    elif mode is "thresh":
        Ttsr_pp = Ttsr.copy()
        Ttsr_pp[(Ttsr < thr[1]) * (Ttsr > thr[0])] = 0
        Ttsr_pp = np.abs(Ttsr_pp)
    else:
        raise ValueError
    return Ttsr_pp


def tsr_factorize(Ttsr_pp: np.ndarray, cctsr: np.ndarray, bdr=2, Nfactor=3, init="nndsvda", solver="cd",
                figdir="", savestr=""):
    """ Factorize the T tensor using NMF, compute the corresponding features for cctsr """
    C, H, W = Ttsr_pp.shape
    if bdr == 0:
        Tmat = Ttsr_pp.reshape(C, H * W)
        ccmat = cctsr.reshape(C, H * W)
    else:
        Tmat = Ttsr_pp[:, bdr:-bdr, bdr:-bdr].reshape(C, (H-2*bdr)*(W-2*bdr))
        ccmat = cctsr[:, bdr:-bdr, bdr:-bdr].reshape(C, (H-2*bdr)*(W-2*bdr))
    nmfsolver = NMF(n_components=Nfactor, init=init, solver=solver)  # mu
    Hmat = nmfsolver.fit_transform(Tmat.T)
    Hmaps = Hmat.reshape([H-2*bdr, W-2*bdr, Nfactor])
    Tcompon = nmfsolver.components_
    exp_var = 1-npnorm(Tmat.T - Hmat @ Tcompon) / npnorm(Tmat)
    print("NMF explained variance %.3f"%exp_var)
    ccfactor = (ccmat @ np.linalg.pinv(Hmat).T )
    # ccfactor = (ccmat @ Hmat)
    # Calculate norm of diff factors
    fact_norms = []
    for i in range(Hmaps.shape[2]):
        rank1_mat = Hmat[:, i:i+1]@Tcompon[i:i+1, :]
        matnorm = npnorm(rank1_mat, ord="fro")
        fact_norms.append(matnorm)
        print("Factor%d norm %.2f"%(i, matnorm))

    reg_cc = np.corrcoef((ccfactor @ Hmat.T).flatten(), ccmat.flatten())[0,1]
    print("Predictability of the corr coef tensor %.3f"%reg_cc)
    # Visualize maps as 3 channel image.  
    plt.imshow(Hmaps[:,:,:3] / Hmaps[:,:,:3].max()) 
    plt.axis('off')
    plt.title("channel merged")
    plt.savefig(join(figdir, "%s_factor_merged.png" % (savestr)))
    plt.savefig(join(figdir, "%s_factor_merged.pdf" % (savestr)))
    plt.show()
    # Visualize maps and their associated channel vector
    [figh, axs] = plt.subplots(2, Nfactor, figsize=[Nfactor*2.7, 5.0])
    for ci in range(Hmaps.shape[2]):
        plt.sca(axs[0, ci])  # show the map correlation
        plt.imshow(Hmaps[:, :, ci] / Hmaps.max())
        plt.axis("off")
        plt.colorbar()
        plt.sca(axs[1, ci])  # show the channel association
        axs[1, ci].plot(ccfactor[:, ci], alpha=0.5)
        ax2 = axs[1, ci].twinx()
        ax2.plot(Tcompon.T[:, ci], color="C1", alpha=0.5)
        ax2.spines['left'].set_color('C0')
        ax2.spines['right'].set_color('C1')
    plt.suptitle("Separate Factors")
    figh.savefig(join(figdir, "%s_factors.png" % (savestr)))
    figh.savefig(join(figdir, "%s_factors.pdf" % (savestr)))
    plt.show()
    return Hmat, Hmaps, Tcompon, ccfactor


def load_featnet(netname: str):
    if netname == "alexnet":
        net = models.alexnet(True)
        net.requires_grad_(False)
        featnet = net.features.cuda().eval()
    elif netname == "vgg16":
        net = models.vgg16(pretrained=True)
        net.requires_grad_(False)
        featnet = net.features.cuda().eval()
    elif netname == "resnet50":
        net = models.resnet50(pretrained=True)
        net.requires_grad_(False)
        featnet = net.cuda().eval()
    elif netname == "resnet50_linf8":
        net = models.resnet50(pretrained=True)
        net.load_state_dict(torch.load(join(ckpt_dir, "imagenet_linf_8_pure.pt")))
        net.requires_grad_(False)
        featnet = net.cuda().eval()
    else:
        raise ValueError
    return featnet, net


def vis_featvec(ccfactor, net, G, layer, netname="alexnet", featnet=None,
        preprocess=preprocess, lr=0.05, MAXSTEP=100, use_adam=True, Bsize=4, langevin_eps=0,
        imshow=True, verbose=False, savestr="", figdir="", ):
    """Feature vector over the whole map"""
    if featnet is None: featnet = net.features
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    finimgs_col, mtg_col, score_traj_col = [], [], []
    for ci in range(ccfactor.shape[1]):
        fact_W = torch.from_numpy(ccfactor[:, ci]).reshape([1,-1,1,1])
        scorer.register_weights({layer: fact_W})
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer,
                                lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps,
                                imshow=imshow, verbose=verbose, figdir=figdir, savestr="fac%d_full_%s"%(ci, savestr))
        vis_featmap_corr(scorer, featnet, finimgs, ccfactor[:, ci], layer, maptype="cov", imgscores=score_traj[-1, :],
                        figdir=figdir, savestr="fac%d_full_%s%s"%(ci, savestr, netname))
        finimgs_col.append(finimgs)
        mtg_col.append(mtg)
        score_traj_col.append(score_traj)
    scorer.clear_hook()
    return finimgs_col, mtg_col, score_traj_col


def vis_featvec_point(ccfactor: np.ndarray, Hmaps: np.ndarray, net, G, layer, netname="alexnet", featnet=None, bdr=2,
              preprocess=preprocess, lr=0.05, MAXSTEP=100, use_adam=True, Bsize=4, langevin_eps=0,
              imshow=True, verbose=False, savestr="", figdir="", ):
    """ Feature vector at the centor of the map as spatial mask. """
    if featnet is None: featnet = net.features
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    finimgs_col, mtg_col, score_traj_col = [], [], []
    for ci in range(ccfactor.shape[1]):
        H, W, _ = Hmaps.shape
        sp_mask = np.pad(np.ones([2, 2, 1]), ((H//2-1+bdr, H-H//2-1+bdr), (W//2-1+bdr, W-W//2-1+bdr),(0,0)),
                         mode="constant", constant_values=0)
        fact_Chtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, ci:ci+1], sp_mask))
        scorer.register_weights({layer: fact_Chtsr})
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer,
                            lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps,
                        imshow=imshow, verbose=verbose, figdir=figdir, savestr="fac%d_cntpnt_%s"%(ci, savestr))
        vis_featmap_corr(scorer, featnet, finimgs, ccfactor[:, ci], layer, maptype="cov", imgscores=score_traj[-1, :],
                        figdir=figdir, savestr="fac%d_cntpnt_%s%s"%(ci, savestr, netname))
        finimgs_col.append(finimgs)
        mtg_col.append(mtg)
        score_traj_col.append(score_traj)
    scorer.clear_hook()
    return finimgs_col, mtg_col, score_traj_col


def vis_featvec_wmaps(ccfactor: np.ndarray, Hmaps: np.ndarray, net, G, layer, netname="alexnet", featnet=None, bdr=2,
             preprocess=preprocess, lr=0.1, MAXSTEP=100, use_adam=True, Bsize=4, langevin_eps=0,
             imshow=True, verbose=False, savestr="", figdir="", ):
    """ Feature vector at the centor of the map as spatial mask. """
    if featnet is None: featnet = net.features
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    finimgs_col, mtg_col, score_traj_col = [], [], []
    for ci in range(ccfactor.shape[1]):
        padded_mask = np.pad(Hmaps[:, :, ci:ci + 1], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
        fact_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, ci:ci + 1], padded_mask))
        show_img(padded_mask[:, :, 0])
        scorer.register_weights({layer: fact_Wtsr})
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer,
                                lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps,
                                imshow=imshow, verbose=verbose, figdir=figdir, savestr="fac%d_map_%s"%(ci, savestr))
        vis_featmap_corr(scorer, featnet, finimgs, ccfactor[:, ci], layer, maptype="cov", imgscores=score_traj[-1, :],
                        figdir=figdir, savestr="fac%d_map_%s%s"%(ci, savestr, netname))
        finimgs_col.append(finimgs)
        mtg_col.append(mtg)
        score_traj_col.append(score_traj)
    scorer.clear_hook()
    return finimgs_col, mtg_col, score_traj_col


def vis_feattsr(cctsr, net, G, layer, netname="alexnet", featnet=None, bdr=2,
                preprocess=preprocess, lr=0.05, MAXSTEP=150, use_adam=True, Bsize=5, langevin_eps=0.03,
                imshow=True, verbose=False, savestr="", figdir="", ):
    if featnet is None: featnet = net.features
    # padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
    # DR_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask))
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    scorer.register_weights({layer: cctsr})
    finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer,
                    lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps,
                    imshow=imshow, verbose=verbose, figdir=figdir, savestr="tsr_%s%s"%(savestr, netname))
    scorer.clear_hook()
    return finimgs, mtg, score_traj


def vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname="alexnet", featnet=None, bdr=2,
                preprocess=preprocess, lr=0.05, MAXSTEP=150, use_adam=True, Bsize=5, langevin_eps=0.03,
                imshow=True, verbose=False, savestr="", figdir="", ):
    """ Visualize the factorized feature tensor """
    if featnet is None: featnet = net.features
    padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
    DR_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask))
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    scorer.register_weights({layer: DR_Wtsr})
    finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer,
                    lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps,
                    imshow=imshow, verbose=verbose, figdir=figdir, savestr="facttsr_%s%s"%(savestr, netname))
    scorer.clear_hook()
    return finimgs, mtg, score_traj


def vis_featmap_corr(scorer: CorrFeatScore, featnet: nn.Module, finimgs: torch.tensor, targvect: np.ndarray, layer: str,
                     maptype="cov", imgscores=[], figdir="", savestr=""):
    """Given a feature vec, the feature map as projecting the feat tensor onto this vector."""
    featnet(finimgs.cuda())
    act_feattsr = scorer.feat_tsr[layer].cpu()
    target_vec = torch.from_numpy(targvect).reshape([1, -1, 1, 1]).float()

    cov_map = (act_feattsr * target_vec).mean(dim=1, keepdim=False) # torch.tensor (B, H, W)
    z_feattsr = (act_feattsr - act_feattsr.mean(dim=1, keepdim=True)) / act_feattsr.std(dim=1, keepdim=True)
    z_featvec = (target_vec - target_vec.mean(dim=1, keepdim=True)) / target_vec.std(dim=1, keepdim=True)
    corr_map = (z_feattsr * z_featvec).mean(dim=1) # torch.tensor (B, H, W)
    for maptype in ["cov", "corr"]:
        map2show = cov_map if maptype == "cov" else corr_map
        NS = map2show.shape[0]
        Mcol = []
        [figh, axs] = plt.subplots(2, NS, figsize=[NS * 2.5, 5.3])
        for ci in range(NS):
            plt.sca(axs[0, ci])  # show the map correlation
            M = plt.imshow((map2show[ci, :, :] / map2show.max()).numpy())
            plt.axis("off")
            plt.title("%.2e" % imgscores[ci])
            plt.sca(axs[1, ci])  # show the image itself
            plt.imshow(ToPILImage()(finimgs[ci, :, :, :]))
            plt.axis("off")
            Mcol.append(M)
        align_clim(Mcol)
        figh.savefig(join(figdir, "%s_%s_img_%smap.png" % (savestr, layer, maptype)))
        figh.savefig(join(figdir, "%s_%s_img_%smap.pdf" % (savestr, layer, maptype)))
        plt.show()
    return cov_map, corr_map

#%%
exp_suffix = "_nobdr_alex"
netname = "alexnet"
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
featnet, net = load_featnet(netname)
#%%
Animal = "Beto"; Expi = 11
corrDict = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_Evol%s_corrTsr.npz" % (Animal, Expi, exp_suffix)), allow_pickle=True)#
cctsr_dict = corrDict.get("cctsr").item()
Ttsr_dict = corrDict.get("Ttsr").item()
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
show_img(ReprStats[Expi-1].Manif.BestImg)
#%%
figroot = "E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis"
figdir = join(figroot, "%s_Exp%02d"%(Animal, Expi))
os.makedirs(figdir, exist_ok=True)
#%%
layer = "conv3"
Ttsr = Ttsr_dict[layer]
cctsr = cctsr_dict[layer]
bdr = 1; NF = 5
Ttsr_pp = rectify_tsr(Ttsr, "abs")  # "mode="thresh", thr=(-5,5))
Hmat, Hmaps, Tcomponents, ccfactor = tsr_factorize(Ttsr_pp, cctsr, bdr=bdr, Nfactor=NF, figdir=figdir, savestr="%s-%s"%(netname, layer))

finimgs, mtg, score_traj = vis_feattsr(cctsr, net, G, layer, netname="alexnet", Bsize=6, figdir=figdir, savestr="")
#%
finimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname="alexnet", Bsize=6, bdr=bdr, figdir=figdir, savestr="alex")
#%
finimgs_col, mtg_col, score_traj_col = vis_featvec(ccfactor, net, G, layer, netname="alexnet", featnet=featnet, Bsize=6, figdir=figdir, savestr="", imshow=False)
#%
finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, layer, netname="alexnet", featnet=featnet, bdr=bdr, Bsize=6, figdir=figdir, savestr="", imshow=False)
#%
finimgs_col, mtg_col, score_traj_col = vis_featvec_point(ccfactor, Hmaps, net, G, layer, netname="alexnet", featnet=None, bdr=bdr, Bsize=6, figdir=figdir, savestr="", imshow=False)
#%%
scorer = CorrFeatScore()
scorer.register_hooks(net, layer, netname=netname)
finimgs_col, mtg_col, score_traj_col = [], [], []
for ci in range(ccfactor.shape[1]):
    H, W, _ = Hmaps.shape
    sp_mask = np.pad(np.ones([2, 2, 1]), ((H//2-1+bdr, H-H//2-1+bdr), (W//2-1+bdr, W-W//2-1+bdr),(0,0)), mode="constant", constant_values=0)
    fact_Chtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, ci:ci+1], sp_mask))
    scorer.register_weights({layer: fact_Chtsr})
    finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, lr=0.05, MAXSTEP=100, use_adam=True, Bsize=6, langevin_eps=0,
              imshow=False, verbose=False)
    vis_featmap_corr(scorer, featnet, finimgs, ccfactor[:, ci], layer, maptype="corr", imgscores=score_traj[-1, :])
    finimgs_col.append(finimgs)
    mtg_col.append(mtg)
    score_traj_col.append(score_traj)
# scorer.clear_hook()
#%%
featnet(finimgs.cuda())
#%%

ci=4
maptype = "cov"

act_feattsr = scorer.feat_tsr[layer].cpu()
target_vec = torch.from_numpy(ccfactor[:, ci:ci+1]).reshape([1,-1,1,1]).float()
cov_map = (act_feattsr * target_vec).mean(dim=1, keepdim=False)
z_feattsr = (act_feattsr - act_feattsr.mean(dim=1, keepdim=True)) / act_feattsr.std(dim=1, keepdim=True)
z_featvec = (target_vec - target_vec.mean(dim=1, keepdim=True)) / target_vec.std(dim=1, keepdim=True)
corr_map = (z_feattsr * z_featvec).mean(dim=1)

map2show = cov_map if maptype == "cov" else corr_map
NS = map2show.shape[0]
#%%
Mcol = []
[figh, axs] = plt.subplots(2, NS, figsize=[NS*2.5, 5.3])
for ci in range(NS):
    plt.sca(axs[0, ci])  # show the map correlation
    M = plt.imshow((map2show[ci, :, :] / map2show.max()).numpy())
    plt.axis("off")
    plt.title("%.2e"%score_traj[-1,ci].item())
    plt.sca(axs[1, ci])
    plt.imshow(ToPILImage()(finimgs[ci, :,:,:]))
    plt.axis("off")
    Mcol.append(M)
align_clim(Mcol)
plt.show()
# show the resulting feature map that match the current feature descriptor
#%%


