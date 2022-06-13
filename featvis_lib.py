"""This library provides higher level api over CorrFeatTsr_visualize_lib
Specifically, it provides functions that visualize a feature vector / tensor in a given layer of CNN
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
from sys import platform
from os.path import join
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import NMF
import matplotlib as mpl
import matplotlib.pylab as plt
mpl.rcParams['pdf.fonttype'] = 42
from numpy.linalg import norm as npnorm
from easydict import EasyDict
from data_loader import mat_path, load_score_mat, loadmat
from CorrFeatTsr_visualize_lib import CorrFeatScore, corr_GAN_visualize, corr_visualize, preprocess, save_imgtsr
from GAN_utils import upconvGAN
import torch
from torch import nn
from torchvision import models
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

if platform == "linux":  # CHPC cluster
    ckpt_dir = "/scratch/binxu.wang/torch"
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        ckpt_dir = r"E:\Cluster_Backup\torch"
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
        ckpt_dir = r"E:\Cluster_Backup\torch"
    # elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2C':  # PonceLab-Desktop Victoria
    #     ckpt_dir = r"E:\Cluster_Backup\torch"
    # elif os.environ['COMPUTERNAME'] == 'DESKTOP-9LH02U9':  # Home_WorkStation Victoria
    #     ckpt_dir = r"E:\Cluster_Backup\torch"
    elif os.environ['COMPUTERNAME'] == 'LAPTOP-U8TSR4RE':
        ckpt_dir = r"D:\torch\checkpoints"
    elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2B': # Alfa rig monkey computer
        ckpt_dir = r"C:\Users\Ponce lab\Documents\Python\torch_nets"
    elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2A': # Alfa rig monkey computer
        ckpt_dir = r"C:\Users\Poncelab-ML2a\Documents\Python\torch_nets"
    else:
        ckpt_dir = r"E:\Cluster_Backup\torch"

def load_featnet(netname: str):
    """API to load common CNNs and their convolutional part. 
    Default behvavior: load onto GPU, and set parameter gradient to false. 
    """
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
        raise NotImplementedError
    return featnet, net


def align_clim(Mcol: mpl.image.AxesImage):
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


def visualize_cctsr_simple(featFetcher, layers2plot, imgcol=(), savestr="Evol", titstr="Alfa_Evol", figdir=""):
    """ Visualize correlated values in the feature tensor.
    copied from experiment_EvolFeatDecompose.py

    Example:
        ExpType = "EM_cmb"
        layers2plot = ['conv3_3', 'conv4_3', 'conv5_3']
        figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, ExpType, )
        figh.savefig(join("S:\corrFeatTsr","VGGsummary","%s_Exp%d_%s_corrTsr_vis.png"%(Animal,Expi,ExpType)))
    """
    nlayer = max(4, len(layers2plot))
    figh, axs = plt.subplots(3,nlayer,figsize=[10/3*nlayer,8])
    if imgcol is not None:
        for imgi in range(len(imgcol)):
            axs[0,imgi].imshow(imgcol[imgi])
            axs[0,imgi].set_title("Highest Score Evol Img")
            axs[0,imgi].axis("off")
    for li, layer in enumerate(layers2plot):
        chanN = featFetcher.cctsr[layer].shape[0]
        tmp=axs[1,li].matshow(np.nansum(featFetcher.cctsr[layer].abs().numpy(),axis=0) / chanN)
        plt.colorbar(tmp, ax=axs[1,li])
        axs[1,li].set_title(layer+" mean abs cc")
        tmp=axs[2,li].matshow(np.nanmax(featFetcher.cctsr[layer].abs().numpy(),axis=0))
        plt.colorbar(tmp, ax=axs[2,li])
        axs[2,li].set_title(layer+" max abs cc")
    figh.suptitle("%s Exp Corr Feat Tensor"%(titstr))
    plt.show()
    figh.savefig(join(figdir, "%s_corrTsr_vis.png" % (savestr)))
    figh.savefig(join(figdir, "%s_corrTsr_vis.pdf" % (savestr)))
    return figh


def rectify_tsr(cctsr: np.ndarray, mode="abs", thr=(-5, 5), Ttsr: np.ndarray=None):
    """ Rectify tensor to prep for NMF """
    if mode == "pos":
        cctsr_pp = np.clip(cctsr, 0, None)
    elif mode == "abs":
        cctsr_pp = np.abs(cctsr)
    elif mode == "thresh":
        thr = list(thr)
        if thr[0] is None: thr[0] = - np.inf
        if thr[1] is None: thr[1] =   np.inf
        cctsr_pp = cctsr.copy()
        cctsr_pp[(cctsr < thr[1]) * (cctsr > thr[0])] = 0
        # cctsr_pp = np.abs(cctsr_pp)
    elif mode == "Tthresh":
        thr = list(thr)
        if thr[0] is None: thr[0] = - np.inf
        if thr[1] is None: thr[1] =   np.inf
        maskTsr = (Ttsr > thr[0]) * (Ttsr < thr[1])
        print("Sparsity after T threshold %.3f"%((~maskTsr).sum() / np.prod(maskTsr.shape)))
        cctsr_pp = cctsr.copy()
        cctsr_pp[maskTsr] = 0
        # ctsr_pp = np.abs(cctsr_pp)
    elif mode == "none":
        cctsr_pp = cctsr
    else:
        raise ValueError
    return cctsr_pp


def tsr_factorize(Ttsr_pp: np.ndarray, cctsr: np.ndarray, bdr=2, Nfactor=3, init="nndsvda", solver="cd",
                figdir="", savestr="", suptit="", show=True):
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
    plt.savefig(join(figdir, "%s_factor_merged.png" % (savestr))) # Indirect factorize
    plt.savefig(join(figdir, "%s_factor_merged.pdf" % (savestr)))
    if show: plt.show()
    else: plt.close()
    # Visualize maps and their associated channel vector
    [figh, axs] = plt.subplots(2, Nfactor, figsize=[Nfactor*2.7, 5.0])
    for ci in range(Hmaps.shape[2]):
        plt.sca(axs[0, ci])  # show the map correlation
        plt.imshow(Hmaps[:, :, ci] / Hmaps.max())
        plt.axis("off")
        plt.colorbar()
        plt.sca(axs[1, ci])  # show the channel association
        axs[1, ci].plot([0, ccfactor.shape[0]], [0, 0], 'k-.', alpha=0.4)
        axs[1, ci].plot(ccfactor[:, ci], alpha=0.5) # show the indirectly computed correlation the left. 
        ax2 = axs[1, ci].twinx()
        ax2.plot(Tcompon.T[:, ci], color="C1", alpha=0.5) # show the directly computed factors for T tensor on the right. 
        ax2.spines['left'].set_color('C0')
        ax2.spines['right'].set_color('C1')
    plt.suptitle("%s Separate Factors"%suptit)
    figh.savefig(join(figdir, "%s_factors.png" % (savestr)))
    figh.savefig(join(figdir, "%s_factors.pdf" % (savestr)))
    if show: plt.show()
    else: plt.close()
    Stat = EasyDict()
    for varnm in ["reg_cc", "fact_norms", "exp_var", "C", "H", "W", "bdr", "Nfactor", "init", "solver"]:
        Stat[varnm] = eval(varnm)
    return Hmat, Hmaps, Tcompon, ccfactor, Stat


def posneg_sep(tsr: np.ndarray, axis=0):
    """Separate the positive and negative entries of a matrix and concatenate along certain axis."""
    return np.concatenate((np.clip(tsr, 0, None), -np.clip(tsr, None, 0)), axis=axis)


def tsr_posneg_factorize(cctsr: np.ndarray, bdr=2, Nfactor=3,
                init="nndsvda", solver="cd", l1_ratio=0, alpha=0, beta_loss="frobenius",
                figdir="", savestr="", suptit="", show=True, do_plot=True, do_save=True):
    """ Factorize the cc tensor using NMF directly
    If any entries of cctsr is negative, it will use `posneg_sep` to create an augmented matrix with only positive entries.
    Then use NMF on that matrix. This process simulates the one sided NNMF. 

    """
    C, H, W = cctsr.shape
    if bdr == 0:
        ccmat = cctsr.reshape(C, H * W)
    else:
        ccmat = cctsr[:, bdr:-bdr, bdr:-bdr].reshape(C, (H-2*bdr)*(W-2*bdr))
    if np.any(ccmat < 0):
        sep_flag = True
        posccmat = posneg_sep(ccmat, 0)
    else:
        sep_flag = False
        posccmat = ccmat
    nmfsolver = NMF(n_components=Nfactor, init=init, solver=solver, l1_ratio=l1_ratio, alpha=alpha, beta_loss=beta_loss)  # mu
    Hmat = nmfsolver.fit_transform(posccmat.T)
    Hmaps = Hmat.reshape([H-2*bdr, W-2*bdr, Nfactor])
    CCcompon = nmfsolver.components_  # potentially augmented CC components
    if sep_flag:  # reproduce the positive and negative factors back. 
        ccfactor = (CCcompon[:, :C] - CCcompon[:, C:]).T
    else:
        ccfactor = CCcompon.T
    exp_var = 1-npnorm(posccmat.T - Hmat @ CCcompon) / npnorm(ccmat)
    print("NMF explained variance %.3f"%exp_var)
    # ccfactor = (ccmat @ np.linalg.pinv(Hmat).T )
    # ccfactor = (ccmat @ Hmat)
    # Calculate norm of diff factors
    fact_norms = []
    for i in range(Hmaps.shape[2]):
        rank1_mat = Hmat[:, i:i+1]@CCcompon[i:i+1, :]
        matnorm = npnorm(rank1_mat, ord="fro")
        fact_norms.append(matnorm)
        print("Factor%d norm %.2f"%(i, matnorm))

    reg_cc = np.corrcoef((ccfactor @ Hmat.T).flatten(), ccmat.flatten())[0,1]
    print("Correlation to the corr coef tensor %.3f"%reg_cc)
    # Visualize maps as 3 channel image.
    if Hmaps.shape[2] < 3: # Add zero channels if < 3 channels are there. 
        Hmaps_plot = np.concatenate((Hmaps, np.zeros((*Hmaps.shape[:2], 3 - Hmaps.shape[2]))), axis=2)
    else:
        Hmaps_plot = Hmaps[:, :, :3]
    if do_plot:
        plt.imshow(Hmaps_plot / Hmaps_plot.max())
        plt.axis('off')
        plt.title("%s\nchannel merged"%suptit)
        if do_save:
            plt.savefig(join(figdir, "%s_dir_factor_merged.png" % (savestr))) # direct factorize
            plt.savefig(join(figdir, "%s_dir_factor_merged.pdf" % (savestr)))
        if show: plt.show()
        else: plt.close()
        # Visualize maps and their associated channel vector
        [figh, axs] = plt.subplots(2, Nfactor, figsize=[Nfactor*2.7, 5.0], squeeze=False)
        for ci in range(Hmaps.shape[2]):
            plt.sca(axs[0, ci])  # show the map correlation
            plt.imshow(Hmaps[:, :, ci] / Hmaps.max())
            plt.axis("off")
            plt.colorbar()
            plt.sca(axs[1, ci])  # show the channel association
            axs[1, ci].plot([0, ccfactor.shape[0]], [0, 0], 'k-.', alpha=0.4)
            axs[1, ci].plot(ccfactor[:, ci], alpha=0.5)
            axs[1, ci].plot(sorted(ccfactor[:, ci]), alpha=0.25)
        plt.suptitle("%s\nSeparate Factors"%suptit)
        if do_save:
            figh.savefig(join(figdir, "%s_dir_factors.png" % (savestr)))
            figh.savefig(join(figdir, "%s_dir_factors.pdf" % (savestr)))
        if show: plt.show()
        else: plt.close()
    Stat = EasyDict()
    for varnm in ["exp_var", "reg_cc", "fact_norms", "exp_var", "C", "H", "W", "bdr", "Nfactor", "init", "solver"]:
        Stat[varnm] = eval(varnm)
    return Hmat, Hmaps, ccfactor, Stat


def vis_featvec(ccfactor, net, G, layer, netname="alexnet", featnet=None, tfms=[],
        preprocess=preprocess, lr=0.05, MAXSTEP=100, use_adam=True, Bsize=4, saveImgN=None, langevin_eps=0,
        imshow=True, verbose=False, savestr="", figdir="", saveimg=False, show_featmap=True, score_mode="dot"):
    """Feature vector over the whole map"""
    if featnet is None: featnet = net.features
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    finimgs_col, mtg_col, score_traj_col = [], [], []
    for ci in range(ccfactor.shape[1]):
        fact_W = torch.from_numpy(ccfactor[:, ci]).reshape([1,-1,1,1])
        scorer.register_weights({layer: fact_W})
        if G is None:
            finimgs, mtg, score_traj = corr_visualize(scorer, featnet, preprocess, layername=layer, tfms=tfms, 
             lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
             imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_full_%s%s-%s"%(ci, savestr, netname, layer))
        else:
            finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, tfms=tfms, 
             lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
             imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_full_%s%s-%s"%(ci, savestr, netname, layer))
        vis_featmap_corr(scorer, featnet, finimgs, ccfactor[:, ci], layer, maptype="cov", imgscores=score_traj[-1, :],
             figdir=figdir, savestr="fac%d_full_%s%s"%(ci, savestr, netname), saveimg=saveimg, showimg=show_featmap)
        finimgs_col.append(finimgs)
        mtg_col.append(mtg)
        score_traj_col.append(score_traj)
    scorer.clear_hook()
    return finimgs_col, mtg_col, score_traj_col


def vis_featvec_point(ccfactor: np.ndarray, Hmaps: np.ndarray, net, G, layer, netname="alexnet", featnet=None, bdr=2, tfms=[],
              preprocess=preprocess, lr=0.05, MAXSTEP=100, use_adam=True, Bsize=4, saveImgN=None, langevin_eps=0, pntsize=2,
              imshow=True, verbose=False, savestr="", figdir="", saveimg=False, show_featmap=True, score_mode="dot"):
    """ Feature vector at the centor of the map as spatial mask. """
    if featnet is None: featnet = net.features
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    finimgs_col, mtg_col, score_traj_col = [], [], []
    for ci in range(ccfactor.shape[1]):
        H, W, _ = Hmaps.shape
        sp_mask = np.pad(np.ones([pntsize, pntsize, 1]), ((H//2-pntsize//2+bdr, H-H//2-pntsize+pntsize//2+bdr),
                                                          (W//2-pntsize//2+bdr, W-W//2-pntsize+pntsize//2+bdr),(0,0)),
                         mode="constant", constant_values=0)
        fact_Chtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, ci:ci+1], sp_mask))
        scorer.register_weights({layer: fact_Chtsr})
        if G is None:
            finimgs, mtg, score_traj = corr_visualize(scorer, featnet, preprocess, layername=layer, tfms=tfms, 
              lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
              imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_cntpnt_%s%s-%s"%(ci, savestr, netname, layer))
        else:
            finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, tfms=tfms, 
              lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
              imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_cntpnt_%s%s-%s"%(ci, savestr, netname, layer))
        vis_featmap_corr(scorer, featnet, finimgs, ccfactor[:, ci], layer, maptype="cov", imgscores=score_traj[-1, :],
                figdir=figdir, savestr="fac%d_cntpnt_%s%s"%(ci, savestr, netname), saveimg=saveimg, showimg=show_featmap)
        finimgs_col.append(finimgs)
        mtg_col.append(mtg)
        score_traj_col.append(score_traj)
    scorer.clear_hook()
    return finimgs_col, mtg_col, score_traj_col


def vis_featvec_wmaps(ccfactor: np.ndarray, Hmaps: np.ndarray, net, G, layer, netname="alexnet", featnet=None, bdr=2, tfms=[],
             preprocess=preprocess, lr=0.1, MAXSTEP=100, use_adam=True, Bsize=4, saveImgN=None, langevin_eps=0,
             imshow=True, verbose=False, savestr="", figdir="", saveimg=False, show_featmap=True, score_mode="dot"):
    """ Feature vector at the centor of the map as spatial mask. """
    if featnet is None: featnet = net.features
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    finimgs_col, mtg_col, score_traj_col = [], [], []
    for ci in range(ccfactor.shape[1]):
        padded_mask = np.pad(Hmaps[:, :, ci:ci + 1], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
        fact_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, ci:ci + 1], padded_mask))
        if show_featmap or imshow: show_img(padded_mask[:, :, 0])
        scorer.register_weights({layer: fact_Wtsr})
        if G is None:
            finimgs, mtg, score_traj = corr_visualize(scorer, featnet, preprocess, layername=layer, tfms=tfms, 
              lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
              imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_map_%s%s-%s"%(ci, savestr, netname, layer))
        else:
            finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, tfms=tfms, 
              lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
              imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="fac%d_map_%s%s-%s"%(ci, savestr, netname, layer))
        vis_featmap_corr(scorer, featnet, finimgs, ccfactor[:, ci], layer, maptype="cov", imgscores=score_traj[-1, :],
                figdir=figdir, savestr="fac%d_map_%s%s"%(ci, savestr, netname), saveimg=saveimg, showimg=show_featmap)
        finimgs_col.append(finimgs)
        mtg_col.append(mtg)
        score_traj_col.append(score_traj)
    scorer.clear_hook()
    return finimgs_col, mtg_col, score_traj_col


def vis_feattsr(cctsr, net, G, layer, netname="alexnet", featnet=None, bdr=2, tfms=[],
                preprocess=preprocess, lr=0.05, MAXSTEP=150, use_adam=True, Bsize=5, saveImgN=None, langevin_eps=0.03,
                imshow=True, verbose=False, savestr="", figdir="", saveimg=False, score_mode="dot"):
    if featnet is None: featnet = net.features
    # padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
    # DR_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask))
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    scorer.register_weights({layer: cctsr})
    if G is None:
        finimgs, mtg, score_traj = corr_visualize(scorer, featnet, preprocess, layername=layer, tfms=tfms, 
          lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
          imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="tsr_%s%s-%s"%(savestr, netname, layer))
    else:
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, tfms=tfms, 
          lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
          imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="tsr_%s%s-%s"%(savestr, netname, layer))
    scorer.clear_hook()
    return finimgs, mtg, score_traj


def vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname="alexnet", featnet=None, bdr=2, tfms=[],
                preprocess=preprocess, lr=0.05, MAXSTEP=150, use_adam=True, Bsize=5, saveImgN=None, langevin_eps=0.03,
                imshow=True, verbose=False, savestr="", figdir="", saveimg=False, score_mode="dot"):
    """ Visualize the factorized feature tensor """
    if featnet is None: featnet = net.features
    padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
    DR_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask))
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    scorer.register_weights({layer: DR_Wtsr})
    if G is None:
        finimgs, mtg, score_traj = corr_visualize(scorer, featnet, preprocess, layername=layer, tfms=tfms, 
          lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
          imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="facttsr_%s%s-%s"%(savestr, netname, layer))
    else:
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, tfms=tfms, 
            lr=lr, MAXSTEP=MAXSTEP, use_adam=use_adam, Bsize=Bsize, langevin_eps=langevin_eps, score_mode=score_mode, saveImgN=saveImgN,
            imshow=imshow, saveimg=saveimg, verbose=verbose, figdir=figdir, savestr="facttsr_%s%s-%s"%(savestr, netname, layer))
    scorer.clear_hook()
    return finimgs, mtg, score_traj


def pad_factor_prod(Hmaps, ccfactor, bdr=0):
    padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
    DR_Wtsr = np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask)
    return DR_Wtsr


def vis_featmap_corr(scorer: CorrFeatScore, featnet: nn.Module, finimgs: torch.tensor, targvect: np.ndarray, layer: str,
                     maptype="cov", imgscores=[], figdir="", savestr="", saveimg=True, showimg=True):
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
        if saveimg:
            figh.savefig(join(figdir, "%s_%s_img_%smap.png" % (savestr, layer, maptype)))
            figh.savefig(join(figdir, "%s_%s_img_%smap.pdf" % (savestr, layer, maptype)))
        if showimg:
            figh.show()
        else:
            plt.close(figh)
    return cov_map, corr_map

#%%
if __name__ == "__main__":
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
    figroot = r"E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis"
    figdir = join(figroot, "%s_Exp%02d"%(Animal, Expi))
    os.makedirs(figdir, exist_ok=True)
    #%%
    layer = "conv3"
    Ttsr = Ttsr_dict[layer]
    cctsr = cctsr_dict[layer]
    bdr = 1; NF = 5
    Ttsr_pp = rectify_tsr(Ttsr, "abs")  # "mode="thresh", thr=(-5,5))
    Hmat, Hmaps, Tcomponents, ccfactor = tsr_factorize(Ttsr_pp, cctsr, bdr=bdr, Nfactor=NF, figdir=figdir, savestr="%s-%s"%(netname, layer))
    
    finimgs, mtg, score_traj = vis_feattsr(cctsr, net, G, layer, netname=netname, Bsize=5, figdir=figdir, savestr="")
    finimgs, mtg, score_traj = vis_feattsr_factor(ccfactor, Hmaps, net, G, layer, netname=netname, Bsize=5,
                                                  bdr=bdr, figdir=figdir, savestr="")
    finimgs_col, mtg_col, score_traj_col = vis_featvec(ccfactor, net, G, layer, netname=netname, featnet=featnet,
                                   Bsize=5, figdir=figdir, savestr="", imshow=False)
    finimgs_col, mtg_col, score_traj_col = vis_featvec_wmaps(ccfactor, Hmaps, net, G, layer, netname=netname,
                                 featnet=featnet, bdr=bdr, Bsize=5, figdir=figdir, savestr="", imshow=False)
    finimgs_col, mtg_col, score_traj_col = vis_featvec_point(ccfactor, Hmaps, net, G, layer, netname=netname,
                                 featnet=featnet, bdr=bdr, Bsize=5, figdir=figdir, savestr="", imshow=False)


    #%% Development zone for feature map visualization
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


