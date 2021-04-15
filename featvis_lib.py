#%% is_lab
import numpy as np
from scipy.io import loadmat
from os.path import join
from sklearn.decomposition import NMF
import matplotlib.pylab as plt
from numpy.linalg import norm as npnorm

from CorrFeatTsr_visualize import CorrFeatScore, corr_GAN_visualize, preprocess
from GAN_utils import upconvGAN
import torch
from torchvision import models
from data_loader import mat_path, load_score_mat, loadmat
ckpt_dir = r"E:\Cluster_Backup\torch"

def show_img(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()

def rectify_tsr(Ttsr, mode="abs", thr=(-5, 5)):
    if mode is "pos_rect":
        Ttsr_pp = np.clip(Ttsr, 0, None)
    elif mode is "abs":
        Ttsr_pp = np.abs(Ttsr)
    elif mode is "thresh":
        Ttsr_pp = Ttsr.copy()
        Ttsr_pp[(Ttsr<thr[1])*(Ttsr>thr[0])] = 0
        Ttsr_pp = np.abs(Ttsr_pp)
    else:
        raise ValueError
    return Ttsr_pp


def tsr_factorize(Ttsr_pp, cctsr, bdr=2, Nfactor=3, init="nndsvda", solver="cd"):
    C, H, W = Ttsr_pp.shape
    if bdr == 0:
        Tmat = Ttsr_pp.reshape(C, H*W)
        ccmat = cctsr.reshape(C, H*W)
    else:
        Tmat = Ttsr_pp[:, bdr:-bdr, bdr:-bdr].reshape(C, (H-2*bdr)*(W-2*bdr))
        ccmat = cctsr[:, bdr:-bdr, bdr:-bdr].reshape(C, (H-2*bdr)*(W-2*bdr))
    nmfsolver = NMF(n_components=Nfactor, init=init, solver=solver)  # mu
    Hmat = nmfsolver.fit_transform(Tmat.T)
    Hmaps = Hmat.reshape([H-2*bdr, W-2*bdr, Nfactor])
    exp_var = 1-npnorm(Tmat.T-Hmat@nmfsolver.components_)/npnorm(Tmat)
    Tcompon = nmfsolver.components_
    print("NMF explained variance %.3f"%exp_var)
    ccfactor = (ccmat @ np.linalg.pinv(Hmat).T )
    # ccfactor = (ccmat @ Hmat )
    reg_cc = np.corrcoef((ccfactor@Hmat.T).flatten(), ccmat.flatten())[0,1]
    print("Predictability of the corr coef tensor %.3f"%reg_cc)
    # Visualize maps
    plt.imshow(Hmaps[:,:,:3] / Hmaps[:,:,:3].max())
    plt.title("channel merged")
    plt.show()
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
    plt.show()
    # Calculate norm of diff factors
    featvecs = nmfsolver.components_
    fact_norms = []
    for i in range(Hmaps.shape[2]):
        rank1_mat = Hmat[:, i:i+1]@featvecs[i:i+1, :]
        matnorm = npnorm(rank1_mat, ord="fro")
        fact_norms.append(matnorm)
        print("Factor%d norm %.2f"%(i, matnorm))
    return Hmat, Hmaps, Tcompon, ccfactor


def load_featnet(netname):
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


def vis_featvec(ccfactor, net, G, layer, netname="alexnet", featnet=None):
    """Feature vector over the whole map"""
    if featnet is None: featnet = net.features
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    for ci in range(ccfactor.shape[1]):
        fact_W = torch.from_numpy(ccfactor[:, ci]).reshape([1,-1,1,1])
        scorer.register_weights({layer: fact_W})
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, lr=0.025, MAXSTEP=100,
                                  imshow=True, verbose=False, adam=True)
    scorer.clear_hook()
    return finimgs, mtg, score_traj


def vis_featvec_map(ccfactor, Hmaps, net, G, layer, netname="alexnet", featnet=None, bdr=2):
    """Feature vector at the centor of the map as spatial mask."""
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    for ci in range(ccfactor.shape[1]):
        H, W, _ = Hmaps.shape
        sp_mask = np.pad(np.ones([2, 2, 1]), ((H//2-1+bdr, H-H//2-1+bdr), (W//2-1+bdr, W-W//2-1+bdr),(0,0)),
                         mode="constant", constant_values=0)
        fact_Chtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, ci:ci+1], sp_mask))
        scorer.register_weights({layer: fact_Chtsr})
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, lr=0.05, MAXSTEP=100,
                              imshow=True, verbose=False)
    scorer.clear_hook()
    return finimgs, mtg, score_traj


def vis_featvec_maps(ccfactor, Hmaps, net, G, layer, netname="alexnet", featnet=None, bdr=2):
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    for ci in range(ccfactor.shape[1]):
        fact_W = torch.from_numpy(ccfactor[:, ci]).reshape([-1,1,1])
        scorer.register_weights({layer: fact_W.unsqueeze(0)})
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, lr=0.025, MAXSTEP=100,
                                  imshow=True, verbose=False, adam=True)
    scorer.clear_hook()
    return finimgs, mtg, score_traj

def vis_featvec_map(ccfactor, Hmaps, net, G, layer, netname="alexnet", featnet=None, bdr=2):
    padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
    DR_Wtsr = torch.from_numpy(np.einsum("ij,klj->ikl", ccfactor[:, :], padded_mask))
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    scorer.register_weights({layer: DR_Wtsr}) #
    finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, featnet, preprocess, layername=layer, lr=0.05,
          MAXSTEP=150, imshow=True, verbose=False, langevin_eps=0.03, Bsize=5)
    scorer.clear_hook()
    return finimgs, mtg, score_traj
#%%
Animal = "Beto"; Expi = 11
exp_suffix = "_nobdr_alex"
netname = "alexnet"
corrDict = np.load(join("S:\corrFeatTsr","%s_Exp%d_Evol%s_corrTsr.npz"%(Animal,Expi,exp_suffix)), allow_pickle=True)#
cctsr_dict = corrDict.get("cctsr").item()
Ttsr_dict = corrDict.get("Ttsr").item()

ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
show_img(ReprStats[Expi-1].Manif.BestImg)

G = upconvGAN("fc6").cuda()
G.requires_grad_(False)