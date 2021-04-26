
from CorrFeatTsr_visualize import CorrFeatScore, corr_GAN_visualize, corr_visualize
from featvis_lib import vis_featvec, vis_feattsr, load_featnet, score_images, fitnl_predscore, vis_feattsr_factor, \
    rectify_tsr, tsr_factorize, vis_featvec_point, vis_featvec_wmaps
import numpy as np
from os.path import join
from GAN_utils import upconvGAN
#%%
ccdir = r"N:\Stimuli\2021-EvolDecomp\2021-04-20-Alfa-02-decomp\meta"
corrDict = np.load(join(ccdir, "%s_corrTsr.npz" % ("Evol")), allow_pickle=True)
factor_data = np.load(join(ccdir, "factor_record.npz"))
Hmat = factor_data["Hmat"]
Hmaps = factor_data["Hmaps"]
Tcomponents = factor_data["Tcomponents"]
ccfactor = factor_data["ccfactor"]
netname = factor_data["netname"].item()
layer = factor_data["layer"].item()
bdr = factor_data["bdr"].item()
NF = factor_data["NF"].item()
rect_mode = factor_data["rect_mode"].item()
torchseed = factor_data["torchseed"].item()

cctsr = corrDict["cctsr"].item()[layer]
Ttsr = corrDict["Ttsr"].item()[layer]
stdtsr = corrDict["featStd"].item()[layer]
covtsr = cctsr * stdtsr
#%%
# scorer = CorrFeatScore()
# scorer.register_hooks(net, netname)
# scorer.register_weights
#%%
netname = "vgg16"
featnet, net = load_featnet(netname)
G = upconvGAN("fc6")
G.requires_grad_(False).cuda().eval();
#%%
figdir = r"O:\corrFeatVis_FactorPredict\tmp"
# vis_feattsr(rectify_tsr(covtsr, mode="thresh", thr=(-np.inf, 0.5)), net, G, layer=layer, netname=netname, bdr=bdr,
#             figdir=figdir, savestr="corr", score_mode="corr")
# # os.makedirs(join(ccdir, "img"), exist_ok=True)
# #%%
# vis_feattsr_factor(ccfactor, Hmaps, net, G, layer=layer, netname=netname, bdr=bdr, figdir=figdir, savestr="corr", score_mode="corr")
#%%
import matplotlib.pylab as plt
from numpy.linalg import norm as npnorm
from sklearn.decomposition import NMF
def posneg_sep(tsr, axis):
    return np.concatenate((np.clip(tsr, 0, None), -np.clip(tsr, None, 0)), axis=axis)


def tsr_posneg_factorize(cctsr: np.ndarray, bdr=2, Nfactor=3, init="nndsvda", solver="cd",
                figdir="", savestr=""):
    """ Factorize the T tensor using NMF, compute the corresponding features for cctsr """
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
    nmfsolver = NMF(n_components=Nfactor, init=init, solver=solver)  # mu
    Hmat = nmfsolver.fit_transform(posccmat.T)
    Hmaps = Hmat.reshape([H-2*bdr, W-2*bdr, Nfactor])
    CCcompon = nmfsolver.components_
    if sep_flag:
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
    print("Predictability of the corr coef tensor %.3f"%reg_cc)
    # Visualize maps as 3 channel image.
    if Hmaps.shape[2] < 3:
        Hmaps_plot = np.concatenate((Hmaps, np.zeros((*Hmaps.shape[:2], 3 - Hmaps.shape[2]))), axis=2)
    else:
        Hmaps_plot = Hmaps[:, :, :3]
    plt.imshow(Hmaps_plot / Hmaps_plot.max())
    plt.axis('off')
    plt.title("channel merged")
    plt.savefig(join(figdir, "%s_factor_merged.png" % (savestr)))
    plt.savefig(join(figdir, "%s_factor_merged.pdf" % (savestr)))
    plt.show()
    # Visualize maps and their associated channel vector
    [figh, axs] = plt.subplots(2, Nfactor, figsize=[Nfactor*2.7, 5.0], squeeze=False)
    for ci in range(Hmaps.shape[2]):
        plt.sca(axs[0, ci])  # show the map correlation
        plt.imshow(Hmaps[:, :, ci] / Hmaps.max())
        plt.axis("off")
        plt.colorbar()
        plt.sca(axs[1, ci])  # show the channel association
        axs[1, ci].plot(ccfactor[:, ci], alpha=0.5)
    plt.suptitle("Separate Factors")
    figh.savefig(join(figdir, "%s_factors.png" % (savestr)))
    figh.savefig(join(figdir, "%s_factors.pdf" % (savestr)))
    plt.show()
    return Hmat, Hmaps, ccfactor
#%%
Hmat_pn, Hmaps_pn, ccfactor_pn = tsr_posneg_factorize(rectify_tsr(covtsr, "pos"), bdr=bdr, Nfactor=4)
#%%
vis_feattsr(covtsr, net, G, layer=layer, netname=netname, bdr=bdr,
                   figdir=figdir, savestr="corr", score_mode="corr")
vis_feattsr(rectify_tsr(covtsr, "pos"), net, G, layer=layer, netname=netname, bdr=bdr,
                   figdir=figdir, savestr="corr_pos", score_mode="corr")
#%%
Hmat_pn, Hmaps_pn, ccfactor_pn = tsr_posneg_factorize(rectify_tsr(covtsr, "pos"), bdr=bdr, Nfactor=3)
vis_feattsr_factor(ccfactor_pn, Hmaps_pn, net, G, layer=layer, netname=netname, bdr=bdr,
                   figdir=figdir, savestr="corr_pos", score_mode="corr")
#%%
# vis_featvec(ccfactor_pn, net, G, layer, netname=netname, savestr="", figdir=figdir, score_mode="corr")
vis_featvec_point(ccfactor_pn, Hmaps_pn, net, G, layer, bdr=bdr, netname=netname, score_mode="corr", pntsize=3,\
                  savestr="corr", figdir=figdir, )
# vis_featvec_wmaps(ccfactor_pn, np.expm1(Hmaps_pn), net, G, layer, bdr=bdr, netname=netname, score_mode="corr",\
#                   savestr="corr_expm1", figdir=figdir, )
