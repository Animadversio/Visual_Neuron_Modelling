import os

import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import johnson_lindenstrauss_min_dim, \
            SparseRandomProjection, GaussianRandomProjection
from sklearn.linear_model import LogisticRegression, LinearRegression, \
    Ridge, Lasso, PoissonRegressor, RidgeCV, LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from featvis_lib import load_featnet
import pickle as pkl
from os.path import join
import numpy as np
import torch
from neural_regress.sklearn_torchify_lib import SRP_torch, PCA_torch, \
    LinearRegression_torch, PLS_torch, SpatialAvg_torch
saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
#%%
from GAN_utils import upconvGAN
from layer_hook_utils import featureFetcher
import torch.nn.functional as F
from torch.optim import Adam
from torch_utils import show_imgrid, save_imgrid
from featvis_lib import CorrFeatScore, tsr_posneg_factorize, rectify_tsr, pad_factor_prod

def load_covtsrs(Animal, Expi, layer, ):
    data = np.load(join(fr"S:\corrFeatTsr\{Animal}_Exp{Expi:d}_Evol_nobdr_res-robust_corrTsr.npz"), allow_pickle=True)
    cctsr_dict = data.get("cctsr").item()
    Ttsr_dict = data.get("Ttsr").item()
    stdtsr_dict = data.get("featStd").item()
    covtsr_dict = {layer: cctsr_dict[layer] * stdtsr_dict[layer] for layer in cctsr_dict}
    Ttsr = Ttsr_dict[layer]
    cctsr = cctsr_dict[layer]
    covtsr = covtsr_dict[layer]
    Ttsr = np.nan_to_num(Ttsr)
    cctsr = np.nan_to_num(cctsr)
    covtsr = np.nan_to_num(covtsr)
    return covtsr, Ttsr, cctsr


def load_NMF_factors(Animal, Expi, layer, NF=3):
    # This is from resnet50_linf8 (res-robust)
    rect_mode = "Tthresh"; thresh = (None, 3); bdr = 1
    explabel = f"{Animal}_Exp{Expi:d}_resnet50_linf8_{layer}_corrfeat_rect{rect_mode}"
    covtsr, Ttsr, cctsr = load_covtsrs(Animal, Expi, layer)
    Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr),
                                                           bdr=bdr, Nfactor=NF, do_plot=False)
    return Hmat, Hmaps, ccfactor, FactStat

#%%
G = upconvGAN()
G.eval().cuda()
G.requires_grad_(False)
featnet, net = load_featnet("resnet50_linf8")
net.eval().cuda()
RGBmean = torch.tensor([0.485, 0.456, 0.406]).cuda().reshape(1, 3, 1, 1)
RGBstd = torch.tensor([0.229, 0.224, 0.225]).cuda().reshape(1, 3, 1, 1)

#%%
# featlayer = ".layer3.Bottleneck5"
# featlayer = ".layer4.Bottleneck2"
# featlayer = ".layer2.Bottleneck3"
# Xtfm_fn = f"{featlayer}_regress_Xtransforms.pkl"
# data = pkl.load(open(join(saveroot, Xtfm_fn), "rb"))
bdr = 1
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        expdir = join(saveroot, f"{Animal}_{Expi:02d}")
        for featlayer in [".layer2.Bottleneck3", ".layer4.Bottleneck2", ]:  #  ".layer3.Bottleneck5"
            layer = featlayer.split(".")[1]
            Hmat3, Hmaps3, ccfactor3, FactStat = load_NMF_factors(Animal, Expi, layer, NF=3)
            padded_mask3 = np.pad(Hmaps3[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
            Hmat1, Hmaps1, ccfactor1, _ = load_NMF_factors(Animal, Expi, layer, NF=1)
            padded_mask1 = np.pad(Hmaps1[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
            padded_mask3 = torch.tensor(padded_mask3).cuda()
            ccfactor3 = torch.tensor(ccfactor3).cuda()
            padded_mask1 = torch.tensor(padded_mask1).cuda()
            ccfactor1 = torch.tensor(ccfactor1).cuda()
            Xtfm_dict = {"spmask3": lambda tsr: torch.einsum("BCHW,HWF->BFC", tsr, padded_mask3).reshape(tsr.shape[0], -1),
                         "featvec3": lambda tsr: torch.einsum("BCHW,CF->BFHW", tsr, ccfactor3).reshape(tsr.shape[0], -1),
                         "factor3": lambda tsr: torch.einsum("BCHW,CF,HWF->BF", tsr, ccfactor3, padded_mask3).reshape(tsr.shape[0], -1),
                         "facttsr3": lambda tsr: torch.einsum("BCHW,CF,HWF->B", tsr, ccfactor3, padded_mask3).reshape(tsr.shape[0], -1),
                         "spmask1": lambda tsr: torch.einsum("BCHW,HWF->BFC", tsr, padded_mask1).reshape(tsr.shape[0], -1),
                         "featvec1": lambda tsr: torch.einsum("BCHW,CF->BFHW", tsr, ccfactor1).reshape(tsr.shape[0], -1),
                         "factor1": lambda tsr: torch.einsum("BCHW,CF,HWF->BF", tsr, ccfactor1, padded_mask1).reshape(tsr.shape[0], -1),
                         "facttsr1": lambda tsr: torch.einsum("BCHW,CF,HWF->B", tsr, ccfactor1, padded_mask1).reshape(tsr.shape[0], -1),
                         }
            regr_data = pkl.load(open(join(expdir, f"{featlayer}_factor-regress_models.pkl"), "rb"))

            featvis_dir = join(expdir, "featvis")
            os.makedirs(featvis_dir, exist_ok=True)
            #%%
            regr_cfgs = [*regr_data.keys()]
            for regrlabel in regr_cfgs:
                Xtype, regressor = regrlabel
                # raise NotImplementedError
                print(f"Processing {Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}")
                clf = regr_data[regrlabel]

                Xtfm_th = Xtfm_dict[Xtype]
                if isinstance(clf, GridSearchCV):
                    clf_th = LinearRegression_torch(clf, device="cuda")
                elif isinstance(clf, PLSRegression):
                    clf_th = PLS_torch(clf, device="cuda")
                else:
                    raise ValueError("Unknown regressor type")
                #%% Visualize the model
                B = 5
                fetcher = featureFetcher(net, input_size=(3, 224, 224), print_module=False,)
                fetcher.record(featlayer, ingraph=True)
                zs = torch.randn(B, 4096).cuda()
                zs.requires_grad_(True)
                optimizer = Adam([zs], lr=0.1)
                for i in range(100):
                    optimizer.zero_grad()
                    imgtsrs = G.visualize(zs)
                    imgtsrs_rsz = F.interpolate(imgtsrs, size=(224, 224), mode='bilinear', align_corners=False)
                    imgtsrs_rsz = (imgtsrs_rsz - RGBmean) / RGBstd
                    featnet(imgtsrs_rsz)
                    activations = fetcher[featlayer]
                    featmat = Xtfm_th(activations) # .flatten(start_dim=1)
                    scores = clf_th(featmat)
                    loss = - scores.sum()
                    loss.backward()
                    optimizer.step()
                    print(f"{i}, {scores.sum().item():.3f}")

                # show_imgrid(imgtsrs)
                save_imgrid(imgtsrs, join(featvis_dir,
                          f"{Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}_vis.png"))

#%%
"""Summarize the results into a montage acrsoo methods"""
from build_montages import crop_from_montage, make_grid_np
saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
perexproot = join(saveroot, "summary", "per_experiment")
regress_cfgs =  [('spmask3', 'Ridge'),
             ('spmask3', 'Lasso'),
             ('featvec3', 'Ridge'),
             ('featvec3', 'Lasso'),
             ('factor3', 'Ridge'),
             ('factor3', 'Lasso'),
             ('facttsr3', 'Ridge'),
             ('facttsr3', 'Lasso'),
             ('spmask1', 'Ridge'),
             ('spmask1', 'Lasso'),
             ('featvec1', 'Ridge'),
             ('featvec1', 'Lasso'),
             ('factor1', 'Ridge'),
             ('factor1', 'Lasso'),
             ('facttsr1', 'Ridge'),
             ('facttsr1', 'Lasso')]
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        expdir = join(saveroot, f"{Animal}_{Expi:02d}")
        featvis_dir = join(expdir, "featvis")
        for featlayer in [".layer2.Bottleneck3", ".layer3.Bottleneck5", ".layer4.Bottleneck2", ]:  #
            proto_col = []
            for regrlabel in regress_cfgs:
                Xtype, regressor = regrlabel
                mtg = plt.imread(join(featvis_dir, f"{Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}_vis.png"))
                proto_first = crop_from_montage(mtg, (0, 0))
                proto_col.append(proto_first)
            method_mtg = make_grid_np(proto_col, nrow=8, rowfirst=False)
            # plt.imsave(join(featvis_dir, f"{Animal}-Exp{Expi:02d}-{featlayer}-factregr_merge_vis.png"), method_mtg, )
            plt.imsave(join(perexproot, f"{Animal}-Exp{Expi:02d}-{featlayer}-factregr_merge_vis.png"), method_mtg, )
            # plt.imshow(method_mtg)
            # plt.show()
            # raise NotImplementedError
#%%