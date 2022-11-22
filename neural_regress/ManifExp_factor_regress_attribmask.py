import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join
from featvis_lib import load_featnet
from scipy.io import loadmat
from data_loader import load_score_mat
import pickle as pkl
from featvis_lib import CorrFeatScore, tsr_posneg_factorize, rectify_tsr, pad_factor_prod
from CorrFeatTsr_predict_lib import fitnl_predscore, score_images, loadimg_preprocess, predict_fit_dataset
from scipy.stats import spearmanr, pearsonr
from neural_regress.regress_lib import calc_features, \
        calc_reduce_features, sweep_regressors, evaluate_prediction, \
        Ridge, Lasso, PoissonRegressor, RidgeCV, LassoCV, LinearRegression, train_test_split

mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
# saveroot = r"E:\OneDrive - Harvard University\CNN_neural_regression"
saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
#%%
from featvis_lib import CorrFeatScore, tsr_posneg_factorize, rectify_tsr, pad_factor_prod

corrTsr_root = r"E:\Network_Data_Sync\corrFeatTsr"
def load_covtsrs(Animal, Expi, layer, ):
    data = np.load(join(corrTsr_root, fr"{Animal}_Exp{Expi:d}_Evol_nobdr_res-robust_corrTsr.npz"), allow_pickle=True)
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
    Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(
            rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr),
            bdr=bdr, Nfactor=NF, do_plot=False)
    return Hmat, Hmaps, ccfactor, FactStat

#%%
from skimage.transform import resize
"""
Recovery using the weights tensor L2 norm abs of sum
"""
from stats_utils import saveallforms
fitroot = r'E:\OneDrive - Harvard University\Manifold_NeuralRegress'
outdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress\summary\weight_attrb_mask"
outdir = r"E:\OneDrive - Harvard University\Manifold_attrb_mask"
os.makedirs(outdir, exist_ok=True)
layer_long = ".layer3.Bottleneck5"
layer = "layer3"
bdr = 1
model_tuple = ('factor3', 'Ridge')
# model_tuple = ('featvec3', 'Lasso')
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46:continue
        Hmat3, Hmaps3, ccfactor3, FactStat = load_NMF_factors(Animal, Expi, "layer3", NF=3)
        padded_mask3 = np.pad(Hmaps3[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
        expdir = join(fitroot, f"{Animal}_{Expi:02d}")
        data = pkl.load(open(join(expdir, f"{layer_long}_factor-regress_models.pkl"), "rb"))
        mdl = data[model_tuple]
        weights = mdl.best_estimator_.coef_
        if model_tuple[0] == "featvec3":
            wmasks = weights.reshape(3, 14, 14).transpose(1, 2, 0)
        elif model_tuple[0] == "factor3":
            wmasks = (padded_mask3 * weights)

        # mask_merge = np.abs(wmasks.sum(axis=2))
        mask_merge = np.sqrt((wmasks**2).sum(axis=2))
        alphamsk = mask_merge/mask_merge.max()
        alphamsk_resize = resize(mask_merge, [256, 256])
        alphamsk_resize = alphamsk_resize / alphamsk_resize.max()
        # plt.imshow(alphamsk_resize, cmap="gray")
        # plt.show()
        plt.imsave(join(outdir, "%s_Exp%02d_mask_L2.png"%(Animal, Expi)),
                        alphamsk_resize, cmap="gray")
        plt.imsave(join(outdir, "%s_Exp%02d_mask_L2_rgba.png"%(Animal, Expi)),
                        np.concatenate((np.ones([256,256,3]),alphamsk_resize[:,:,np.newaxis]), axis=2))
        np.savez(join(outdir, "%s_Exp%02d_mask_L2"%(Animal, Expi)),
                 weightmsk=alphamsk, alphamsk_resize=alphamsk_resize)
        # raise Exception
        # plt.figure(figsize=[5, 5])
        # plt.imshow(mask_merge/mask_merge.max(), cmap="gray")
        # plt.title(f"{Animal} Exp{Expi:02d}", fontsize=16)
        # plt.axis("off")
        # plt.tight_layout()
        # saveallforms([outdir], f"{Animal}_Exp{Expi:02d}_{layer}_{model_tuple[0]}_{model_tuple[1]}_sum_weight_mask")
        # plt.show()
#%%