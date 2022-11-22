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
    Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(
            rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr),
            bdr=bdr, Nfactor=NF, do_plot=False)
    return Hmat, Hmaps, ccfactor, FactStat

#%%
"""
Recovery using the weights tensor L2 norm abs of sum
"""
from stats_utils import saveallforms
fitroot = r'E:\OneDrive - Harvard University\Manifold_NeuralRegress'
outdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress\summary\weight_attrb_mask"

layer_long = ".layer3.Bottleneck5"
layer = "layer3"
bdr = 1
# model_tuple = ('factor3', 'Ridge')
model_tuple = ('featvec3', 'Lasso')
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

        mask_merge = np.abs(wmasks.sum(axis=2))#(wmasks**2).sum(axis=2)
        plt.figure(figsize=[5, 5])
        plt.imshow(mask_merge/mask_merge.max(), cmap="gray")
        plt.title(f"{Animal} Exp{Expi:02d}", fontsize=16)
        plt.axis("off")
        plt.tight_layout()
        saveallforms([outdir], f"{Animal}_Exp{Expi:02d}_{layer}_{model_tuple[0]}_{model_tuple[1]}_sum_weight_mask")
        plt.show()

#%%
"""
Recovery using backpropagation 
"""
sumdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress\summary"

df_all = pd.read_csv(join(sumdir, "Combined_Penalize-FactorRegression_summary.csv"), index_col=0)
df_all_reset = df_all.reset_index(inplace=False)
df_all_reset["layer_s"] = df_all_reset.layer.apply(lambda s: s.split(".")[1])
#%%
df_filter = df_all_reset[(df_all_reset.FeatRed == "factor3") \
                        & (df_all_reset.regressor == "Ridge") \
                        & (df_all_reset.layer_s == "layer3") \
                        & ~((df_all_reset.Animal == "Alfa") & (df_all_reset.Expi == 10))]

df_filter.groupby("img_space")["rho_p"].agg(["mean", "sem", "count"])
#%%
for label in data:
    print(label, len(data[label].best_estimator_.coef_))

#%%
df_filter2 = df_all_reset[True \
                        &  (df_all_reset.layer_s == "layer3") \
                        & ~((df_all_reset.Animal == "Alfa") & (df_all_reset.Expi == 10))
                        &  (df_all_reset.img_space == 'Manif')
                        &  (~df_all_reset.FeatRed.str.contains("1"))]

df_filter2.groupby(["regressor", "FeatRed"], ).\
    agg({   "D2": ["mean", "sem"],
         "rho_p": ["mean", "sem", "count"],
         })  #.sort_values(by=["regressor","FeatRed"], ascending=False)

sumdir = r"E:\OneDrive - Harvard University\Manuscript_Manifold\Response\NeuralRegress_cmp"
df_filter2.groupby(["regressor", "FeatRed"], ).\
    agg({   "D2": ["mean", "sem"],
         "rho_p": ["mean", "sem", "count"],
         }).to_csv(join(sumdir, "layer3_method_cmp_Manif_summary.csv"))
#%%
sumdir = r"E:\OneDrive - Harvard University\Manuscript_Manifold\Response\NeuralRegress_cmp"
df_filter3 = df_all_reset[True \
                        &  (df_all_reset.layer_s == "layer3") \
                        &  ((df_all_reset.Animal == "Beto") & (df_all_reset.Expi == 11))
                        &  ((df_all_reset.img_space == 'all') | (df_all_reset.img_space == 'Manif'))
                        &  (~df_all_reset.FeatRed.str.contains("1")) & (~df_all_reset.FeatRed.str.contains("3"))
                        ]
import seaborn as sns
g = sns.FacetGrid(df_filter3, row="FeatRed", col="regressor",
              size=3, ylim=(0, 1), aspect=0.5)
g.map(plt.bar, "img_space", "rho_p").add_legend()
saveallforms(sumdir, "Beto_Exp11_predict_cmp", g.fig)
plt.show()
# df_filter3.groupby(["regressor", "FeatRed"], ).\
#     agg({   "D2": ["mean", "sem"],
#          "rho_p": ["mean", "sem", "count"],