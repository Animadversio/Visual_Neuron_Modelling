""" Train regression models for Evolution data and compare it with CorrFeatModel.

Plan for this pipeline:
* Loading in the Image path and response vector. (Y Vector)
* Define & save the feature reducer:
    * ImageNet PCA
    * SRP
    * spatial averaging
* Calculate features and reducing them on the fly, resulting in a XDict.
    * Using `calc_reduce_features`
* Sweep through the list of models, and cross validate each of them:
    * Partial Least Squares
    * Ridge
    * Lasso
* Predict the response for the evolution experiments, manifold exp, gabor and pasupathy.
    * Evolution
    * Manifold
    * Gabor
    * Pasupathy
"""

import os
from os.path import join
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import matplotlib.pylab as plt
from GAN_utils import upconvGAN
from insilico_Exp_torch import TorchScorer
from ZO_HessAware_Optimizers import CholeskyCMAES
from layer_hook_utils import get_module_names, register_hook_by_module_names, \
    layername_dict, featureFetcher
from scipy.io import loadmat
from data_loader import load_score_mat
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
Pasupath = r"N:\Stimuli\2019-Manifold\pasupathy-wg-f-4-ori"
Gaborpath = r"N:\Stimuli\2019-Manifold\gabor"
#%%
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
from dataset_utils import ImagePathDataset, DataLoader
#%% Newer version pipeline
# saveroot = r"E:\OneDrive - Harvard University\CNN_neural_regression"
# os.makedirs(join(saveroot, "resnet50_linf8"), exist_ok=True)
featnet, net = load_featnet("resnet50_linf8")
#%% Import Prediction pipeline from libray
from neural_regress.regress_lib import calc_features, \
        calc_reduce_features, sweep_regressors, evaluate_prediction, \
        merge_dict_arrays, merge_arrays, evaluate_dict
from scipy.stats import pearsonr, spearmanr
saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
#%%
""" 
Before proceed, run the `feattsr_ImageNet_PCA.py` at least once
to get the PCA features and the SRP feature reducers. 
"""
# # build the feature tensor transforms, SRP, PCA, spatial averaging
# n_components_ = johnson_lindenstrauss_min_dim(n_samples=len(score_vect), eps=0.1)
# srp = SparseRandomProjection(n_components=n_components_, random_state=0)
# srp.fit(np.zeros((1, 1024 * 15 * 15)))
# #%%
# # pkl.load(open(join(saveroot, "resnet50_linf8", f"pca_{featlayer}.pkl"), "rb"))
# n_components_ = johnson_lindenstrauss_min_dim(n_samples=3000, eps=0.1) # len(score_vect)
# srp = SparseRandomProjection(n_components=n_components_, random_state=0)
# srp.fit(np.zeros((1, 2048 * 8 * 8)))
# #%%
# pkl.dump({"srp": srp, "pca": pca}, open(join(saveroot, f"{featlayer}_regress_Xtransforms.pkl"), "wb"))
#%% Load the feature reduction transforms, Define all feature transforms.
featlayer = '.layer4.Bottleneck2'
data = pkl.load(open(join(saveroot, f"{featlayer}_regress_Xtransforms.pkl"), "rb"))
srp = data["srp"]
pca = data["pca"]
Xfeat_transformer = {'pca': lambda tsr: pca.transform(tsr.reshape(tsr.shape[0], -1)),
                     "srp": lambda tsr: srp.transform(tsr.reshape(tsr.shape[0], -1)),
                     "sp_avg": lambda tsr: tsr.mean(axis=(2, 3))}
#%%
import matplotlib
matplotlib.use("Agg")
#%%
# ".layer3.Bottleneck5"
for Animal in ["Alfa", "Beto"]:
    MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    for Expi in range(1, len(EStats) + 1):
        expdir = join(saveroot, f"{Animal}_{Expi:02d}")
        os.makedirs(expdir, exist_ok=True)
        # load the image paths and the response vector
        score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)], stimdrive="N")

        Xfeat_dict = calc_reduce_features(score_vect, imgfullpath_vect, Xfeat_transformer, net, featlayer,)
        y_all = score_vect
        #%%
        ridge = Ridge(alpha=1.0)
        lasso = Lasso(alpha=1.0)
        pls = PLSRegression(n_components=25)
        # poissreg = PoissonRegressor(alpha=1.0, max_iter=500)
        # kr_rbf = KernelRidge(alpha=1.0, kernel="rbf", gamma=None, )
        regressors = [ridge, lasso, pls]
        regressor_names = ["Ridge", "Lasso", "PLS"]
        result_df, fit_models = sweep_regressors(Xfeat_dict, y_all, regressors, regressor_names,)
        #%%
        result_df["layer"] = featlayer
        result_df["Expi"] = Expi
        result_df["Animal"] = Animal
        # save results
        result_df.to_csv(join(expdir, f"{featlayer}_regress_results.csv"), index=True)
        pkl.dump(fit_models, open(join(expdir, f"{featlayer}_regress_models.pkl"), "wb"))
        #%%
        df_evol, eval_evol, y_pred_evol = evaluate_prediction(fit_models, Xfeat_dict, score_vect, label=f"{featlayer}-{'Evol'}", savedir=expdir)
        #%%
        score_manif, imgfullpath_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="N")
        Xfeat_manif = calc_reduce_features(score_manif, imgfullpath_manif, Xfeat_transformer, net, featlayer,)
        df_manif, eval_manif, y_pred_manif = evaluate_prediction(fit_models, Xfeat_manif, score_manif, label=f"{featlayer}-{'Manif'}", savedir=expdir)
        #%%
        score_pasu, imgfullpath_pasu = load_score_mat(EStats, MStats, Expi, "Pasu_avg", wdws=[(50, 200)], stimdrive="N")
        Xfeat_pasu = calc_reduce_features(score_pasu, imgfullpath_pasu, Xfeat_transformer, net, featlayer,)
        df_pasu, eval_pasu, y_pred_pasu = evaluate_prediction(fit_models, Xfeat_pasu, score_pasu, label=f"{featlayer}-{'Pasu'}", savedir=expdir)
        #%%
        score_gabor, imgfullpath_gabor = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)], stimdrive="N")
        Xfeat_gabor = calc_reduce_features(score_gabor, imgfullpath_gabor, Xfeat_transformer, net, featlayer,)
        df_gabor, eval_gabor, y_pred_gabor = evaluate_prediction(fit_models, Xfeat_gabor, score_gabor, label=f"{featlayer}-{'Gabor'}", savedir=expdir)
        #%%
        score_natref, imgfullpath_natref = load_score_mat(EStats, MStats, Expi, "EvolRef_avg", wdws=[(50, 200)], stimdrive="N")
        Xfeat_natref = calc_reduce_features(score_natref, imgfullpath_natref, Xfeat_transformer, net, featlayer,)
        df_natref, eval_natref, y_pred_natref = evaluate_prediction(fit_models, Xfeat_natref, score_natref, label=f"{featlayer}-{'EvolRef'}", savedir=expdir)
        #%%
        for (Xtype, regrname) in y_pred_manif:
            y_manif = y_pred_manif[(Xtype, regrname)]
            y_gabor = y_pred_gabor[(Xtype, regrname)] if len(score_gabor) > 0 else []
            y_pasu = y_pred_pasu[(Xtype, regrname)] if len(score_pasu) > 0 else []
            y_natref = y_pred_natref[(Xtype, regrname)] if len(score_natref) > 0 else []
            plt.figure(figsize=[6, 5.5])
            plt.gca().axline((1, 1), slope=1, ls=":", color="k")
            plt.scatter(y_manif, score_manif, label="manif", alpha=0.3)
            plt.scatter(y_gabor, score_gabor, label="gabor", alpha=0.3)
            plt.scatter(y_pasu, score_pasu, label="pasu", alpha=0.3)
            plt.scatter(y_natref, score_natref, label="natref", alpha=0.3)
            plt.title(f"{Animal}-Exp{Expi:02d} {Xtype}-{regrname}")
            plt.xlabel("Predicted Spike Rate")
            plt.ylabel("Observed Spike Rate")
            plt.legend()
            plt.savefig(join(expdir, f"{Xtype}_{regrname}_{featlayer}_pred_performance.png"))
            plt.show()
        #%%
        for (Xtype, regrname) in y_pred_evol:
            plt.figure(figsize=[6, 5.5])
            plt.gca().axline((1, 1), slope=1, ls=":", color="k")
            plt.scatter(y_pred_evol[(Xtype, regrname)], score_vect,
                        label="Evol", alpha=0.3, s=9)
            plt.title(f"{Animal}-Exp{Expi:02d} {Xtype}-{regrname}-{featlayer}")
            plt.xlabel("Predicted Spike Rate")
            plt.ylabel("Observed Spike Rate")
            plt.legend()
            plt.savefig(join(expdir, f"{Xtype}_{regrname}_{featlayer}_train_performance.png"))
            plt.show()
#%%
"""
Evaluation statistics for all stimuli or allnat stimuli, save to csv
Load the predicted scores from the saved PKL files, merge them properly
then correlate /R2 with the observed spike rate. 
Plot the prediction results with the stats on top. 
"""
for Animal in ["Alfa", "Beto"]:
    MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    for Expi in range(1, len(EStats) + 1):
        expdir = join(saveroot, f"{Animal}_{Expi:02d}")
        score_manif, imgfullpath_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="N")
        score_pasu, imgfullpath_pasu = load_score_mat(EStats, MStats, Expi, "Pasu_avg", wdws=[(50, 200)], stimdrive="N")
        score_gabor, imgfullpath_gabor = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)], stimdrive="N")
        score_natref, imgfullpath_natref = load_score_mat(EStats, MStats, Expi, "EvolRef_avg", wdws=[(50, 200)], stimdrive="N")
        for featlayer in ['.layer2.Bottleneck3',
                          '.layer3.Bottleneck5',
                          '.layer4.Bottleneck2']:
            y_pred_manif = pkl.load(open(join(expdir, f"eval_predvec_{featlayer}-{'Manif'}.pkl"), "rb"))
            y_pred_pasu = pkl.load(open(join(expdir, f"eval_predvec_{featlayer}-{'Pasu'}.pkl"), "rb"))
            y_pred_gabor = pkl.load(open(join(expdir, f"eval_predvec_{featlayer}-{'Gabor'}.pkl"), "rb"))
            y_pred_natref = pkl.load(open(join(expdir, f"eval_predvec_{featlayer}-{'EvolRef'}.pkl"), "rb"))
            y_merge_dict = merge_dict_arrays(y_pred_pasu, y_pred_gabor, y_pred_natref)
            score_merge = merge_arrays(score_pasu, score_gabor, score_natref)
            df_natmerge, eval_natmerge, _ = evaluate_dict(y_merge_dict, score_merge, label=f"{featlayer}-allnat", savedir=expdir)

            y_merge2_dict = merge_dict_arrays(y_pred_manif, y_pred_pasu, y_pred_gabor, y_pred_natref)
            score_merge2 = merge_arrays(score_manif, score_pasu, score_gabor, score_natref)
            df_merge, eval_merge, _ = evaluate_dict(y_merge2_dict, score_merge2, label=f"{featlayer}-all", savedir=expdir)
            # newer version of plotting function, with the stats on title.
            for (Xtype, regrname) in y_merge_dict.keys():
                SN = df_natmerge.loc[Xtype, regrname]
                SA = df_merge.loc[Xtype, regrname]
                statstr_nat = f"no Manif: rho_p {SN.rho_p:.3f} R2 {SN.D2:.3f} imgN {SN.imgN}"
                statstr_all = f"all: rho_p {SA.rho_p:.3f} R2 {SA.D2:.3f} imgN {SA.imgN}"
                y_manif = y_pred_manif[(Xtype, regrname)]
                y_gabor = y_pred_gabor[(Xtype, regrname)] if len(score_gabor) > 0 else []
                y_pasu = y_pred_pasu[(Xtype, regrname)] if len(score_pasu) > 0 else []
                y_natref = y_pred_natref[(Xtype, regrname)] if len(score_natref) > 0 else []

                plt.figure(figsize=[6, 5.5])
                plt.gca().axline((1, 1), slope=1, ls=":", color="k")
                plt.scatter(y_manif, score_manif, label="manif", alpha=0.3)
                plt.scatter(y_gabor, score_gabor, label="gabor", alpha=0.3)
                plt.scatter(y_pasu, score_pasu, label="pasu", alpha=0.3)
                plt.scatter(y_natref, score_natref, label="natref", alpha=0.3)
                plt.title(f"{Animal}-Exp{Expi:02d} {Xtype}-{regrname}\n{statstr_nat}\n{statstr_all}")
                plt.xlabel("Predicted Spike Rate")
                plt.ylabel("Observed Spike Rate")
                plt.legend()
                plt.savefig(join(expdir, f"{Xtype}_{regrname}_{featlayer}_pred_performance.png"))
                plt.show()



#%% Dev zone
# score_manif, imgfullpath_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)], stimdrive="N")
# Xfeat_manif = calc_reduce_features(score_manif, imgfullpath_manif, Xfeat_transformer, net, featlayer,)
# df_manif, eval_manif, y_pred_manif = evaluate_prediction(Xfeat_manif, score_manif, label=f"{featlayer}-{'Manif'}")

# for imgset_str in ["Manif", "Gabor", "Pasu", "EvolRef"]:
# label = f"{featlayer}-{'Manif'}"
# df_manif.to_csv(join(expdir, f"eval_predict_{label}.csv"), index=True)
# pkl.dump(eval_manif, open(join(expdir, f"eval_stats_{label}.pkl"), "wb"))
# pkl.dump(y_pred_manif, open(join(expdir, f"eval_predict_vector_{label}.pkl"), "wb"))
#%%
saveroot = r"E:\OneDrive - Harvard University\CNN_neural_regression"
os.makedirs(join(saveroot, "resnet50_linf8"), exist_ok=True)
featnet, net = load_featnet("resnet50_linf8")
#%%
for Expi in range(20, 47):
    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi,
                                "Evol", wdws=[(50, 200)], stimdrive="N")
    #%%
    featlayer = ".layer4.Bottleneck2"  # ".layer3.Bottleneck5" # ".layer2.Bottleneck3" # ".layer3.Bottleneck5" #
    for featlayer in [".layer2.Bottleneck3", ".layer3.Bottleneck5", ".layer4.Bottleneck2",]:
        feattsr = calc_features(score_vect, imgfullpath_vect, net, featlayer,)  # B x C x H x W

        featmat = feattsr.reshape(feattsr.shape[0], -1)  # B x (C*H*W)
        featmat_avg = feattsr.mean(axis=(2, 3))  # B x C
        del feattsr
        SRP = SparseRandomProjection().fit(featmat)
        srp_featmat = SRP.transform(featmat)  # B x n_components
        pca_featmat = PCA(n_components=500).fit_transform(featmat) # B x n_components
        score_spkcount = np.round(score_vect * 0.15)
        Xdict = {"srp": srp_featmat, "pca": pca_featmat, "all": featmat, "sp_avg": featmat_avg}
        ydict = {"count": score_spkcount, "rate": score_vect}
        #%%
        result_summary = {}
        ytype = "count"  # "rate"
        xtype = "sp_avg"  # "srp"  # "all" # "sp_avg"
        for xtype in ["sp_avg", "pca", "srp"]:  # "all" # "sp_avg"
            for ytype in ["count", "rate"]:
                X_all = Xdict[xtype]  # score_vect
                y_all = ydict[ytype]  # featmat_avg #   # featmat_avg  # srp_featmat  # featmat
                y_train, y_test, X_train, X_test = train_test_split(
                    y_all, X_all, test_size=0.2, random_state=42, shuffle=True
                )
                nfeat = X_all.shape[1]
                nsamp = X_all.shape[0]
                nsamp_train = X_train.shape[0]

                ridge = Ridge(alpha=1.0)
                poissreg = PoissonRegressor(alpha=1.0, max_iter=500)
                kr_rbf = KernelRidge(alpha=1.0, kernel="rbf", gamma=None, )
                kr_poly = KernelRidge(alpha=1.0, kernel="poly", gamma=None, )
                for estim, label in zip([ridge, poissreg, kr_rbf, kr_poly],
                                        ["Ridge", "Poisson", "Kernel_rbf", "Kernel_poly",]):
                    if nsamp > 3300 and label in ["Kernel_poly"]:  # out of memory
                        D2_test = np.nan
                        D2_train = np.nan
                        alpha = np.nan
                    else:
                        clf = GridSearchCV(estimator=estim, n_jobs=8,
                                           param_grid=dict(alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, 1E5],),
                                           ).fit(X_train, y_train)
                        D2_test = clf.score(X_test, y_test)
                        D2_train = clf.score(X_train, y_train)
                        alpha = clf.best_params_["alpha"]
                    result_summary[(xtype, ytype, label)] = \
                        {"alpha": alpha, "train_score": D2_train, "test_score": D2_test, "n_feat": nfeat}

                result_df = pd.DataFrame(result_summary)
                print(result_df)

        #%
        result_df.T.to_csv(join(saveroot, fr"resnet50_linf8\{Animal}_Exp{Expi:02}_resnet50_linf8_{featlayer}_regression_results.csv"))
#%%
Animal, Expi = "Alfa", 19
featlayer = ".layer3.Bottleneck5"
result_df = pd.read_csv(join(saveroot, fr"resnet50_linf8\{Animal}_Exp{Expi:02}_resnet50_linf8_{featlayer}_regression_results.csv"))
score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi,
                                "Evol", wdws=[(50, 200)], stimdrive="N")





