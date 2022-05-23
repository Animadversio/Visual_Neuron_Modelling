"""
Fit a factorized regression model onto the data.
* Preparation
    - Load network
    - Load experiment data
    - Load correlation tensor previously computed (for a layer)
* Fit the factorized regression model, and predict the responses.
    - Use the correlated spatial mask to do feature selection, and fit the tensors.
    - Fit the regression of the features using Ridge or Lasso.
    - Predict the responses of other image datasets.
    - Save the results.
"""
import os
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
#%%
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


def merge_dict_arrays(*dict_arrays):
    """
    Merge a list of dicts into a single dict.
    """
    keys = []
    for d in dict_arrays:
        if len(d) != 0:
            keys = [*d.keys()]
            break

    for d in dict_arrays:
        if len(d) == 0:
            for k in keys:
                d[k] = np.array([])

    merged_dict = {}
    # TODO: solve the empty dict problem
    for key in keys:
        merged_dict[key] = np.concatenate([d[key] for d in dict_arrays], axis=0)

    return merged_dict


def merge_arrays(*arrays):
    """
    Merge a list of arrays into a single array.
    """
    return np.concatenate(arrays, axis=0)


def evaluate_dict(y_pred_dict, y_true, label, savedir=None):
    print(label, f"  N imgs: {len(y_true)}")
    eval_dict = {}
    for (Xtype, regrname), y_pred in y_pred_dict.items():
        D2 = 1 - np.var(y_pred - y_true) / np.var(y_true)  # regr.score(Xfeat_dict[Xtype], y_true)
        rho_p, pval_p = pearsonr(y_pred, y_true)
        rho_s, pval_s = spearmanr(y_pred, y_true)
        print(
            f"{Xtype} {regrname} Prediction Pearson: {rho_p:.3f} {pval_p:.1e} Spearman: {rho_s:.3f} {pval_s:.1e} D2: {D2:.3f}")
        eval_dict[(Xtype, regrname)] = {"rho_p": rho_p, "pval_p": pval_p, "rho_s": rho_s, "pval_s": pval_s,
                                        "D2": D2, "imgN": len(y_true)}
    parts = label.split("-")
    layer, datasetstr = parts[-2], parts[-1]
    df = pd.DataFrame(eval_dict).T
    df["label"] = label
    df["layer"] = layer
    df["img_space"] = datasetstr
    if savedir is not None:
        df.to_csv(join(savedir, f"eval_predict_{label}.csv"), index=True)
        pkl.dump(eval_dict, open(join(savedir, f"eval_stats_{label}.pkl"), "wb"))
        pkl.dump(y_pred_dict, open(join(savedir, f"eval_predvec_{label}.pkl"), "wb"))

    return df, eval_dict, y_pred_dict


#%% Import Prediction pipeline from libray
from neural_regress.regress_lib import calc_features, \
        calc_reduce_features, sweep_regressors, evaluate_prediction, \
        Ridge, Lasso, PoissonRegressor, RidgeCV, LassoCV, LinearRegression, train_test_split

mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
# saveroot = r"E:\OneDrive - Harvard University\CNN_neural_regression"
saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
featnet, net = load_featnet("resnet50_linf8")
# %%
import matplotlib
matplotlib.use("Agg") # 'module://backend_interagg'
#%%
"""Remember to save the factors """
#%%
# layer = "layer3"
# featlayer = ".layer3.Bottleneck5"

# layer = "layer4"
# featlayer = ".layer4.Bottleneck2"
layer = "layer2"
featlayer = ".layer2.Bottleneck3"
bdr = 1
for Animal in ["Alfa", "Beto"]:
    MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    for Expi in range(1, len(EStats) + 1):
        # if Animal == "Alfa" and Expi <= 10: continue
        expdir = join(saveroot, f"{Animal}_{Expi:02d}")
        os.makedirs(expdir, exist_ok=True)
        # load the image paths and the response vector
        score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)], stimdrive="N")
        # get Hmap, ccfactor.
        # raise NotImplementedError
        #%%
        Hmat32, Hmaps32, ccfactor32, FactStat2 = load_NMF_factors(Animal, Expi, layer, NF=3)
        #%%
        # Use Hmaps to get the projected feature vector
        # Use Hmaps to define a X_transform function in dict.
        Hmat3, Hmaps3, ccfactor3, FactStat = load_NMF_factors(Animal, Expi, layer, NF=3)
        padded_mask3 = np.pad(Hmaps3[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
        Xfeat_transformer = {"spmask3": lambda tsr: np.einsum("BCHW,HWF->BFC", tsr, padded_mask3).reshape(tsr.shape[0], -1),
                             "featvec3": lambda tsr: np.einsum("BCHW,CF->BFHW", tsr, ccfactor3).reshape(tsr.shape[0], -1),
                             "factor3": lambda tsr: np.einsum("BCHW,CF,HWF->BF", tsr, ccfactor3, padded_mask3).reshape(tsr.shape[0], -1),
                             "facttsr3": lambda tsr: np.einsum("BCHW,CF,HWF->B", tsr, ccfactor3, padded_mask3).reshape(tsr.shape[0], -1),  }

        Hmat1, Hmaps1, ccfactor1, _ = load_NMF_factors(Animal, Expi, layer, NF=1)
        padded_mask1 = np.pad(Hmaps1[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
        Xfeat_transformer.update({
            "spmask1": lambda tsr: np.einsum("BCHW,HWF->BFC", tsr, padded_mask1).reshape(tsr.shape[0], -1),
            "featvec1": lambda tsr: np.einsum("BCHW,CF->BFHW", tsr, ccfactor1).reshape(tsr.shape[0], -1),
            "factor1": lambda tsr: np.einsum("BCHW,CF,HWF->BF", tsr, ccfactor1, padded_mask1).reshape(tsr.shape[0], -1),
            "facttsr1": lambda tsr: np.einsum("BCHW,CF,HWF->B", tsr, ccfactor1, padded_mask1).reshape(tsr.shape[0], -1),
        })

        Xfeat_dict = calc_reduce_features(score_vect, imgfullpath_vect, Xfeat_transformer, net, featlayer, img_dim=(224, 224))
        #%% Fit the model with RidgeCV or LassoCV
        y_all = score_vect
        ridge = Ridge(alpha=1.0)
        lasso = Lasso(alpha=1.0)
        # poissreg = PoissonRegressor(alpha=1.0, max_iter=500)
        # kr_rbf = KernelRidge(alpha=1.0, kernel="rbf", gamma=None, )
        regressors = [ridge, lasso]
        regressor_names = ["Ridge", "Lasso"]
        result_df, fit_models = sweep_regressors(Xfeat_dict, y_all, regressors, regressor_names, )
        # %%
        result_df["layer"] = featlayer
        result_df["Expi"] = Expi
        result_df["Animal"] = Animal
        # save results
        result_df.to_csv(join(expdir, f"{featlayer}_factor-regress_results.csv"), index=True)
        pkl.dump(fit_models, open(join(expdir, f"{featlayer}_factor-regress_models.pkl"), "wb"))
        # %%
        df_evol, eval_evol, y_pred_evol = evaluate_prediction(fit_models, Xfeat_dict, score_vect,
                                                  label=f"factreg-{featlayer}-{'Evol'}", savedir=expdir)
        # %
        score_manif, imgfullpath_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)],stimdrive="N")
        Xfeat_manif = calc_reduce_features(score_manif, imgfullpath_manif, Xfeat_transformer, net, featlayer, img_dim=(224, 224))
        df_manif, eval_manif, y_pred_manif = evaluate_prediction(fit_models, Xfeat_manif, score_manif,
                                                  label=f"factreg-{featlayer}-{'Manif'}", savedir=expdir)
        # %
        score_pasu, imgfullpath_pasu = load_score_mat(EStats, MStats, Expi, "Pasu_avg", wdws=[(50, 200)], stimdrive="N")
        Xfeat_pasu = calc_reduce_features(score_pasu, imgfullpath_pasu, Xfeat_transformer, net, featlayer, img_dim=(224, 224))
        df_pasu, eval_pasu, y_pred_pasu = evaluate_prediction(fit_models, Xfeat_pasu, score_pasu,
                                                  label=f"factreg-{featlayer}-{'Pasu'}", savedir=expdir)
        # %
        score_gabor, imgfullpath_gabor = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)],stimdrive="N")
        Xfeat_gabor = calc_reduce_features(score_gabor, imgfullpath_gabor, Xfeat_transformer, net, featlayer, img_dim=(224, 224))
        df_gabor, eval_gabor, y_pred_gabor = evaluate_prediction(fit_models, Xfeat_gabor, score_gabor,
                                                  label=f"factreg-{featlayer}-{'Gabor'}", savedir=expdir)
        # %
        score_natref, imgfullpath_natref = load_score_mat(EStats, MStats, Expi, "EvolRef_avg", wdws=[(50, 200)],stimdrive="N")
        Xfeat_natref = calc_reduce_features(score_natref, imgfullpath_natref, Xfeat_transformer, net, featlayer, img_dim=(224, 224))
        df_natref, eval_natref, y_pred_natref = evaluate_prediction(fit_models, Xfeat_natref, score_natref,
                                                   label=f"factreg-{featlayer}-{'EvolRef'}", savedir=expdir)
        #%%
        y_merge_dict = merge_dict_arrays(y_pred_pasu, y_pred_gabor, y_pred_natref)
        score_merge = merge_arrays(score_pasu, score_gabor, score_natref)
        df_natmerge, eval_natmerge, _ = evaluate_dict(y_merge_dict, score_merge,
                                                      label=f"factreg-{featlayer}-allnat", savedir=expdir)

        y_merge2_dict = merge_dict_arrays(y_pred_manif, y_pred_pasu, y_pred_gabor, y_pred_natref)
        score_merge2 = merge_arrays(score_manif, score_pasu, score_gabor, score_natref)
        df_merge, eval_merge, _ = evaluate_dict(y_merge2_dict, score_merge2,
                                                label=f"factreg-{featlayer}-all", savedir=expdir)
        # %%
        for (Xtype, regrname) in fit_models.keys():
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
            plt.savefig(join(expdir, f"{Xtype}_{regrname}_{featlayer}_factreg_pred_performance.png"))
            plt.show()
#%%

#%% Scratches
layer = "layer3"
netname = "resnet50_linf8"
scorer = CorrFeatScore()
scorer.register_hooks(net, layer, netname="resnet50_linf8")
#%%
figdir = r"E:\OneDrive - Harvard University\CNN_neural_regression\resnet50_linf8\CorrFeatTsr_predict"
#%%
DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)
scorer.register_weights({layer: DR_Wtsr})
pred_score = score_images(featnet, scorer, layer, imgfullpath_vect, imgloader=loadimg_preprocess, batchsize=80,)
scorer.clear_hook()
nlfunc, popt, pcov, scaling, nlpred_score, Stat = fitnl_predscore(pred_score.numpy(), score_vect)
#%%
from CorrFeatTsr_predict_lib import score_images_torchdata
scorer = CorrFeatScore()
scorer.register_hooks(net, layer, netname="resnet50_linf8")
rank1_Wtsr = [pad_factor_prod(Hmaps[:, :, i:i+1], ccfactor[:, i:i+1], bdr=bdr) for i in range(3)]
stack_Wtsr = np.stack(rank1_Wtsr, axis=0)
scorer.register_weights({layer: stack_Wtsr})
pred_score = score_images_torchdata(featnet, scorer, layer,
                imgfullpath_vect, batchsize=80, workers=6)
scorer.clear_hook()

#%%
nlfunc, popt, pcov, scaling, nlpred_score, Stat = fitnl_predscore(pred_score.numpy(), score_vect)
#%%
train_X, test_X, train_y, test_y \
    = train_test_split(pred_score.numpy(), score_vect, test_size=0.2, random_state=42)
clf = LinearRegression(normalize=True).fit(train_X, train_y, )
D2_test = clf.score(test_X, test_y)  # 0.16728044321972735
D2_train = clf.score(train_X, train_y)  # 0.16270582493435726
print(f"Independent weights for three factors: Train {D2_train:.3f}, test {D2_test:.3f}")
print(f"Factor weights {clf.coef_}")
#%%
clf = LinearRegression().fit(train_X.mean(axis=1, keepdims=True), train_y, )
D2_test = clf.score(test_X.mean(axis=1, keepdims=True), test_y)  # 0.1532078264571166
D2_train = clf.score(train_X.mean(axis=1, keepdims=True), train_y)  # 0.15515980956827935
print(f"Merged three factors: Train {D2_train:.3f}, test {D2_test:.3f}")
#%%

#%%
#%%
# pred = train_test_split(pred_score.numpy(), score_vect, test_size=0.2, random_state=42)
pred_train, pred_test, orig_train, orig_test = train_test_split(nlpred_score, score_vect, test_size=0.2, random_state=42)
print()
#%%
Animal, Expi = "Alfa", 19
featlayer = ".layer4.Bottleneck2"
score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)], stimdrive="N")
feattsr = calc_features(score_vect, imgfullpath_vect, net, featlayer, workers=12, batch_size=64)
# 4000 images, batch_size=64, workers=8 took ~19 sec.
#%%