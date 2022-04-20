"""Simple implementation of error driven modelling of the an insilico visual unit

"""
import torch
# import torchvision.models as models
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import matplotlib.pylab as plt
from GAN_utils import upconvGAN
from insilico_Exp_torch import TorchScorer
from featvis_lib import load_featnet
from layer_hook_utils import featureFetcher
from ZO_HessAware_Optimizers import CholeskyCMAES
from layer_hook_utils import get_module_names, register_hook_by_module_names, layername_dict
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
dataroot = r"E:\OneDrive - Harvard University\CNN_neural_regression"
#%%
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%%
from neural_regress.insilico_modelling_lib import sweep_regressors, compare_activation_prediction, \
    PoissonRegressor, Ridge, KernelRidge, Lasso
# Build the regressors
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=1.0)
poissreg = PoissonRegressor(alpha=1.0, max_iter=500)
kr_rbf = KernelRidge(alpha=1.0, kernel="rbf", gamma=None, )
regressor_dict = {"Ridge": ridge, "Lasso": lasso, "Poisson": poissreg, "KernelRBF": kr_rbf}
#%%
# Target neuron network
scorer = TorchScorer("resnet50")
scorer.select_unit(("resnet50", ".layer3.Bottleneck5", 5, 6, 6), allow_grad=True)
#%%
regresslayer = ".layer3.Bottleneck5"
featnet, net = load_featnet("resnet50_linf8")
featFetcher = featureFetcher(featnet, input_size=(3, 227, 227),
                             device="cuda", print_module=False)
featFetcher.record(regresslayer,)
#%% Evolution experiment
feattsr_all = []
resp_all = []
optimizer = CholeskyCMAES(4096, population_size=None, init_sigma=3.0)
z_arr = np.zeros((1, 4096))  # optimizer.init_x
pbar = tqdm(range(100))
for i in pbar:
    imgs = G.visualize(torch.tensor(z_arr).float().cuda())
    resp = scorer.score(imgs, )
    z_arr_new = optimizer.step_simple(resp, z_arr)
    z_arr = z_arr_new
    with torch.no_grad():
        featnet(scorer.preprocess(imgs, input_scale=1.0))

    del imgs
    print(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
    pbar.set_description(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
    feattsr = featFetcher[regresslayer][:, :, 5, 5]
    feattsr_all.append(feattsr.cpu().numpy())
    resp_all.append(resp)

resp_all = np.concatenate(resp_all, axis=0)
feattsr_all = np.concatenate(feattsr_all, axis=0)
#%%

#%%
regressors = [ridge, kr_rbf, ]  # poissreg,
regressor_names = ["Ridge", "KernelRBF"]  # "Poisson",
Xdict = {"sp_rf": feattsr_rf}
y_all = resp_all
result_df, fit_models = sweep_regressors(Xdict, y_all, regressors, regressor_names,)
# result_df.to_csv(join(dataroot, "insilico_results\\sweep_regressors_sub.csv"))
# print(result_df)
#%% Error driven experiment
optimizer = CholeskyCMAES(4096, population_size=None, init_sigma=3.0)
z_arr = np.zeros((1, 4096))  # optimizer.init_x
pbar = tqdm(range(50))
feattsr_buffer = []
resp_buffer = []
pred_buffer = []
for i in pbar:
    imgs = G.visualize(torch.tensor(z_arr).float().cuda())
    resp = scorer.score(imgs, )
    with torch.no_grad():
        featnet(scorer.preprocess(imgs, input_scale=1.0))

    del imgs
    pbar.set_description(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
    feattsr = featFetcher[regresslayer][:, :, 5, 5].cpu().numpy()
    model_pred_score = fit_models["sp_rf", "Ridge"].predict(feattsr)

    error = resp - model_pred_score
    abserror = np.abs(error)
    z_arr_new = optimizer.step_simple(abserror, z_arr)
    z_arr = z_arr_new
    print(f"{i}: resp {resp.mean():.2f}+-{resp.std():.2f}", end=" ")
    print(f"pred {model_pred_score.mean():.2f}+-{model_pred_score.std():.2f}", end=" ")
    print(f"err {abserror.mean():.2f}+-{abserror.std():.2f}")
    feattsr_buffer.append(feattsr)
    resp_buffer.append(resp)
    pred_buffer.append(model_pred_score)

resp_new = np.concatenate(resp_buffer, axis=0)
pred_new = np.concatenate(pred_buffer, axis=0)
feattsr_new = np.concatenate(feattsr_buffer, axis=0)
#%  Add new buffer data to the pile.
resp_all = np.concatenate((resp_all, resp_new), axis=0)
feattsr_all = np.concatenate((feattsr_all, feattsr_new), axis=0)
print("Whole data bucket resp %s feature %s"%(resp_all.shape, feattsr_all.shape))
#%%
Xdict = {"sp_rf": feattsr_all}
y_all = resp_all
regressors = [ridge, ]  # poissreg, kr_rbf,
regressor_names = ["Ridge", ]  # "Poisson", "KernelRBF"
result_df2, fit_models = sweep_regressors(Xdict, y_all, regressors, regressor_names,)

#%% Evaluation on random dataset

#%%
def activity_driven_collection(scorer, featFetcher, feat_red_fun:lambda x:x,
                               feattsr_all=None, resp_all=None):
    if feattsr_all is None:
        feattsr_all = []
    if resp_all is None:
        resp_all = []
    optimizer = CholeskyCMAES(4096, population_size=None, init_sigma=3.0)
    z_arr = np.zeros((1, 4096))  # optimizer.init_x
    pbar = tqdm(range(100))
    for i in pbar:
        imgs = G.visualize(torch.tensor(z_arr).float().cuda())
        resp = scorer.score(imgs, )
        z_arr_new = optimizer.step_simple(resp, z_arr)
        z_arr = z_arr_new
        with torch.no_grad():
            featnet(scorer.preprocess(imgs, input_scale=1.0))

        del imgs
        print(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
        pbar.set_description(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
        feattsr = featFetcher[regresslayer]
        feat_reduce = feat_red_fun(feattsr).cpu().numpy() # [:, :, 5, 5]
        feattsr_all.append(feat_reduce)
        resp_all.append(resp)

    resp_all = np.concatenate(resp_all, axis=0)
    feattsr_all = np.concatenate(feattsr_all, axis=0)
    return resp_all, feattsr_all


def error_driven_collection(scorer, featFetcher, fit_model,
                            feat_red_fun:lambda x:x, feattsr_all=None, resp_all=None):
    optimizer = CholeskyCMAES(4096, population_size=None, init_sigma=3.0)
    z_arr = np.zeros((1, 4096))  # optimizer.init_x
    pbar = tqdm(range(50))
    feattsr_buffer = []
    resp_buffer = []
    pred_buffer = []
    generations = []
    for i in pbar:
        imgs = G.visualize(torch.tensor(z_arr).float().cuda())
        resp = scorer.score(imgs, )
        with torch.no_grad():
            featnet(scorer.preprocess(imgs, input_scale=1.0))

        del imgs
        pbar.set_description(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
        feattsr = featFetcher[regresslayer]
        feat_reduce = feat_red_fun(feattsr).cpu().numpy()  # [:, :, 5, 5]
        model_pred_score = fit_model.predict(feat_reduce)

        error = resp - model_pred_score
        abserror = np.abs(error)
        z_arr_new = optimizer.step_simple(abserror, z_arr)
        z_arr = z_arr_new
        print(f"{i}: resp {resp.mean():.2f}+-{resp.std():.2f}", end=" ")
        print(f"pred {model_pred_score.mean():.2f}+-{model_pred_score.std():.2f}", end=" ")
        print(f"err {abserror.mean():.2f}+-{abserror.std():.2f}")
        feattsr_buffer.append(feat_reduce)
        resp_buffer.append(resp)
        pred_buffer.append(model_pred_score)
        generations.append(i * np.ones_like(resp))

    resp_new = np.concatenate(resp_buffer, axis=0)
    pred_new = np.concatenate(pred_buffer, axis=0)
    feattsr_new = np.concatenate(feattsr_buffer, axis=0)
    generations = np.concatenate(generations, axis=0)
    # %  Add new buffer data to the pile.
    print("Pearson r: %.3f (P=%.1e)" % pearsonr(pred_new, resp_new))
    print("Spearman r: %.3f (P=%.1e)" % spearmanr(pred_new, resp_new))

    feattsr_all = feattsr_new if feattsr_all is None \
            else np.concatenate((feattsr_all, feattsr_new), axis=0)
    resp_all = resp_new if resp_all is None \
            else np.concatenate((resp_all, resp_new), axis=0)
    print("Whole data bucket resp %s feature %s" % (resp_all.shape, feattsr_all.shape))
    return resp_new, pred_new, generations, resp_all, feattsr_all


def update_model(resp_all, feattsr_all, regressor_key="Ridge",):
    if isinstance(regressor_key, str):
        regressor_key = (regressor_key,)

    Xdict = {"sp_rf": feattsr_all}
    y_all = resp_all
    regressors = [regressor_dict[k] for k in regressor_key]  # poissreg, kr_rbf,
    regressor_names = [*regressor_key]  # "Poisson", "KernelRBF"
    result_df2, fit_models = sweep_regressors(Xdict, y_all, regressors, regressor_names,)
    return result_df2, fit_models


def eval_model_randomGAN(scorer, featFetcher, fit_model, feat_red_fun:lambda x:x,
                         n_batch=100, batchsize=40, codescale=1.0,):
    pbar = tqdm(range(n_batch))
    eval_resp_col = []
    eval_pred_col = []
    for i in pbar:
        z_arr = np.random.randn(batchsize, 4096) * codescale
        imgs = G.visualize(torch.tensor(z_arr).float().cuda())
        resp = scorer.score(imgs, )
        with torch.no_grad():
            featnet(scorer.preprocess(imgs, input_scale=1.0))

        del imgs
        feattsr = featFetcher[regresslayer]
        feat_reduce = feat_red_fun(feattsr).cpu().numpy() # [:, :, 5, 5]
        model_pred_score = fit_model.predict(feat_reduce)
        pbar.set_description(f"{i}: {resp.mean():.2f}+-{resp.std():.2f} pred {model_pred_score.mean():.2f}+-{model_pred_score.std():.2f}")
        eval_resp_col.append(resp)
        eval_pred_col.append(model_pred_score)

    eval_pred_vec = np.concatenate(eval_pred_col, axis=0)
    eval_resp_vec = np.concatenate(eval_resp_col, axis=0)
    rho_s = spearmanr(eval_pred_vec, eval_resp_vec)
    rho_p = pearsonr(eval_pred_vec, eval_resp_vec)
    print(f"Pearson r: {rho_p[0]:.2f} (P={rho_p[1]:.1e})")
    print(f"Spearman r: {rho_s[0]:.2f} (P={rho_s[1]:.1e})")
    return eval_resp_vec, eval_pred_vec, rho_s, rho_p

#%%
scorer = TorchScorer("resnet50_linf_8")
scorer.select_unit(("resnet50_linf_8", ".layer3.Bottleneck5", 5, 6, 6), allow_grad=True)

regresslayer = ".layer3.Bottleneck5"
featnet, net = load_featnet("resnet50_linf8")
featFetcher = featureFetcher(featnet, input_size=(3, 227, 227),
                             device="cuda", print_module=False)
featFetcher.record(regresslayer,)
feat_red_fun = lambda tsr: tsr[:, :, 7, 7]

#%% Active learning loop
regr_key = "Ridge"#"Ridge"
resp_all, feattsr_all = activity_driven_collection(scorer, featFetcher, feat_red_fun, )
#%
for itr in range(20):
    result_df, fit_models = update_model(resp_all, feattsr_all, regressor_key=regr_key,)
    resp_new, pred_new, generations, resp_all, feattsr_all = error_driven_collection(\
                                scorer, featFetcher,
                                fit_models["sp_rf", regr_key], feat_red_fun,
                                feattsr_all=feattsr_all, resp_all=resp_all)
    plt.figure()
    plt.scatter(generations, resp_new, s=16, label="resp", alpha=0.7)
    plt.scatter(generations, pred_new, s=16, label="prediction", alpha=0.7)
    plt.legend()
    plt.show()

eval_resp_vec, eval_pred_vec, rho_s, rho_p = \
    eval_model_randomGAN(scorer, featFetcher, fit_models["sp_rf", regr_key], feat_red_fun, \
                         n_batch=100, batchsize=40)
print(f"Pearson r: {rho_p[0]:.2f} (P={rho_p[1]:.1e})")
print(f"Spearman r: {rho_s[0]:.2f} (P={rho_s[1]:.1e})")
#%% Pos
respccmat = np.corrcoef(resp_all[:, np.newaxis].T, feattsr_all.T)
ccoef = respccmat[0, 1:]
wcoef = fit_models["sp_rf", regr_key].best_estimator_.coef_
#%%
wcc_rho, wcc_rho_P = pearsonr(wcoef, ccoef)
print("Correlation between resp and feature unit (same unit in neighboring columns) %.3f"%ccoef[5])
print("Correlation between Ridge regression weight and correlation coef %.3f" % wcc_rho)

#%%
plt.figure(figsize=(5, 5))
plt.scatter(ccoef, wcoef, s=25, alpha=0.5)
plt.axis("equal")
plt.xlabel("Correlation coef")
plt.ylabel("Ridge regression weight")
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize=(5, 5))
plt.scatter(eval_resp_vec, eval_pred_vec, s=25, alpha=0.5)
plt.axis("equal")
plt.xlabel("Real Resp")
plt.ylabel("Pred Resp")
plt.tight_layout()
plt.show()
