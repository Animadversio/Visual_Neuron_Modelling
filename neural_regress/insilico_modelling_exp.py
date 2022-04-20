import torch
import torchvision.models as fit_models
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

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
dataroot = r"E:\OneDrive - Harvard University\CNN_neural_regression"
#%%
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
# scorer = TorchScorer("resnet50")
# module_names, module_types, module_spec = get_module_names(scorer.model, input_size=(3, 227, 227), device="cuda");
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
    feattsr = featFetcher[regresslayer]
    feattsr_all.append(feattsr.cpu().numpy())
    resp_all.append(resp)

resp_all = np.concatenate(resp_all, axis=0)
feattsr_all = np.concatenate(feattsr_all, axis=0)
#%% Linear Modelling imports
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import johnson_lindenstrauss_min_dim, \
            SparseRandomProjection, GaussianRandomProjection
from sklearn.linear_model import LogisticRegression, LinearRegression, \
    Ridge, Lasso, PoissonRegressor, RidgeCV, LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
#%%
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Resize
from torch_utils import show_imgrid
from NN_PC_visualize.NN_PC_lib import \
    create_imagenet_valid_dataset, Dataset, DataLoader

denormalizer = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                         std=[1/0.229, 1/0.224, 1/0.225])
normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
resizer = Resize(227, )


def sweep_regressors(Xdict, y_all, regressors, regressor_names,):
    result_summary = {}
    models = {}
    idx_train, idx_test = train_test_split(
        np.arange(len(y_all)), test_size=0.2, random_state=42, shuffle=True
    )
    for xtype in Xdict:
        X_all = Xdict[xtype]  # score_vect
        y_train, y_test = y_all[idx_train], y_all[idx_test]
        X_train, X_test = X_all[idx_train], X_all[idx_test]
        nfeat = X_train.shape[1]
        for estim, label in zip(regressors, regressor_names):
            clf = GridSearchCV(estimator=estim, n_jobs=8,
                               param_grid=dict(alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, 1E5], ),
                               ).fit(X_train, y_train)
            D2_test = clf.score(X_test, y_test)
            D2_train = clf.score(X_train, y_train)
            alpha = clf.best_params_["alpha"]
            result_summary[(xtype, label)] = \
                {"alpha": alpha, "train_score": D2_train, "test_score": D2_test, "n_feat": nfeat}
            models[(xtype, label)] = clf

        result_df = pd.DataFrame(result_summary)

    print(result_df.T)
    return result_df.T, models


def compare_activation_prediction(target_scores_natval, pred_scores_natval, exptitle=""):
    result_col = {}
    for k in pred_scores_natval:
        rho_s = spearmanr(target_scores_natval, pred_scores_natval[k])
        rho_p = pearsonr(target_scores_natval, pred_scores_natval[k])
        R2 = 1 - np.var(pred_scores_natval[k] - target_scores_natval) / np.var(target_scores_natval)
        print(k, f"spearman: {rho_s.correlation:.3f} P={rho_s.pvalue:.1e}",
                 f"pearson: {rho_p[0]:.3f} P={rho_p[1]:.1e} R2={R2:.3f}")
        result_col[k] = {"spearman": rho_s.correlation, "pearson": rho_p[0],
                         "spearman_pval": rho_s.pvalue, "pearson_pval": rho_p[1],
                         "R2": R2, "dataset": exptitle, "n_sample": len(target_scores_natval)}

        plt.figure()
        plt.scatter(target_scores_natval, pred_scores_natval[k], s=16, alpha=0.5)
        plt.xlabel("Target scores")
        plt.ylabel("Predicted scores")
        plt.axis('equal')
        plt.title(f"{exptitle} {k}\n"
                  f"corr pearsonr {rho_p[0]:.3f} P={rho_p[1]:.1e}\n"
                  f"corr spearmanr {rho_s.correlation:.3f} P={rho_s.pvalue:.1e} R2={R2:.3f}")
        plt.tight_layout()
        plt.savefig(join(dataroot, "insilico_results", f"{exptitle}_{k}_regress.png"))
        plt.show()

    test_result_df = pd.DataFrame(result_col)
    test_result_df.T.to_csv(join(dataroot, "insilico_results", f"{exptitle}_regress_results.csv"))
    return test_result_df.T

#%% Prepare image features and feature tfms for modelling
featmat = feattsr_all.reshape(feattsr_all.shape[0], -1)  # B x (C*H*W)
featmat_avg = feattsr_all.mean(axis=(2, 3))  # B x C
featmat_rf = feattsr_all[:, :, 6, 6]  # B x n_components
srp = SparseRandomProjection().fit(featmat)
srp_featmat = srp.transform(featmat)  # B x n_components
pca = PCA(n_components=500)
pca_featmat = pca.fit_transform(featmat)  # B x n_components

Xdict = {"srp": srp_featmat, "pca": pca_featmat,
         "sp_avg": featmat_avg, "sp_rf": featmat_rf}
Xfeat_transformer = {'pca': lambda tsr: pca.transform(tsr.reshape(tsr.shape[0], -1)),
                     "srp": lambda tsr: srp.transform(tsr.reshape(tsr.shape[0], -1)),
                     "sp_rf": lambda tsr: tsr[:, :, 6, 6],
                     "sp_avg": lambda tsr: tsr.mean(axis=(2, 3))}
#%%
ridge = Ridge(alpha=1.0)
poissreg = PoissonRegressor(alpha=1.0, max_iter=500)
kr_rbf = KernelRidge(alpha=1.0, kernel="rbf", gamma=None, )
regressors = [ridge, poissreg, kr_rbf, ]
regressor_names = ["Ridge", "Poisson", "KernelRBF"]
y_all = resp_all

#%%
result_df, fit_models = sweep_regressors(Xdict, y_all, regressors, regressor_names, )
result_df.to_csv(join(dataroot, "insilico_results\\sweep_regressors.csv"))
#%%
result_df, fit_models = sweep_regressors(Xdict, y_all, [ridge, kr_rbf], ["Ridge","KernelRBF"],)
result_df.to_csv(join(dataroot, "insilico_results\\sweep_regressors_sub.csv"))
#%%
"""Predict Scores for Evolution images"""

model_list = [('pca', "Ridge"), ('srp', "Ridge"), ('sp_avg', "Ridge"),
              ('sp_rf', "Ridge"), ('sp_rf', "KernelRBF")]

pred_scores_evol = defaultdict(list)
for k in model_list:
    featmat_tfm = Xdict[k[0]] #Xfeat_transformer[k[0]](feattsr)
    pred_score = fit_models[k].predict(featmat_tfm)
    pred_scores_evol[k] = pred_score

#%%
idx_train, idx_test = train_test_split(
        range(len(y_all)), test_size=0.2, random_state=42, shuffle=True
    )
target_scores_evol_test = resp_all[idx_test]
target_scores_evol_train = resp_all[idx_train]
pred_scores_evol_test = {k: v[idx_test] for k, v in pred_scores_evol.items()}
pred_scores_evol_train = {k: v[idx_train] for k, v in pred_scores_evol.items()}
compare_activation_prediction(target_scores_evol_test, pred_scores_evol_test,
                            exptitle="evolution-test")
compare_activation_prediction(target_scores_evol_train, pred_scores_evol_train,
                            exptitle="evolution-train")

#%%
dataset = create_imagenet_valid_dataset(imgpix=227, normalize=True)
data_loader = DataLoader(dataset, batch_size=120,
              shuffle=False, num_workers=8)
#%% Get scores from ImageNet validation set
target_scores_natval = []
for i, (imgs, _) in tqdm(enumerate(data_loader)):
    imgs = imgs.cuda()
    with torch.no_grad():
        score_batch = scorer.score(denormalizer(imgs))
    target_scores_natval.append(score_batch)
    break

target_scores_natval = np.concatenate(target_scores_natval, axis=0)
#%%
""" Predicting ImageNet validation set. """
dataset = create_imagenet_valid_dataset(imgpix=227, normalize=True)
data_loader = DataLoader(dataset, batch_size=100,
                         shuffle=False, num_workers=8)

model_list = [('pca', "Ridge"), ('srp', "Ridge"), ('sp_avg', "Ridge"),
              ('sp_rf', "Ridge"), ('sp_rf', "KernelRBF")]
target_scores_natval = []
pred_scores_natval = defaultdict(list)
for i, (imgs, _) in tqdm(enumerate(data_loader)):
    imgs = imgs.cuda()
    with torch.no_grad():
        # score_batch = scorer.score(denormalizer(imgs))
        score_batch = scorer.score(imgs, skip_preprocess=True)

    target_scores_natval.append(score_batch)
    with torch.no_grad():
        featnet(imgs)
        feattsr = featFetcher[regresslayer]
        feattsr = feattsr.cpu().numpy()

    for k in model_list:
        featmat_tfm = Xfeat_transformer[k[0]](feattsr)
        pred_score = fit_models[k].predict(featmat_tfm)
        pred_scores_natval[k].append(pred_score)


target_scores_natval = np.concatenate(target_scores_natval, axis=0)
for k in pred_scores_natval:
    pred_scores_natval[k] = np.concatenate(pred_scores_natval[k], axis=0)

#%%
compare_activation_prediction(target_scores_natval, pred_scores_natval, "ImageNet-valudation")

#%%
np.corrcoef(target_scores_natval, pred_scores_natval['srp', "Ridge"])
#%%
msk = target_scores_natval > 0.1
pearsonr(target_scores_natval[msk], pred_scores_natval['srp', "Ridge"][msk])
#%%
np.corrcoef(target_scores_natval, pred_scores_natval['sp_avg', "Ridge"])

#%%


#%%
sortidx = np.argsort(target_scores_natval)
print(target_scores_natval[sortidx[-10:]])
imgs = [denormalizer(dataset[i][0]) for i in sortidx[-10:]]
show_imgrid(imgs, nrow=5)
#%%
plt.hist(target_scores_natval, bins=100,alpha=0.4)
plt.hist(pred_scores_natval["pca", "Ridge"], bins=100, alpha=0.2)
plt.hist(pred_scores_natval["srp", "Ridge"], bins=100, alpha=0.2)
plt.legend(["Target scores", "PCA", "SRP"])
plt.xlim([-1, 2])
plt.ylim([0, 200])
plt.show()


#%%
""" Predicting GAN images. """
target_scores_gan = []
pred_scores_gan = defaultdict(list)
for i in tqdm(range(200)):
    imgs = G.visualize(2 * torch.randn(40, 4096).cuda())
    with torch.no_grad():
        score_batch = scorer.score(imgs)
    target_scores_gan.append(score_batch)

    with torch.no_grad():
        featnet(resizer(normalizer(imgs)))
        feattsr = featFetcher[regresslayer]
        feattsr = feattsr.cpu().numpy()
        featmat = feattsr.reshape(feattsr.shape[0], -1)

    featmat_pca = pca.transform(featmat)
    pred_score_pca = fit_models['pca', "Ridge"].predict(featmat_pca)
    pred_scores_gan['pca', "Ridge"].append(pred_score_pca)

    featmat_srp = srp.transform(featmat)
    pred_score_srp = fit_models['srp', "Ridge"].predict(featmat_srp)
    pred_scores_gan['srp', "Ridge"].append(pred_score_srp)
    pred_score_srp = fit_models['srp', "KernelRBF"].predict(featmat_srp)
    pred_scores_gan['srp', "KernelRBF"].append(pred_score_srp)

    featmat_avg = feattsr.mean(axis=(2, 3))
    pred_score_avg = fit_models['sp_avg', "Ridge"].predict(featmat_avg)
    pred_scores_gan['sp_avg', "Ridge"].append(pred_score_avg)

    featmat_rf = feattsr[:, :, 6, 6]
    pred_score_rf = fit_models['sp_rf', "Ridge"].predict(featmat_rf)
    pred_scores_gan['sp_rf', "Ridge"].append(pred_score_rf)

for k in pred_scores_gan:
    pred_scores_gan[k] = np.concatenate(pred_scores_gan[k], axis=0)

target_scores_gan = np.concatenate(target_scores_gan, axis=0)
#%%
compare_activation_prediction(target_scores_gan, pred_scores_gan, "GAN-random")
#%%
#%%
model_list = [('pca', "Ridge"), ('srp', "Ridge"), ('sp_avg', "Ridge"),
              ('sp_rf', "Ridge"), ('sp_rf', "KernelRBF")]
target_scores_gan = []
pred_scores_gan = defaultdict(list)
for i in tqdm(range(200)):
    imgs = G.visualize(2 * torch.randn(40, 4096).cuda())
    with torch.no_grad():
        score_batch = scorer.score(imgs)
    target_scores_gan.append(score_batch)

    with torch.no_grad():
        featnet(resizer(normalizer(imgs)))
        feattsr = featFetcher[regresslayer]
        feattsr = feattsr.cpu().numpy()
        # featmat = feattsr.reshape(feattsr.shape[0], -1)

    for k in model_list:
        featmat_tfm = Xfeat_transformer[k[0]](feattsr)
        pred_score = fit_models[k].predict(featmat_tfm)
        pred_scores_gan[k].append(pred_score)

for k in pred_scores_gan:
    pred_scores_gan[k] = np.concatenate(pred_scores_gan[k], axis=0)

target_scores_gan = np.concatenate(target_scores_gan, axis=0)
#%%
compare_activation_prediction(target_scores_gan, pred_scores_gan, "GAN-random_std2")

#%% Another evolution experiment
model_list = [('pca', "Ridge"), ('srp', "Ridge"), ('sp_avg', "Ridge"),
              ('sp_rf', "Ridge"), ('sp_rf', "KernelRBF")]
target_scores_reevol = []
pred_scores_reevol = defaultdict(list)

optimizer = CholeskyCMAES(4096, population_size=None, init_sigma=3.0)
z_arr = np.zeros((1, 4096))  # optimizer.init_x
for i in tqdm(range(100)):
    imgs = G.visualize(torch.tensor(z_arr).float().cuda())
    with torch.no_grad():
        score_batch = scorer.score(imgs)
    target_scores_reevol.append(score_batch)
    z_arr = optimizer.step_simple(score_batch, z_arr)

    with torch.no_grad():
        featnet(resizer(normalizer(imgs)))
        feattsr = featFetcher[regresslayer]
        feattsr = feattsr.cpu().numpy()

    for k in model_list:
        featmat_tfm = Xfeat_transformer[k[0]](feattsr)
        pred_score = fit_models[k].predict(featmat_tfm)
        pred_scores_reevol[k].append(pred_score)

for k in pred_scores_gan:
    pred_scores_reevol[k] = np.concatenate(pred_scores_reevol[k], axis=0)

target_scores_reevol = np.concatenate(target_scores_reevol, axis=0)

#%%
compare_activation_prediction(target_scores_reevol, pred_scores_reevol,
                              "Reevolution-SameUnit2")
#%%
show_imgrid(imgs, )
#%% Synopsis
df_evoltrain = compare_activation_prediction(target_scores_evol_train, pred_scores_evol_train, "evolution-train")
df_evoltest = compare_activation_prediction(target_scores_evol_test, pred_scores_evol_test, "evolution-test")
df_GANrand = compare_activation_prediction(target_scores_gan, pred_scores_gan, "GAN-random_std2")
df_reevol = compare_activation_prediction(target_scores_reevol, pred_scores_reevol, "Reevolution-SameUnit2")
df_ImageNet = compare_activation_prediction(target_scores_natval, pred_scores_natval, "ImageNet-valudation")
df_synopsis = pd.concat([df_evoltrain, df_evoltest, df_reevol, df_GANrand, df_ImageNet], axis=0)
df_synopsis.to_csv(join(dataroot, "insilico_results", "synopsis.csv"))
#%%
df_synopsis = df_synopsis.astype({'spearman': 'float64', 'pearson': 'float64',
                    "spearman_pval": 'float64', "pearson_pval": 'float64', "R2": 'float64',
                    "dataset": str, "n_sample": int})
df_synops_col = df_synopsis.reset_index()
df_synops_col.rename(columns={"level_0": "xtype", "level_1": "regressor"}, inplace=True)
#%%
#%% Evaluate the generalization gap
for stat in ["pearson", "spearman", "R2"]:
    df_synops_col.groupby(["xtype", "regressor", "dataset"], sort=False)[stat].mean()\
        .unstack(level=[0,1]).plot(kind="barh",)
    plt.xlabel(stat)
    plt.tight_layout()
    plt.savefig(join(dataroot, "insilico_results", \
                     "model_generalization_synopsis_" + stat + ".png"))
    plt.savefig(join(dataroot, "insilico_results", \
                     "model_generalization_synopsis_" + stat + ".pdf"))
    plt.show()

#%%
import pickle as pkl
with open(join(dataroot, "insilico_results", "Evol_train_model_generalize.pkl"), "wb") as f:
    pkl.dump({"target_scores_evol_train": target_scores_evol_train, "pred_scores_evol_train": pred_scores_evol_train,
            "target_scores_evol_test": target_scores_evol_test, "pred_scores_evol_test": pred_scores_evol_test,
            "target_scores_gan": target_scores_gan, "pred_scores_gan": pred_scores_gan,
            "target_scores_reevol": target_scores_reevol, "pred_scores_reevol": pred_scores_reevol,
            "target_scores_natval": target_scores_natval, "pred_scores_natval": pred_scores_natval, }
    , f)
#%%
df_synopsis.loc["sp_rf", "Ridge"].plot(kind="barh", y="pearson", x="dataset", )
plt.tight_layout()
plt.show()
#%%
df_synopsis.plot(kind="barh", y="pearson", x="dataset", stacked=False,
                 color=["red", "blue", "green", "orange", "black"])
plt.tight_layout()
plt.show()
#%%
df_synopsis.groupby(level=[0,1]).unstack(level=[0,1]).plot(kind="barh", y="pearson", x="dataset",
                 color=["red", "blue", "green", "orange", "black"])
plt.tight_layout()
plt.show()
#%%
df_synops_col.groupby(["xtype", "regressor", "dataset"], sort=False)["R2"].mean()\
    .unstack(level=[0,1]).plot(kind="barh",)
plt.xlabel("R squared")
plt.xlim([-0.4, 1])
plt.tight_layout()
plt.show()
#%%
df_synopsis.groupby(level=[0,1]).groupby("dataset").mean().unstack(level=[0,1]).\
    plot(kind="barh", y="pearson", x="dataset",)
#%%
df_synopsis.groupby(by="dataset", level=[0,1]).plot(kind="barh", y="pearson", x="dataset", )