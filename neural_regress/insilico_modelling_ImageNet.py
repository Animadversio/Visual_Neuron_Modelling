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
from sklearn.decomposition import IncrementalPCA
from sklearn.random_projection import SparseRandomProjection
from neural_regress.insilico_modelling_lib import compare_activation_prediction, sweep_regressors, \
    resizer, normalizer, denormalizer, PoissonRegressor, RidgeCV, Ridge, KernelRidge
#%% Prepare CNN, GAN players
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)

scorer = TorchScorer("resnet50")
scorer.select_unit(("resnet50", ".layer3.Bottleneck5", 5, 6, 6), allow_grad=False)

regresslayer = ".layer3.Bottleneck5"
featnet, net = load_featnet("resnet50_linf8")
featFetcher = featureFetcher(featnet, input_size=(3, 227, 227),
                             device="cuda", print_module=False)
featFetcher.record(regresslayer,)
#%%
Xfeat_transformer = {'pca': lambda tsr: pca.transform(tsr.reshape(tsr.shape[0], -1)),
                     "srp": lambda tsr: srp.transform(tsr.reshape(tsr.shape[0], -1)),
                     "sp_rf": lambda tsr: tsr[:, :, 6, 6],
                     "sp_avg": lambda tsr: tsr.mean(axis=(2, 3))}
#%%
Xfeat_transformer = {"sp_rf": lambda tsr: tsr[:, :, 6, 6].copy(),
                     "sp_avg": lambda tsr: tsr.mean(axis=(2, 3))}
#%%
from torch_utils import show_imgrid
from NN_PC_visualize.NN_PC_lib import \
    create_imagenet_valid_dataset, Dataset, DataLoader
dataset = create_imagenet_valid_dataset(imgpix=227, normalize=True)
data_loader = DataLoader(dataset, batch_size=100,
              shuffle=False, num_workers=8)
#%%
# ipca = IncrementalPCA(n_components=1000, batch_size=100)
# srp = SparseRandomProjection(n_components=1000, ).fit(np.random.rand(10, 1024 * 15 * 15))
target_scores_natval = []
Xfeat_dict = defaultdict(list)
for i, (imgs, _) in tqdm(enumerate(data_loader)):
    imgs = imgs.cuda()
    with torch.no_grad():
        score_batch = scorer.score(imgs, skip_preprocess=True)

    target_scores_natval.append(score_batch)
    with torch.no_grad():
        featnet(imgs)
        feattsr = featFetcher[regresslayer]
        feattsr = feattsr.cpu().numpy()

    Xfeat_dict["sp_rf"].append(feattsr[:, :, 6, 6].copy())
    Xfeat_dict["sp_avg"].append(feattsr.mean(axis=(2, 3)))
    # Xfeat_dict["srp"].append(srp.transform(feattsr.reshape(feattsr.shape[0], -1)))
    # ipca.partial_fit(feattsr.reshape(feattsr.shape[0], -1))
    if i >= 150:
        break
#%%
target_scores_natval = np.concatenate(target_scores_natval, axis=0)
#%%
import pickle as pkl
with open(join("E:\\ImageNet_Features.pkl"), "wb") as f:
    pkl.dump(Xfeat_dict, f)
#%%
y_all = target_scores_natval
ridge = Ridge(alpha=1.0)
poissreg = PoissonRegressor(alpha=1.0, max_iter=500)
kr_rbf = KernelRidge(alpha=1.0, kernel="rbf", gamma=None, )
regressors = [ridge, poissreg, kr_rbf, ]
regressor_names = ["Ridge", "Poisson", "KernelRBF"]
result_df, fit_models = sweep_regressors(Xfeat_dict, y_all, regressors, regressor_names,)
#%%
pred_scores_natval = defaultdict(list)
for k in fit_models:
    featmat_tfm = Xfeat_dict[k[0]] #Xfeat_transformer[k[0]](feattsr)
    pred_score = fit_models[k].predict(featmat_tfm)
    pred_scores_natval[k] = pred_score

#%%
idx_train, idx_test = train_test_split(
        range(len(y_all)), test_size=0.2, random_state=42, shuffle=True
    )
target_scores_test = y_all[idx_test]
target_scores_train = y_all[idx_train]
pred_scores_test = {k: v[idx_test] for k, v in pred_scores_natval.items()}
pred_scores_train = {k: v[idx_train] for k, v in pred_scores_natval.items()}
#%%
savedir = r'E:\OneDrive - Harvard University\CNN_neural_regression\insilico_results\ImageNet_train'
compare_activation_prediction(target_scores_train, pred_scores_train, "ImageNet_train", savedir=savedir)
compare_activation_prediction(target_scores_test, pred_scores_test, "ImageNet_test", savedir=savedir)
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
model_list = [('sp_avg', "Poisson"), ('sp_rf', "KernelRBF"),
              ('sp_rf', "Ridge"), ('sp_rf', "Poisson")]
target_scores_gan = []
pred_scores_gan = defaultdict(list)
for i in tqdm(range(100)):
    imgs = G.visualize(2 * torch.randn(40, 4096).cuda())
    with torch.no_grad():
        score_batch = scorer.score(imgs)
    target_scores_gan.append(score_batch)

    with torch.no_grad():
        featnet(resizer(normalizer(imgs)))
        feattsr = featFetcher[regresslayer]
        feattsr = feattsr.cpu().numpy()

    for k in model_list:
        featmat_tfm = Xfeat_transformer[k[0]](feattsr)
        pred_score = fit_models[k].predict(featmat_tfm)
        pred_scores_gan[k].append(pred_score)

for k in pred_scores_gan:
    pred_scores_gan[k] = np.concatenate(pred_scores_gan[k], axis=0)

target_scores_gan = np.concatenate(target_scores_gan, axis=0)
#%%
compare_activation_prediction(target_scores_gan, pred_scores_gan,
                              "GAN-random", savedir=savedir)

#%%
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

for k in pred_scores_reevol:
    pred_scores_reevol[k] = np.concatenate(pred_scores_reevol[k], axis=0)
target_scores_reevol = np.concatenate(target_scores_reevol, axis=0)

#%%
compare_activation_prediction(target_scores_reevol, pred_scores_reevol,
                              "Reevolution-SameUnit", savedir=savedir)
#%% BigGAN evolution
from GAN_utils import BigGAN_wrapper, loadBigGAN
BGAN = loadBigGAN().cpu().eval()
BG = BigGAN_wrapper(BGAN)
#%%
target_scores_BigGANevol = []
pred_scores_BigGANevol = defaultdict(list)

optimizer = CholeskyCMAES(256, population_size=None, init_sigma=0.07)
z_arr = np.zeros((1, 256))  # optimizer.init_x
for i in tqdm(range(150)):
    imgs = BG.visualize(torch.tensor(z_arr).float())#.cuda()
    with torch.no_grad():
        score_batch = scorer.score(imgs.cuda())
    target_scores_BigGANevol.append(score_batch)
    z_arr = optimizer.step_simple(score_batch, z_arr)

    with torch.no_grad():
        featnet(resizer(normalizer(imgs.cuda())))
        feattsr = featFetcher[regresslayer]
        feattsr = feattsr.cpu().numpy()

    for k in model_list:
        featmat_tfm = Xfeat_transformer[k[0]](feattsr)
        pred_score = fit_models[k].predict(featmat_tfm)
        pred_scores_BigGANevol[k].append(pred_score)

for k in pred_scores_BigGANevol:
    pred_scores_BigGANevol[k] = np.concatenate(pred_scores_BigGANevol[k], axis=0)

target_scores_BigGANevol = np.concatenate(target_scores_BigGANevol, axis=0)
#%%
compare_activation_prediction(target_scores_BigGANevol, pred_scores_BigGANevol,
                              "BigGANevol-SameUnit", savedir=savedir)
#%%
df_ImageNet_train = compare_activation_prediction(target_scores_train, pred_scores_train, "ImageNet_train", savedir=savedir)
df_ImageNet_test  = compare_activation_prediction(target_scores_test, pred_scores_test, "ImageNet_test", savedir=savedir)
df_ganrand = compare_activation_prediction(target_scores_gan, pred_scores_gan,
                              "GAN-random", savedir=savedir)
df_reevol = compare_activation_prediction(target_scores_reevol, pred_scores_reevol,
                              "Reevolution-SameUnit", savedir=savedir)
df_BigGANevol = compare_activation_prediction(target_scores_BigGANevol, pred_scores_BigGANevol,
                              "BigGANevol-SameUnit", savedir=savedir)
#%%
df_synopsis = pd.concat([df_ImageNet_train, df_ImageNet_test,
                         df_ganrand, df_reevol, df_BigGANevol], axis=0)
df_synopsis.to_csv(join(savedir, "synopsis.csv"))
#%%
df_synopsis = df_synopsis.astype({'spearman': 'float64', 'pearson': 'float64',
                    "spearman_pval": 'float64', "pearson_pval": 'float64', "R2": 'float64',
                    "dataset": str, "n_sample": int})
df_synops_col = df_synopsis.reset_index()
df_synops_col.rename(columns={"level_0": "xtype", "level_1": "regressor"}, inplace=True)
#%%
for stat in ["pearson", "spearman", "R2"]:
    df_synops_col.groupby(["xtype", "regressor", "dataset"], sort=False)[stat].mean()\
        .unstack(level=[0, 1]).plot(kind="barh",)
    plt.xlabel(stat)
    plt.tight_layout()
    if stat == "R2": plt.xlim([-0.3, 0.3])
    plt.savefig(join(savedir, "model_generalization_synopsis_" + stat + ".png"))
    plt.savefig(join(savedir, "model_generalization_synopsis_" + stat + ".pdf"))
    plt.show()
