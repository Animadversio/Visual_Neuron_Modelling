"""
Factorized regression by alternative least square with corrfeat factor initialization.
Still testing.
"""
import matplotlib.pyplot as plt
from featvis_lib import CorrFeatScore, tsr_posneg_factorize, rectify_tsr, pad_factor_prod
from CorrFeatTsr_predict_lib import fitnl_predscore, score_images, loadimg_preprocess, predict_fit_dataset
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import johnson_lindenstrauss_min_dim, \
            SparseRandomProjection, GaussianRandomProjection
from sklearn.linear_model import LogisticRegression, LinearRegression, \
    Ridge, Lasso, PoissonRegressor, RidgeCV, LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
#%%
import os
from os.path import join
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
from tqdm import tqdm
from featvis_lib import load_featnet
from layer_hook_utils import featureFetcher
from dataset_utils import ImagePathDataset, DataLoader
from CorrFeatTsr_predict_lib import score_images_torchdata

def calc_features(score_vect, imgfullpath_vect, net, featlayer,
                  batch_size=80, workers=6, img_dim=(224, 224)):
    """
    Calculate features for a set of images.
    :param score_vect: numpy vector of scores
    :param imgfullpath_vect: a list full path to images
    :param net: net to extract features from
    :param featlayer: layer to extract features from
    :param batch_size: batch size for DataLoader
    :param workers: number of workers for DataLoader
    :param img_dim: image dimensions
    :return:
    """
    imgdata = ImagePathDataset(imgfullpath_vect, score_vect, img_dim=img_dim)
    imgloader = DataLoader(imgdata, batch_size=batch_size, shuffle=False, num_workers=workers)

    featFetcher = featureFetcher(net, print_module=False)
    featFetcher.record(featlayer, )
    feattsr_col = []
    for i, (imgtsr, score) in tqdm(enumerate(imgloader)):
        with torch.no_grad():
            net(imgtsr.cuda())
        feattsr = featFetcher[featlayer]
        feattsr_col.append(feattsr.cpu().numpy())

    feattsr_all = np.concatenate(feattsr_col, axis=0)
    print("feature tensor shape", feattsr_all.shape)
    print("score vector shape", score_vect.shape)
    del feattsr_col, featFetcher
    return feattsr_all


def calc_features_times_spatialmask(score_vect, imgfullpath_vect, net, featlayer,
                  spatialmask, batch_size=80, workers=6, img_dim=(224, 224)):
    """
    spatialmask: nChannels x H x W
    """
    imgdata = ImagePathDataset(imgfullpath_vect, score_vect, img_dim=img_dim)
    imgloader = DataLoader(imgdata, batch_size=batch_size, shuffle=False, num_workers=workers)

    featFetcher = featureFetcher(net, print_module=False)
    featFetcher.record(featlayer, )
    feattsr_col = []
    for i, (imgtsr, score) in tqdm(enumerate(imgloader)):
        with torch.no_grad():
            net(imgtsr.cuda())
        feattsr = featFetcher[featlayer]
        feattsr_prod = (feattsr.unsqueeze(-1) * spatialmask).sum(2, 3)
        feattsr_col.append(feattsr_prod.cpu().numpy())

    feattsr_all = np.concatenate(feattsr_col, axis=0)
    print("feature tensor shape", feattsr_all.shape)
    print("score vector shape", score_vect.shape)
    del feattsr_col, featFetcher
    return feattsr_all

#%%
from scipy.io import loadmat
from data_loader import load_score_mat
featnet, net = load_featnet("resnet50_linf8")

saveroot = r"E:\OneDrive - Harvard University\CNN_neural_regression"
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
Animal, Expi = "Alfa", 19
featlayer = ".layer3.Bottleneck5"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
result_df = pd.read_csv(join(saveroot, fr"resnet50_linf8\{Animal}_Exp{Expi:02}_resnet50_linf8_{featlayer}_regression_results.csv"))
score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi,
                                "Evol", wdws=[(50, 200)], stimdrive="N")
#%%
layer = "layer3"
netname = "resnet50_linf8"
#%% Loading or compute the factorization.
NF = 3; rect_mode = "Tthresh"; thresh = (None, 3)
init = "nndsvda"; solver="cd"; l1_ratio=0; alpha=0; beta_loss="frobenius"; bdr = 1
explabel = f"{Animal}_Exp{Expi:02}_resnet50_linf8_{layer}_corrfeat_rect{rect_mode}_init{init}_solver{solver}_l1{l1_ratio}_alpha{alpha}_beta{beta_loss}"
figdir = r"E:\OneDrive - Harvard University\CNN_neural_regression\resnet50_linf8\CorrFeatTsr_predict"
data = np.load(join(fr"S:\corrFeatTsr\{Animal}_Exp{Expi:02}_Evol_nobdr_res-robust_corrTsr.npz"), allow_pickle=True)
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
Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr),
         bdr=bdr, Nfactor=NF, init=init, solver=solver, l1_ratio=l1_ratio, alpha=alpha, beta_loss=beta_loss,
         figdir=figdir, savestr="%s-%scov" % (netname, layer), suptit=explabel, show=True,)

#%%
scorer = CorrFeatScore()
scorer.register_hooks(net, layer, netname="resnet50_linf8")
rank1_Wtsr = [pad_factor_prod(Hmaps[:, :, i:i+1], ccfactor[:, i:i+1], bdr=bdr) for i in range(3)]
stack_Wtsr = np.stack(rank1_Wtsr, axis=0)
scorer.register_weights({layer: stack_Wtsr})
pred_score = score_images_torchdata(featnet, scorer, layer,
                imgfullpath_vect, batchsize=80, workers=6)
scorer.clear_hook()
#%% Iterative methods for factorized regression
#%%
padded_mask = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
sp_masks = torch.tensor(padded_mask).float().cuda()
feat_spprod = calc_features_times_spatialmask(score_vect, imgfullpath_vect, net, featlayer,
                  sp_masks, batch_size=80, workers=6, img_dim=(224, 224))
#%%
feattsr_all = calc_features(score_vect, imgfullpath_vect, net, featlayer,
                  batch_size=80, workers=6, img_dim=(224, 224))  # (N, C, H, W)
#%%
feat_featprod = np.einsum("bchw,cn->bhwn", feattsr_all, ccfactor)  # (N, H, W, NF)
#%%
train_idx, test_idx = train_test_split(np.arange(len(score_vect)), test_size=0.2, random_state=42)
#%%
Nimg, C, H, W = feattsr_all.shape
NF = ccfactor.shape[1]
#%%
feat_spprod = np.einsum("bchw,hwn->bcn", feattsr_all, padded_mask)  # (N, C, NF)
Xall = feat_spprod.reshape(feat_spprod.shape[0], -1)
clf = RidgeCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4,  1E5])
clf.fit(Xall[train_idx, :], score_vect[train_idx])
D2_train = clf.score(Xall[train_idx, :], score_vect[train_idx])
D2_test = clf.score(Xall[test_idx, :], score_vect[test_idx])
print("Feature updates alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f"%(clf.alpha_, D2_train, D2_test))
ccfactor_prime = clf.coef_.reshape(C, NF)
#%%
feat_featprod = np.einsum("bchw,cn->bhwn", feattsr_all, ccfactor_prime)  # (N, H, W, NF)
Xall = feat_featprod.reshape(feat_featprod.shape[0], -1)
clf2 = RidgeCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4,  1E5])
clf2.fit(Xall[train_idx, :], score_vect[train_idx])
D2_train2 = clf2.score(Xall[train_idx, :], score_vect[train_idx])
D2_test2 = clf2.score(Xall[test_idx, :], score_vect[test_idx])
print("alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f"%(clf2.alpha_, D2_train2, D2_test2))
#%%
Hmaps_prime = clf2.coef_.reshape(H, W, NF)
#%%
import matplotlib.pylab as plt
plt.imshow(Hmaps_prime / Hmaps_prime.max())
plt.show()

#%% Step functions for alternative least square.
def update_ccfactor(feattsr_all, score_vect, ccfactor, padded_Hmaps, train_idx, test_idx):
    Nimg, C, H, W = feattsr_all.shape
    NF = ccfactor.shape[1]
    feat_spprod = np.einsum("bchw,hwn->bcn", feattsr_all, padded_Hmaps)  # (N, C, NF)
    Xall = feat_spprod.reshape(feat_spprod.shape[0], -1)
    # clf = RidgeCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, 1E5, 1E6])
    clf = LassoCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, 1E5, ])
    clf.fit(Xall[train_idx, :], score_vect[train_idx])
    D2_train = clf.score(Xall[train_idx, :], score_vect[train_idx])
    D2_test = clf.score(Xall[test_idx, :], score_vect[test_idx])
    print("Feature updates alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f" % (clf.alpha_, D2_train, D2_test))
    ccfactor_prime = clf.coef_.reshape(C, NF)
    return ccfactor_prime, clf, D2_train, D2_test


def update_Hmaps(feattsr_all, score_vect, ccfactor, padded_Hmaps, train_idx, test_idx):
    Nimg, C, H, W = feattsr_all.shape
    NF = ccfactor.shape[1]
    feat_featprod = np.einsum("bchw,cn->bhwn", feattsr_all, ccfactor)  # (N, H, W, NF)
    Xall = feat_featprod.reshape(feat_featprod.shape[0], -1)
    clf2 = RidgeCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, 1E5, 1E6])
    clf2.fit(Xall[train_idx, :], score_vect[train_idx])
    D2_train2 = clf2.score(Xall[train_idx, :], score_vect[train_idx])
    D2_test2 = clf2.score(Xall[test_idx, :], score_vect[test_idx])
    print("Hmap updates, alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f" % (clf2.alpha_, D2_train2, D2_test2))
    Hmaps_prime = clf2.coef_.reshape(H, W, NF)
    return Hmaps_prime, clf, D2_train, D2_test


def update_Hmap_ccfact(feattsr_all, score_vect, ccfactor, padded_Hmaps, train_idx, test_idx):
    Nimg, C, H, W = feattsr_all.shape
    NF = ccfactor.shape[1]

    feat_spprod = np.einsum("bchw,hwn->bcn", feattsr_all, padded_Hmaps)  # (N, C, NF)
    Xall = feat_spprod.reshape(feat_spprod.shape[0], -1)
    clf = RidgeCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, 1E5, 1E6])
    clf.fit(Xall[train_idx, :], score_vect[train_idx])
    D2_train = clf.score(Xall[train_idx, :], score_vect[train_idx])
    D2_test = clf.score(Xall[test_idx, :], score_vect[test_idx])
    print("Feature updates alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f" % (clf.alpha_, D2_train, D2_test))
    ccfactor_prime = clf.coef_.reshape(C, NF)

    feat_featprod = np.einsum("bchw,cn->bhwn", feattsr_all, ccfactor_prime)  # (N, H, W, NF)
    Xall = feat_featprod.reshape(feat_featprod.shape[0], -1)
    clf2 = RidgeCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, 1E5, 1E6])
    clf2.fit(Xall[train_idx, :], score_vect[train_idx])
    D2_train2 = clf2.score(Xall[train_idx, :], score_vect[train_idx])
    D2_test2 = clf2.score(Xall[test_idx, :], score_vect[test_idx])
    print("Hmap updates, alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f" % (clf2.alpha_, D2_train2, D2_test2))
    Hmaps_prime = clf2.coef_.reshape(H, W, NF)
    return Hmaps_prime, ccfactor_prime, clf, clf2, D2_train, D2_test, D2_train2, D2_test2


def eval_factors(feattsr_all, score_vect, ccfactor, padded_Hmaps, train_idx, test_idx):
    linpred_score = np.einsum("bchw,hwN,cN->b", feattsr_all, padded_Hmaps, ccfactor)
    clf_orig = LinearRegression().fit(linpred_score[train_idx,np.newaxis], score_vect[train_idx])
    D2_train = clf_orig.score(linpred_score[train_idx, np.newaxis], score_vect[train_idx])
    D2_test = clf_orig.score(linpred_score[test_idx, np.newaxis], score_vect[test_idx])
    print("Original weights Evaluation: Training set D2 %.3f\t Test set D2 %.3f" % (D2_train, D2_test))
    return clf_orig, D2_train, D2_test

#%%
padded_Hmaps = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
Hmaps_prime, ccfactor_prime, clf, clf2, D2_train, D2_test, D2_train2, D2_test2 = \
    update_Hmap_ccfact(feattsr_all, score_vect, ccfactor, padded_Hmaps, train_idx, test_idx)
#%%
ccfactor, padded_Hmaps = ccfactor_prime, Hmaps_prime
Hmaps_prime, ccfactor_prime, clf, clf2, D2_train, D2_test, D2_train2, D2_test2 = \
    update_Hmap_ccfact(feattsr_all, score_vect, ccfactor, padded_Hmaps, train_idx, test_idx)
#%%
ccfactor_prime, clf, D2_train, D2_test = update_ccfactor(feattsr_all, score_vect, ccfactor, padded_Hmaps, train_idx, test_idx)
Hmaps_prime, clf, D2_train, D2_test = update_Hmaps(feattsr_all, score_vect, ccfactor, padded_Hmaps, train_idx, test_idx)
#%%
padded_Hmaps = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
clf_orig, D2_train, D2_test = eval_factors(feattsr_all, score_vect, ccfactor, padded_Hmaps, train_idx, test_idx)
for i in range(4):
    Hmaps_prime, ccfactor_prime, clf, clf2, D2_train, D2_test, D2_train2, D2_test2 = \
        update_Hmap_ccfact(feattsr_all, score_vect, ccfactor, padded_Hmaps, train_idx, test_idx)
    ccfactor, padded_Hmaps = ccfactor_prime, Hmaps_prime

#%%

