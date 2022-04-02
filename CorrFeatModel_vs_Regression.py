import os
from os.path import join
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
from tqdm import tqdm
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
Animal = "Alfa"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']

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
def calc_features(score_vect, imgfullpath_vect, net, featlayer, batch_size=40, workers=6, img_dim=(227, 227)):
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

#%%
saveroot = r"E:\OneDrive - Harvard University\CNN_neural_regression"
os.makedirs(join(saveroot, "resnet50_linf8"), exist_ok=True)
featnet, net = load_featnet("resnet50_linf8")
#%%
for Expi in range(19, 47):# 47):
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





#%% Corr Feature Tsr
# 157K
from featvis_lib import CorrFeatScore, tsr_posneg_factorize, rectify_tsr, pad_factor_prod
from CorrFeatTsr_predict_lib import fitnl_predscore, score_images, loadimg_preprocess, predict_fit_dataset
# DR_Wtsr = np.load(join(saveroot, fr"resnet50_linf8\{Animal}_Exp{Expi:02}_resnet50_linf8_{layer}_DR_Wtsr.npy"))
#%%
layer = "layer3"
netname = "resnet50_linf8"
scorer = CorrFeatScore()
scorer.register_hooks(net, layer, netname="resnet50_linf8")
#%%
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
pcr = make_pipeline(PCA(n_components=200), PoissonRegressor(alpha=0.1))
pcr.fit(X_train, y_train)
pcr.score(X_test, y_test)






#%% Dev zone
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4,  1E5]).fit(X_train, y_train)
D2_train = clf.score(X_train, y_train)
D2_test = clf.score(X_test, y_test)
print("alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f"%(clf.alpha_, D2_train, D2_test))
result_summary["Ridge"] = {"alpha": clf.alpha_, "train_score": D2_train, "test_score": D2_test}


best_test = -1E3
alpha_, D2_train_, D2_test_ = None, None, None
for alpha in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4,  1E5]:
    clf = PoissonRegressor(alpha=alpha).fit(X_train, y_train)
    D2_train = clf.score(X_train, y_train)
    D2_test = clf.score(X_test, y_test)
    print("alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f"%(alpha, D2_train, D2_test))
    if D2_test > best_test:
        best_test = D2_test
        alpha_, D2_train_, D2_test_ = alpha, D2_train, D2_test
result_summary["Poisson"] = {"alpha": alpha_, "train_score": D2_train_, "test_score": D2_test_}

#%%
krmodel = KernelRidge(alpha=1, kernel="rbf", gamma=None, )
clf = GridSearchCV(estimator=krmodel, n_jobs=-1,
                   param_grid=dict(alpha=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, 1E5],),
                   ).fit(X_train, y_train)
D2_test = clf.score(X_test, y_test)
print(clf.best_params_["alpha"], clf.best_score_, D2_test)
#%%
imgdata = ImagePathDataset(imgfullpath_vect, score_vect, img_dim=(227, 227))
imgloader = DataLoader(imgdata, batch_size=40, shuffle=False, num_workers=6)
#%%
featlayer = ".layer2.Bottleneck3"  # ".layer4.Bottleneck2"  # ".layer3.Bottleneck5"
featFetcher = featureFetcher(net)
featFetcher.record(featlayer, )
#%%
feattsr_col = []
for i, (imgtsr, score) in tqdm(enumerate(imgloader)):
    # print(imgtsr.shape, score.shape)
    with torch.no_grad():
        featnet(imgtsr.cuda())
    feattsr = featFetcher[featlayer]
    feattsr_col.append(feattsr.cpu().numpy())

feattsr = np.concatenate(feattsr_col, axis=0)
print("feature tensor shape", feattsr.shape)
print("score vector shape", score_vect.shape)
del feattsr_col
#%%
featmat = feattsr.reshape(feattsr.shape[0], -1)
SRP = SparseRandomProjection().fit(featmat)
srp_featmat = SRP.transform(featmat)
featmat_avg = feattsr.mean(axis=(2, 3))
score_spkcount = np.round(score_vect * 0.15)
#%%
y_all = score_spkcount  # score_vect
X_all = srp_featmat # featmat_avg #   # featmat_avg  # srp_featmat  # featmat
y_train, y_test, X_train, X_test = train_test_split(
    y_all, X_all, test_size=0.2, random_state=42, shuffle=True
)
#%%
clf = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, ]).fit(X_train, y_train)
D2_train = clf.score(X_train, y_train)
D2_test = clf.score(X_test, y_test)
print("alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f"%(clf.alpha_, D2_train, D2_test))
#%%
clf = LassoCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, ]).fit(X_train, y_train)
D2_train = clf.score(X_train, y_train)
D2_test = clf.score(X_test, y_test)
print("alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f"%(clf.alpha_, D2_train, D2_test))
#%%
for alpha in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, ]:
    clf = PoissonRegressor(alpha=alpha).fit(X_train, y_train)
    D2_train = clf.score(X_train, y_train)
    D2_test = clf.score(X_test, y_test)
    print("alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f"%(alpha, D2_train, D2_test))

#%%
for alpha in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4,]:
    clf = KernelRidge(alpha=alpha, kernel='rbf', gamma=None, ).fit(X_train, y_train)
    D2_train = clf.score(X_train, y_train)
    D2_test = clf.score(X_test, y_test)
    print("alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f"%(alpha, D2_train, D2_test))
#%%
for alpha in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4,]:
    clf = KernelRidge(alpha=alpha, kernel='laplacian', gamma=None, ).fit(X_train, y_train)
    D2_train = clf.score(X_train, y_train)
    D2_test = clf.score(X_test, y_test)
    print("alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f"%(alpha, D2_train, D2_test))
#%%
for alpha in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4,]:
    clf = KernelRidge(alpha=alpha, kernel="poly", gamma=None, ).fit(X_train, y_train)
    D2_train = clf.score(X_train, y_train)
    D2_test = clf.score(X_test, y_test)
    print("alpha %.1e: Training set D2 %.3f\t Test set D2 %.3f"%(alpha, D2_train, D2_test))

#%%
score_pasu, imgfullpath_pasu = load_score_mat(EStats, MStats, Expi,
                                            "Pasu_avg", wdws=[(50, 200)], stimdrive="N")
#%%
clf = PLSRegression(n_components=500).fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
#%%