from sklearn.random_projection import johnson_lindenstrauss_min_dim, \
    SparseRandomProjection, GaussianRandomProjection
from sklearn.linear_model import LogisticRegression, LinearRegression, \
    Ridge, Lasso, PoissonRegressor, RidgeCV, LassoCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from scipy.stats import spearmanr, pearsonr
from torchvision.transforms import ToPILImage, ToTensor, Normalize, Resize
from torch_utils import show_imgrid
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from os.path import join
from collections import defaultdict
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# from NN_PC_visualize.NN_PC_lib import \
#     create_imagenet_valid_dataset, Dataset, DataLoader

denormalizer = Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                         std=[1/0.229, 1/0.224, 1/0.225])
normalizer = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
resizer = Resize(227, )
import torch
from tqdm import tqdm
from layer_hook_utils import featureFetcher
from torch.utils.data import Dataset, DataLoader
from dataset_utils import ImagePathDataset, DataLoader
from copy import deepcopy

def calc_features(score_vect, imgfullpath_vect, net, featlayer,
                  batch_size=40, workers=6, img_dim=(227, 227)):
    """
    Calculate features for a set of images.
    :param score_vect: numpy vector of scores,
            if None, then it's default to be zeros. ImagePathDataset will handle None scores.
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


def calc_reduce_features(score_vect, imgfullpath_vect, feat_transformers, net, featlayer,
                  batch_size=40, workers=6, img_dim=(227, 227)):
    """Calculate reduced features for a set of images. (for memory saving)

    :param score_vect: numpy vector of scores,
            if None, then it's default to be zeros. ImagePathDataset will handle None scores.
    :param imgfullpath_vect: a list full path to images
    :param feattsr_reducer: a dict of functions that reduce a feature tensor to a vector
        Examples:
                    {"none": lambda x: x, }
        Examples:
            Xfeat_transformer = {'pca': lambda tsr: pca.transform(tsr.reshape(tsr.shape[0], -1)),
                     "srp": lambda tsr: srp.transform(tsr.reshape(tsr.shape[0], -1)),
                     "sp_rf": lambda tsr: tsr[:, :, 6, 6],
                     "sp_avg": lambda tsr: tsr.mean(axis=(2, 3))}
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
    feattsr_col = defaultdict(list)
    for i, (imgtsr, score) in tqdm(enumerate(imgloader)):
        with torch.no_grad():
            net(imgtsr.cuda())
        feattsr = featFetcher[featlayer]
        for tfmname, feat_transform in feat_transformers.items():
            feattsr_col[tfmname].append(feat_transform(feattsr.cpu().numpy()))
        # feattsr_col.append(feattsr.cpu().numpy())
    for tfmname in feattsr_col:
        feattsr_col[tfmname] = np.concatenate(feattsr_col[tfmname], axis=0)
        print(tfmname, "feature tensor shape", feattsr_col[tfmname].shape)
    # feattsr_all = np.concatenate(feattsr_col, axis=0)
    print("score vector shape", score_vect.shape)
    del featFetcher
    return feattsr_col


def sweep_regressors(Xdict, y_all, regressors, regressor_names,):
    """
    Sweep through a list of regressors (with cross validation), and input type (Xdict)
     return the best regressor and its parameters
    :param Xdict:
    :param y_all:
    :param regressors:
    :param regressor_names:
    :return:
        result_df: dataframe with the results of the regression
        models: dict of regressor objects
    """
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
            if hasattr(estim, "alpha"):
                clf = GridSearchCV(estimator=estim, n_jobs=8,
                                   param_grid=dict(alpha=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1E4, 1E5], ),
                                   ).fit(X_train, y_train)
                alpha = clf.best_params_["alpha"]
            else:
                clf = estim.fit(X_train, y_train)
                clf = deepcopy(clf)
                alpha = np.nan
            D2_train = clf.score(X_train, y_train)
            D2_test = clf.score(X_test, y_test)
            result_summary[(xtype, label)] = \
                {"alpha": alpha, "train_score": D2_train, "test_score": D2_test, "n_feat": nfeat}
            models[(xtype, label)] = clf

        result_df = pd.DataFrame(result_summary)

    print(result_df.T)
    return result_df.T, models


def compare_activation_prediction(target_scores_natval, pred_scores_natval, exptitle="", savedir=""):
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
        plt.savefig(join(savedir, f"{exptitle}_{k}_regress.png"))
        plt.show()

    test_result_df = pd.DataFrame(result_col)
    test_result_df.T.to_csv(join(savedir, f"{exptitle}_regress_results.csv"))
    return test_result_df.T
