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
import pickle as pkl
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
            Here we assume the input to each transformer is a numpy array (not torch tensor)
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


from torch.utils.data import Subset, SubsetRandomSampler
def calc_reduce_features_dataset(dataset, feat_transformers, net, featlayer,
                  batch_size=40, workers=6, img_dim=(227, 227), idx_range=None):
    """Calculate reduced features for a set of images. (for memory saving)

    :param dataset: Image Dataset
    :param feattsr_reducer: a dict of functions that reduce a feature tensor to a vector
            Here we assume the input to each transformer is a numpy array (not torch tensor)
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
    if idx_range is None:
        imgloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    else:
        imgloader = DataLoader(Subset(dataset, idx_range), batch_size=batch_size,
                               shuffle=False, num_workers=workers,)
    # imgdata = dataset
    # imgloader = DataLoader(imgdata, batch_size=batch_size, shuffle=False, num_workers=workers)

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
    # print("score vector shape", score_vect.shape)
    del featFetcher
    return feattsr_col


def sweep_regressors(Xdict, y_all, regressors, regressor_names,):
    """
    Sweep through a list of regressors (with cross validation), and input type (Xdict)
    For each combination of Xtype and regressor, return the best CVed regressor
        and its parameters, in a model_dict and a dataframe
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


def evaluate_prediction(fit_models, Xfeat_dict, y_true, label="", savedir=None):
    """
    Evaluate the prediction of a dict of models
    :param fit_models: dict of models
    :param Xfeat_dict: dict of feature tensors, input to the models
    :param y_true: true y values
    :param label: label for saving, saved to the dataframe.
    :param savedir: directory to save the results. If None, no saving.
        saved `df` `eval_dict` `y_pred_dict`
    :return:
        df: dataframe summarize evaluation statistics of each model
        eval_dict: same thing in a dict
        y_pred_dict: dict of predictions vectors, used to do other evaluation.

    Ported from ManifExp_regression_fit
    """
    print(label, f"  N imgs: {len(y_true)}")
    eval_dict = {}
    y_pred_dict = {}
    for (Xtype, regrname), regr in fit_models.items():
        try:
            y_pred = regr.predict(Xfeat_dict[Xtype])
            if isinstance(regr, PLSRegression):
                y_pred = y_pred[:, 0]
            D2 = regr.score(Xfeat_dict[Xtype], y_true)
            rho_p, pval_p = pearsonr(y_pred, y_true)
            rho_s, pval_s = spearmanr(y_pred, y_true)
            print(f"{Xtype} {regrname} Prediction Pearson: {rho_p:.3f} {pval_p:.1e} Spearman: {rho_s:.3f} {pval_s:.1e} D2: {D2:.3f}")
            eval_dict[(Xtype, regrname)] = {"rho_p":rho_p, "pval_p":pval_p, "rho_s":rho_s, "pval_s":pval_s, "D2":D2, "imgN":len(y_true)}
            y_pred_dict[(Xtype, regrname)] = y_pred
        except:
            continue
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


def merge_dict_arrays(*dict_arrays):
    """
    Merge a list of dicts into a single dict.
    """
    # skip the empty dicts, use the dict with the most keys
    keys = []
    for d in dict_arrays:
        if len(d) != 0:
            keys = [*d.keys()]
            break
    # fill the empty dicts with the keys paired with empty arrays (this dataset has no images in this )
    for d in dict_arrays:
        if len(d) == 0:
            for k in keys:
                d[k] = np.array([])

    # TODO: solve variable array length problem. Every value in each array should be the same length.
    merged_dict = {}
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
