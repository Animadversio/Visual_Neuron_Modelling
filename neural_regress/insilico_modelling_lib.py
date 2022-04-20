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
