"""
Create torch version of sklearn models in order to pass gradient and feature visualization.
Used in `ManifExp_regression_featvisualize.py`
"""
from sklearn.linear_model import LogisticRegression, LinearRegression, \
    Ridge, Lasso, PoissonRegressor, RidgeCV, LassoCV
from sklearn.random_projection import johnson_lindenstrauss_min_dim, \
            SparseRandomProjection, GaussianRandomProjection
from sklearn.linear_model._base import LinearModel
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import torch
import numpy as np
from typing import List, Union


class SRP_torch(torch.nn.Module):
    def __init__(self, srp: SparseRandomProjection, device="cpu"):
        super(SRP_torch, self).__init__()
        matcoo = srp.components_.tocoo()
        self.components = torch.sparse.FloatTensor(
            torch.LongTensor([matcoo.row.tolist(), matcoo.col.tolist()]),
            torch.FloatTensor(matcoo.data.astype(np.float32))).to(device)

    def forward(self, X):
        if X.ndim > 2:
            X = X.flatten(start_dim=1)
        return torch.sparse.mm(self.components, X.T).T


class PCA_torch(torch.nn.Module):
    def __init__(self, pca: PCA, device="cpu"):
        super(PCA_torch, self).__init__()
        self.n_features = pca.n_features_in_
        self.n_components = pca.n_components
        self.mean = torch.from_numpy(pca.mean_).to(device)  # (n_features,)
        self.components = torch.from_numpy(pca.components_).to(device)  # (n_components, n_features)

    def forward(self, X):
        if X.ndim > 2:
            X = X.flatten(start_dim=1)
        X = X - self.mean
        return torch.mm(X, self.components.T)


class LinearRegression_torch(torch.nn.Module):
    def __init__(self, linear_regression: Union[LinearModel, GridSearchCV], device="cpu"):
        super(LinearRegression_torch, self).__init__()
        if isinstance(linear_regression, GridSearchCV):
            assert isinstance(linear_regression.estimator, LinearModel)
            self.coef = torch.from_numpy(linear_regression.best_estimator_.coef_).float().to(device)
            self.intercept = torch.tensor(linear_regression.best_estimator_.intercept_).float().to(device)
        else:
            self.coef = torch.from_numpy(linear_regression.coef_).float().to(device)
            self.intercept = torch.tensor(linear_regression.intercept_).float().to(device)
        if self.coef.ndim == 1:
            self.coef = self.coef.unsqueeze(1)

    def forward(self, X):
        return torch.mm(X, self.coef) + self.intercept


class PLS_torch(torch.nn.Module):
    def __init__(self, pls: PLSRegression, device="cpu"):
        super(PLS_torch, self).__init__()
        self.n_components = pls.n_components
        self.n_features = pls.n_features_in_
        self.coef = torch.from_numpy(pls.coef_).to(device)  # (n_components, n_features)
        self.x_mean = torch.from_numpy(pls.x_mean_).to(device)  # (n_features,)
        self.x_std = torch.from_numpy(pls.x_std_).to(device)  # (n_features,)
        self.y_mean = torch.from_numpy(pls.y_mean_).to(device)  # (n_targets,)

    def forward(self, X):
        X = X - self.x_mean
        X = X / self.x_std
        Ypred = torch.mm(X, self.coef)
        return Ypred + self.y_mean


class SpatialAvg_torch(torch.nn.Module):
    def __init__(self, device="cpu"):
        super(SpatialAvg_torch, self).__init__()

    def forward(self, X):
        if X.ndim == 4:
            return X.mean(dim=(2, 3))
        elif X.ndim == 3:
            return X.mean(dim=2)
        else:
            return X


if __name__ == "__main__":
    import pickle as pkl
    from os.path import join
    saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
    featlayer = ".layer3.Bottleneck5"
    Xtfm_fn = f"{featlayer}_regress_Xtransforms.pkl"
    data = pkl.load(open(join(saveroot, Xtfm_fn), "rb"))
    Animal, Expi = "Alfa", 3
    expdir = join(saveroot, f"{Animal}_{Expi:02d}")
    regr_data = pkl.load(open(join(expdir, f"{featlayer}_regress_models.pkl"), "rb"))
    #%% Test torch modules have same results as sklearn
    X = np.random.rand(10, 1024)
    pls = regr_data[('sp_avg', 'PLS')]
    y_pred = pls.predict(X)
    pls_th = PLS_torch(pls)
    y_pred_th = pls_th(torch.from_numpy(X))
    assert torch.allclose(torch.from_numpy(y_pred), y_pred_th)
    #%%
    X = np.random.rand(10, 1024).astype(np.float32)
    reg = regr_data[('sp_avg', 'Lasso')]
    y_pred = reg.predict(X)
    reg_th = LinearRegression_torch(reg)
    y_pred_th = reg_th(torch.from_numpy(X))
    assert torch.allclose(torch.from_numpy(y_pred[:, np.newaxis]), y_pred_th)
    #%%
    X = np.random.rand(10, 1024).astype(np.float32)
    reg = regr_data[('sp_avg', 'Ridge')]
    y_pred = reg.predict(X)
    reg_th = LinearRegression_torch(reg)
    y_pred_th = reg_th(torch.from_numpy(X))
    assert torch.allclose(torch.from_numpy(y_pred[:, np.newaxis]), y_pred_th)
    #%% PCA
    X = np.random.rand(10, 230400).astype(np.float32)
    pca = data["pca"]
    pca_th = PCA_torch(pca)
    X_red = pca.transform(X)
    X_red_th = pca_th(torch.from_numpy(X))
    assert torch.allclose(torch.from_numpy(X_red), X_red_th)
    #%% SRP
    X = np.random.rand(10, 230400).astype(np.float32)
    srp = data["srp"]
    srp_th = SRP_torch(srp)
    X_red = srp.transform(X)
    X_red_th = srp_th(torch.from_numpy(X))
    assert torch.allclose(torch.from_numpy(X_red).float(), X_red_th, atol=1E-5)