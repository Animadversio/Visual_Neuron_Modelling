"""
 * load in linear weights and the linear projection weights
 * port the linear weights to a torch model.
 * visualize the model by back propagation.
 * Sumarize the images.
"""

#%% Newer version pipeline
import os

import matplotlib.pyplot as plt
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
import pickle as pkl
from os.path import join
import numpy as np
import torch
from neural_regress.sklearn_torchify_lib import SRP_torch, PCA_torch, \
    LinearRegression_torch, PLS_torch, SpatialAvg_torch
saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
#%%
from GAN_utils import upconvGAN
from layer_hook_utils import featureFetcher
import torch.nn.functional as F
from torch.optim import Adam
from torch_utils import show_imgrid, save_imgrid
#%%
G = upconvGAN()
G.eval().cuda()
G.requires_grad_(False)
featnet, net = load_featnet("resnet50_linf8")
net.eval().cuda()
RGBmean = torch.tensor([0.485, 0.456, 0.406]).cuda().reshape(1, 3, 1, 1)
RGBstd = torch.tensor([0.229, 0.224, 0.225]).cuda().reshape(1, 3, 1, 1)

#%%
# featlayer = ".layer3.Bottleneck5"
# featlayer = ".layer4.Bottleneck2"
featlayer = ".layer2.Bottleneck3"
Xtfm_fn = f"{featlayer}_regress_Xtransforms.pkl"
data = pkl.load(open(join(saveroot, Xtfm_fn), "rb"))
#%%
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        expdir = join(saveroot, f"{Animal}_{Expi:02d}")
        regr_data = pkl.load(open(join(expdir, f"{featlayer}_regress_models.pkl"), "rb"))
        featvis_dir = join(expdir, "featvis")
        os.makedirs(featvis_dir, exist_ok=True)
        #%%
        regr_cfgs = [*regr_data.keys()]
        for regrlabel in regr_cfgs:
            Xtype, regressor = regrlabel
            print(f"Processing {Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}")
            clf = regr_data[regrlabel]
            if Xtype == 'pca':
                Xtfm_th = PCA_torch(data['pca'], device="cpu")
            elif Xtype == 'srp':
                Xtfm_th = SRP_torch(data['srp'], device="cpu")
            elif Xtype == 'sp_avg':
                Xtfm_th = SpatialAvg_torch()
            if isinstance(clf, GridSearchCV):
                clf_th = LinearRegression_torch(clf, device="cpu")
            elif isinstance(clf, PLSRegression):
                clf_th = PLS_torch(clf, device="cpu")
            else:
                raise ValueError("Unknown regressor type")
            #%% Visualize the model
            B = 5
            fetcher = featureFetcher(net, input_size=(3, 227, 227))
            fetcher.record(featlayer, ingraph=True)
            zs = torch.randn(B, 4096).cuda()
            zs.requires_grad_(True)
            optimizer = Adam([zs], lr=0.1)
            for i in range(100):
                optimizer.zero_grad()
                imgtsrs = G.visualize(zs)
                imgtsrs_rsz = F.interpolate(imgtsrs, size=(227, 227), mode='bilinear', align_corners=False)
                imgtsrs_rsz = (imgtsrs_rsz - RGBmean) / RGBstd
                featnet(imgtsrs_rsz)
                activations = fetcher[featlayer]
                featmat = Xtfm_th(activations.cpu()) # .flatten(start_dim=1)
                scores = clf_th(featmat)
                loss = - scores.sum()
                loss.backward()
                optimizer.step()
                print(f"{i}, {scores.sum().item():.3f}")

            # show_imgrid(imgtsrs)
            save_imgrid(imgtsrs, join(featvis_dir,
                      f"{Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}_vis.png"))
#%%
"""Summarize the results into a montage acrsoo methods"""
from build_montages import crop_from_montage, make_grid_np
# featlayer = ".layer3.Bottleneck5"
featlayer = ".layer4.Bottleneck2"

for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        expdir = join(saveroot, f"{Animal}_{Expi:02d}")
        featvis_dir = join(expdir, "featvis")
        proto_col = []
        for regrlabel in regr_cfgs:
            Xtype, regressor = regrlabel
            mtg = plt.imread(join(featvis_dir, f"{Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}_vis.png"))
            proto_first = crop_from_montage(mtg, (0, 0))
            proto_col.append(proto_first)
        method_mtg = make_grid_np(proto_col, nrow=3)
        plt.imsave(join(featvis_dir, f"{Animal}-Exp{Expi:02d}-{featlayer}-regr_merge_vis.png"), method_mtg, )

#%%
import matplotlib.pyplot as plt
#%%
# regr_cfgs = [ ('srp', 'Ridge'),
#  ('srp', 'Lasso'),
#  ('srp', 'PLS'),
#              ('sp_avg', 'Ridge'),
#              ('sp_avg', 'Lasso'),
#              ('sp_avg', 'PLS'),
#              ('pca', 'Ridge'),
#              ('pca', 'Lasso'),
#              ('pca', 'PLS'),
# ]
#%%
if isinstance(clf, GridSearchCV):
    """ Predict
    return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_
    """
    weights = clf.best_estimator_.coef_
    intercept = clf.best_estimator_.intercept_
else:
    """ Predict
    X -= self.x_mean_
    X /= self.x_std_
    Ypred = np.dot(X, self.coef_)
    return Ypred + self.y_mean_
    """
    weights = clf.coef_
#%%
""" PCA transform
X = X - self.mean_
X_transformed = np.dot(X, self.components_.T)
"""
Xtfmmat = torch.from_numpy(data["pca"].components_)
#%%
""" SRP transform
X_new = safe_sparse_dot(X, self.components_.T, dense_output=self.dense_output)
return X_new
"""
# https://gist.github.com/aesuli/319d71707a5ee96086aa2439b87d4e38
matcoo = data["srp"].components_.tocoo()
print('Mat srp', matcoo)
Xtfmmat = torch.sparse.LongTensor(
    torch.LongTensor([matcoo.row.tolist(), matcoo.col.tolist()]),
    torch.LongTensor(matcoo.data.astype(np.int32)))

#%% Define torch based modules to back propagate, moved to sklearn_torchify_lib
from sklearn.linear_model._base import LinearModel
from typing import List, Union
class SRP_torch(torch.nn.Module):
    def __init__(self, srp: SparseRandomProjection):
        super(SRP_torch, self).__init__()
        matcoo = srp.components_.tocoo()
        self.components = torch.sparse.FloatTensor(
            torch.LongTensor([matcoo.row.tolist(), matcoo.col.tolist()]),
            torch.FloatTensor(matcoo.data.astype(np.float32)))

    def forward(self, X):
        return torch.sparse.mm(self.components, X.T).T


class PCA_torch(torch.nn.Module):
    def __init__(self, pca: PCA):
        super(PCA_torch, self).__init__()
        self.n_features = pca.n_features_in_
        self.n_components = pca.n_components
        self.mean = torch.from_numpy(pca.mean_)  # (n_features,)
        self.components = torch.from_numpy(pca.components_)  # (n_components, n_features)

    def forward(self, X):
        X = X - self.mean
        return torch.mm(X, self.components.T)


class LinearRegression_torch(torch.nn.Module):
    def __init__(self, linear_regression: Union[LinearModel, GridSearchCV]):
        super(LinearRegression_torch, self).__init__()
        if isinstance(linear_regression, GridSearchCV):
            assert isinstance(linear_regression.estimator, LinearModel)
            self.coef = torch.from_numpy(linear_regression.best_estimator_.coef_)
            self.intercept = torch.tensor(linear_regression.best_estimator_.intercept_)
        else:
            self.coef = torch.from_numpy(linear_regression.coef_)
            self.intercept = torch.tensor(linear_regression.intercept_)
        if self.coef.ndim == 1:
            self.coef = self.coef.unsqueeze(1)

    def forward(self, X):
        return torch.mm(X, self.coef) + self.intercept


class PLS_torch(torch.nn.Module):
    def __init__(self, pls: PLSRegression):
        super(PLS_torch, self).__init__()
        self.n_components = pls.n_components
        self.n_features = pls.n_features_in_
        self.coef = torch.from_numpy(pls.coef_)  # (n_components, n_features)
        self.x_mean = torch.from_numpy(pls.x_mean_)  # (n_features,)
        self.x_std = torch.from_numpy(pls.x_std_)  # (n_features,)
        self.y_mean = torch.from_numpy(pls.y_mean_)  # (n_targets,)

    def forward(self, X):
        X = X - self.x_mean
        X = X / self.x_std
        Ypred = torch.mm(X, self.coef)
        return Ypred + self.y_mean

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
