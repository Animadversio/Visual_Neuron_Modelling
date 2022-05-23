import os
from os.path import join
import torch
import torchvision.models as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import matplotlib.pylab as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from featvis_lib import load_featnet
from dataset_utils import ImagePathDataset, DataLoader
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
Pasupath = r"N:\Stimuli\2019-Manifold\pasupathy-wg-f-4-ori"
Gaborpath = r"N:\Stimuli\2019-Manifold\gabor"
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
print(matplotlib.get_backend())
matplotlib.use('Agg')
# matplotlib.use('module://backend_interagg')
#%%
from neural_regress.regress_lib import calc_features, \
        calc_reduce_features, sweep_regressors, evaluate_prediction, \
        merge_dict_arrays, merge_arrays, evaluate_dict
from scipy.stats import pearsonr, spearmanr
featlayer = '.layer3.Bottleneck5'
data = pkl.load(open(join(saveroot, f"{featlayer}_regress_Xtransforms.pkl"), "rb"))
srp = data["srp"]
pca = data["pca"]
#%%
import seaborn as sns
from featvis_lib import tsr_posneg_factorize, rectify_tsr
from neural_regress.sklearn_torchify_lib import SRP_torch, PCA_torch, \
    LinearRegression_torch, PLS_torch, SpatialAvg_torch
saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
outdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress\summary\weight_vis"
tsrshapes = {'.layer2.Bottleneck3':(512, 29, 29),
            '.layer3.Bottleneck5':(1024, 15, 15),
            '.layer4.Bottleneck2':(2048, 8, 8)}
#%%
plot_each_method = False
for featlayer in ['.layer2.Bottleneck3', '.layer3.Bottleneck5', '.layer4.Bottleneck2']:
    data = pkl.load(open(join(saveroot, f"{featlayer}_regress_Xtransforms.pkl"), "rb"))
    srp = data["srp"]
    pca = data["pca"]
    for Animal in ["Alfa", "Beto"]:
        for Expi in range(1, 47):
            if Animal == "Beto" and Expi == 46: continue
            expdir = join(saveroot, f"{Animal}_{Expi:02d}")
            os.makedirs(join(expdir, "weight_vis"), exist_ok=True)
            regr_data = pkl.load(open(join(expdir, f"{featlayer}_regress_models.pkl"), "rb"))
            regr_cfgs = [*regr_data.keys()]
            eff_wtsr_dict = {}
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
                #%%
                dummyX = torch.ones((1, *tsrshapes[featlayer]), requires_grad=True)
                dummyY = clf_th(Xtfm_th(dummyX))
                dummyY.backward()
                effective_weights = dummyX.grad.data
                eff_wtsr = effective_weights.reshape(tsrshapes[featlayer])
                eff_wtsr_dict[regrlabel] = eff_wtsr
                #%%
                if plot_each_method:
                    titlestr = f"{Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}"
                    fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
                    sns.heatmap(eff_wtsr.mean(dim=0), ax=axs[0], )
                    axs[0].axis("equal")
                    axs[0].set_title("mean weight")
                    sns.heatmap(eff_wtsr.abs().mean(dim=0), ax=axs[1], )
                    axs[1].axis("equal")
                    axs[1].set_title("mean abs weight")
                    plt.suptitle(titlestr)
                    # plt.tight_layout()
                    plt.show()

            # figh1, axs1 = plt.subplots(2, 3, figsize=(11, 8))
            # figh2, axs2 = plt.subplots(2, 3, figsize=(11, 8))
            # for i, FeatRed in enumerate(["pca", "srp"]):
            #     for j, regressor in enumerate(["Ridge", "Lasso", "PLS"]):
            #         eff_wtsr = eff_wtsr_dict[(FeatRed, regressor)]
            #         axs1[i, j].imshow(eff_wtsr.mean(dim=0))
            #         axs1[i, j].set_title(f"{FeatRed}-{regressor}", fontsize=12)
            #         axs1[i, j].axis("image")
            #         axs2[i, j].imshow(eff_wtsr.abs().mean(dim=0))
            #         axs2[i, j].set_title(f"{FeatRed}-{regressor}", fontsize=12)
            #         axs2[i, j].axis("image")
            # figh1.suptitle(f"{Animal}-Exp{Expi:02d}-{featlayer} Mean Weight", fontsize=14)
            # figh2.suptitle(f"{Animal}-Exp{Expi:02d}-{featlayer} Mean Abs Weight", fontsize=14)
            # figh1.tight_layout()
            # figh2.tight_layout()
            # figh1.savefig(join(outdir, f"{Animal}-Exp{Expi:02d}-{featlayer}_mean_weight.png"))
            # figh1.savefig(join(outdir, f"{Animal}-Exp{Expi:02d}-{featlayer}_mean_weight.pdf"))
            # figh2.savefig(join(outdir, f"{Animal}-Exp{Expi:02d}-{featlayer}_meanabs_weight.png"))
            # figh2.savefig(join(outdir, f"{Animal}-Exp{Expi:02d}-{featlayer}_meanabs_weight.pdf"))
            # figh1.show()
            # figh2.show()
            # plt.close(figh1)
            # plt.close(figh2)
            FactStat_dict = {}
            for i, FeatRed in enumerate(["pca", "srp"]):
                for j, regressor in enumerate(["Ridge", "Lasso", "PLS"]):
                    Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(
                        eff_wtsr_dict[(FeatRed, regressor)].numpy(),
                        bdr=0, Nfactor=3, do_plot=True,
                        figdir=join(expdir, "weight_vis"),
                        savestr=f"{Animal}-Exp{Expi:02d}-{featlayer}-{FeatRed}-{regressor}_regress",)
                    FactStat_dict[(FeatRed, regressor)] = FactStat
            pkl.dump(FactStat_dict, open(join(expdir, "weight_vis", f"{Animal}-Exp{Expi:02d}-{featlayer}_regress_factstat.pkl"), "wb"))
            # raise ValueError("Done")
            plt.close("all")

# sns.FacetGrid(data=eff_wtsr_dict, row="regressor", col="FeatRed", size=4).map(plt.imshow, "mean_abs_weight").add_legend()
#%%
Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(
    eff_wtsr_dict[('pca', 'Ridge')].numpy(),
       bdr=0, Nfactor=3, do_plot=True)
#%%
Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(\
    rectify_tsr(eff_wtsr_dict[('pca', 'Ridge')].numpy(), mode="abs", thr=(0,0), Ttsr=None),
                                           bdr=1, Nfactor=3, do_plot=True)
