"""in silico modelling experiment
Using one model to regress the other. Compare
* RF estimate / weight visualization
* Prediction of held out response
* Prediction of out of sample response like ImageNet Validation
"""
import torch
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
import pickle as pkl
from featvis_lib import tsr_posneg_factorize, tsr_factorize
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
dataroot = r"E:\OneDrive - Harvard University\CNN_neural_regression"
#%% Plot the anatomical receptive field of the target scorer unit.
import os
import seaborn as sns
from dataset_utils import create_imagenet_valid_dataset, DataLoader, Subset
from neural_regress.regress_lib import calc_reduce_features_dataset,\
    evaluate_prediction, sweep_regressors, Ridge, Lasso
from grad_RF_estim import grad_RF_estimate, gradmap2RF_square, fit_2dgauss
from grad_RF_estim import grad_population_RF_estimate
from scipy.stats import pearsonr, spearmanr
def estimate_RF_for_fit_models(fit_models, targetlayer, Hmaps, ccfactor, savedir,
                               prefix="", reps=100, batch=1, show=True,):
    gradmap_dict = {}
    for (FeatReducer, regressor, ) in fit_models:
        print(f"Processing {FeatReducer} {regressor}")
        clf_ = fit_models[(FeatReducer, regressor, )].best_estimator_
        if FeatReducer == "srp":
            Wtsr = (clf_.coef_ @ srp.components_).reshape(1, 1024, 15, 15)
        elif FeatReducer == "pca":
            Wtsr = (clf_.coef_ @ pca.components_).reshape(1, 1024, 15, 15)
        elif FeatReducer == "spmask3":
            ccfactor_fit = clf_.coef_.reshape(-1, 1024, )
            Wtsr = np.einsum("FC,HWF->CHW", ccfactor_fit, Hmaps).reshape(1, 1024, 15, 15)
        elif FeatReducer == "featvec3":
            Hmap_fit = clf_.coef_.reshape([-1, 15, 15])
            Wtsr = np.einsum("CF,FHW->CHW", ccfactor, Hmap_fit).reshape(1, 1024, 15, 15)
        elif FeatReducer == "factor3":
            weight_fit = clf_.coef_
            Wtsr = np.einsum("CF,HWF,F->CHW", ccfactor, Hmaps, weight_fit).reshape(1, 1024, 15, 15)
        else:
            raise ValueError("FeatReducer (Xtfm) not recognized")
        Wtsr_th = torch.tensor(Wtsr).float().cuda()
        gradAmpmap = grad_population_RF_estimate(scorer.model, targetlayer, Wtsr_th,
                         input_size=(3, 227, 227), device="cuda", reps=reps, batch=batch,
                         label=f"{prefix}-{regressor}-{FeatReducer}", show=show, figdir=savedir,)
        fitdict = fit_2dgauss(gradAmpmap, f"{prefix}-{regressor}-{FeatReducer}", outdir=savedir, plot=True)
        gradmap_dict[(FeatReducer, regressor, )] = (gradAmpmap, fitdict)

    return gradmap_dict

def shorten_layername(name):
    return name.replace(".layer","L").replace(".Bottleneck", "Btn")


def IoU(mask1, mask2):
    return np.sum(np.logical_and(mask1, mask2)) / np.sum(np.logical_or(mask1, mask2))


def summarize_rf_cmp(gradAmpmap, fitdict, gradmap_dict, expstr, expdir=None):
    regressor_names = ["Ridge", "Lasso"]
    featredlist = ["spmask3", "featvec3", "factor3", "pca", "srp"]
    df_col = ["regressor", "featred", "cval", "pval", "cval_fit", "pval_fit", "iou"]
    df = []
    figh, axs = plt.subplots(len(regressor_names), len(featredlist), figsize=(15, 7.5))
    figh2, axs2 = plt.subplots(len(regressor_names), len(featredlist), figsize=(15, 7.5))
    figh3, axs3 = plt.subplots(len(regressor_names), len(featredlist), figsize=(15, 7.5))
    for (FeatReducer, regressor,) in gradmap_dict:
        gradAmpmap_model, fitdict_model = gradmap_dict[(FeatReducer, regressor,)]
        cval, pval = pearsonr(gradAmpmap.flatten(), gradAmpmap_model.flatten())
        cval_fit, pval_fit = pearsonr(fitdict["fitmap"].flatten(), fitdict_model["fitmap"].flatten())
        mask1 = fitdict["fitmap"] > 0.3 * fitdict["amplitude"] + fitdict['offset']
        mask2 = fitdict_model["fitmap"] > 0.3 * fitdict_model["amplitude"] + fitdict_model['offset']
        iou = IoU(mask1, mask2)
        titlestr = f"{regressor} {FeatReducer}\n Pearson corr raw {cval:.3f} fit {cval_fit:.3f}\nIOU {iou:.3f}"
        df.append(pd.DataFrame([[regressor, FeatReducer, cval, pval, cval_fit, pval_fit, iou]], columns=df_col))
        print(titlestr.replace("\n", " "))
        rowi = regressor_names.index(regressor)
        coli = featredlist.index(FeatReducer)
        plt.sca(axs[rowi, coli])
        plt.contour(fitdict["fitmap"], cmap="gray", label="Ground truth RF")
        plt.contour(fitdict_model["fitmap"], cmap="summer", label="Model RF")
        plt.axis("image")
        plt.gca().invert_yaxis()
        plt.title(titlestr)
        plt.sca(axs2[rowi, coli])
        plt.imshow(fitdict_model["fitmap"], )
        plt.title(titlestr)
        plt.sca(axs3[rowi, coli])
        plt.imshow(gradAmpmap_model, )
        plt.title(titlestr)
    figh.suptitle(f"{expstr} Regression RF recovery cmp", fontsize=16)
    figh.tight_layout()
    figh2.suptitle(f"{expstr} Regression RF recovery cmp", fontsize=16)
    figh2.tight_layout()
    figh3.suptitle(f"{expstr} Regression RF recovery cmp", fontsize=16)
    figh3.tight_layout()
    plt.show()
    df = pd.concat(df, axis=0)
    df.reset_index(drop=True, inplace=True)
    if expdir is not None:
        if not isinstance(expdir, list):
            expdir = [expdir]
        for dir in expdir:
            figh.savefig(join(dir, f"{expstr}-RegrRF-cmpall-contour.png"))
            figh2.savefig(join(dir, f"{expstr}-RegrRF-cmpall-fitmap.png"))
            figh3.savefig(join(dir, f"{expstr}-RegrRF-cmpall-gradmap.png"))
            df.to_csv(join(dir, f"{expstr}-RegrRF-cmp-statistics.csv"), )
    return df, figh, figh2, figh3
#%%
G = upconvGAN("fc6").cuda()
G.requires_grad_(False)
#%%
# Target neuron network
targetlayer = ".layer3.Bottleneck5"
targetunit = (10, 3, 3) #(5, 6, 6)
scorer = TorchScorer("resnet50")
scorer.select_unit(("resnet50", targetlayer, *targetunit), allow_grad=True)
regresslayer = ".layer3.Bottleneck5"
featnet, net = load_featnet("resnet50_linf8")
featFetcher = featureFetcher(featnet, input_size=(3, 227, 227),
                             device="cuda", print_module=False)
featFetcher.record(regresslayer,)
#%%
Xtfm_dir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
data = pkl.load(open(join(Xtfm_dir, f"{regresslayer}_regress_Xtransforms.pkl"),"rb"))
srp = data["srp"]
pca = data["pca"]
#%%
outdir = join(dataroot, r"insilico_final\resnet50_linf8-resnet50")
sumdir = join(outdir, "summary")
os.makedirs(outdir, exist_ok=True)
targetlayer = ".layer3.Bottleneck5"
targetunit = (10, 3, 3)
#%%
matplotlib.use("Agg") # 'module://backend_interagg'
#%%
from torch_utils import show_imgrid, save_imgrid
from grad_RF_estim import GAN_grad_RF_estimate
for (targetunit, targetlayer) in [
                                  # ((10, 9, 9), ".layer3.Bottleneck5"),
                                  # ((20, 9, 3), ".layer3.Bottleneck5"),
                                  # ((15, 3, 10), ".layer3.Bottleneck5"),
                                  # ((15, 3, 6), ".layer3.Bottleneck1"),
                                  # ((15, 10, 6), ".layer2.Bottleneck2"),
                                  # ((5, 6, 6), ".layer4.Bottleneck0"),
                                  # ((5, 3, 3), ".layer4.Bottleneck1"),
                                  # ((5, 3, 6), ".layer4.Bottleneck0"),
                                  # ((5, 6, 3), ".layer4.Bottleneck1"),
                                  # ((5, 2, 3), ".layer4.Bottleneck2"),
                                  # ((5, 6, 3), ".layer4.Bottleneck2"),
                                  # ((5, 10, 4), ".layer3.Bottleneck3"),
                                  # ((5, 3, 4), ".layer3.Bottleneck3"),
                                  # ((5, 14, 7), ".layer2.Bottleneck3"),
                                #
                                # ((25, 3, 3), ".layer3.Bottleneck5"),
                                # ((25, 3, 10), ".layer3.Bottleneck5"),
                                # ((25, 10, 3), ".layer3.Bottleneck5"),
                                # ((25, 10, 10), ".layer3.Bottleneck5"),
                                # ((25, 7, 7), ".layer3.Bottleneck5"),
                                # ((25, 3, 7), ".layer3.Bottleneck5"),
                                # ((25, 10, 7), ".layer3.Bottleneck5"),
                                # ((25, 7, 3), ".layer3.Bottleneck5"),
                                # ((25, 7, 10), ".layer3.Bottleneck5"),
                                # #
                                # ((25, 3, 3), ".layer3.Bottleneck3"),
                                # ((25, 3, 10), ".layer3.Bottleneck3"),
                                # ((25, 10, 3), ".layer3.Bottleneck3"),
                                # ((25, 10, 10), ".layer3.Bottleneck3"),
                                # ((25, 7, 7), ".layer3.Bottleneck3"),
                                # ((25, 3, 7), ".layer3.Bottleneck3"),
                                # ((25, 10, 7), ".layer3.Bottleneck3"),
                                # ((25, 7, 3), ".layer3.Bottleneck3"),
                                # ((25, 7, 10), ".layer3.Bottleneck3"),
                                #
                                ((25, 3, 3), ".layer3.Bottleneck1"),
                                ((25, 3, 10), ".layer3.Bottleneck1"),
                                ((25, 10, 3), ".layer3.Bottleneck1"),
                                ((25, 10, 10), ".layer3.Bottleneck1"),
                                ((25, 7, 7), ".layer3.Bottleneck1"),
                                ((25, 3, 7), ".layer3.Bottleneck1"),
                                ((25, 10, 7), ".layer3.Bottleneck1"),
                                ((25, 7, 3), ".layer3.Bottleneck1"),
                                ((25, 7, 10), ".layer3.Bottleneck1"),
                                # #
                                # ((15, 5, 5), ".layer2.Bottleneck3"),
                                # ((15, 5, 23), ".layer2.Bottleneck3"),
                                # ((15, 23, 5), ".layer2.Bottleneck3"),
                                # ((15, 23, 23), ".layer2.Bottleneck3"),
                                # ((15, 13, 13), ".layer2.Bottleneck3"),
                                # ((15, 5, 13), ".layer2.Bottleneck3"),
                                # ((15, 23, 13), ".layer2.Bottleneck3"),
                                # ((15, 13, 5), ".layer2.Bottleneck3"),
                                # ((15, 13, 23), ".layer2.Bottleneck3"),
                                # ## #
                                ((15, 5, 5), ".layer2.Bottleneck1"),
                                ((15, 5, 23), ".layer2.Bottleneck1"),
                                ((15, 23, 5), ".layer2.Bottleneck1"),
                                ((15, 23, 23), ".layer2.Bottleneck1"),
                                ((15, 13, 13), ".layer2.Bottleneck1"),
                                ((15, 5, 13), ".layer2.Bottleneck1"),
                                ((15, 23, 13), ".layer2.Bottleneck1"),
                                ((15, 13, 5), ".layer2.Bottleneck1"),
                                ((15, 13, 23), ".layer2.Bottleneck1"),
                                # #
                                ((15, 5, 5), ".layer2.Bottleneck2"),
                                ((15, 5, 23), ".layer2.Bottleneck2"),
                                ((15, 23, 5), ".layer2.Bottleneck2"),
                                ((15, 23, 23), ".layer2.Bottleneck2"),
                                ((15, 13, 13), ".layer2.Bottleneck2"),
                                ((15, 5, 13), ".layer2.Bottleneck2"),
                                ((15, 23, 13), ".layer2.Bottleneck2"),
                                ((15, 13, 5), ".layer2.Bottleneck2"),
                                ((15, 13, 23), ".layer2.Bottleneck2"),
                                #
                                # ((10, 2, 6), ".layer4.Bottleneck1"),
                                # ((10, 6, 6), ".layer4.Bottleneck1"),
                                # ((10, 6, 2), ".layer4.Bottleneck1"),
                                # ((10, 4, 4), ".layer4.Bottleneck1"),
                                # ((10, 2, 4), ".layer4.Bottleneck1"),
                                # ((10, 6, 4), ".layer4.Bottleneck1"),
                                # ((10, 4, 2), ".layer4.Bottleneck1"),
                                # ((10, 4, 6), ".layer4.Bottleneck1"),
                                # ((10, 2, 2), ".layer4.Bottleneck1"),
                                #
                                # ((10, 2, 6), ".layer4.Bottleneck0"),
                                # ((10, 6, 6), ".layer4.Bottleneck0"),
                                # ((10, 6, 2), ".layer4.Bottleneck0"),
                                # ((10, 4, 4), ".layer4.Bottleneck0"),
                                # ((10, 2, 4), ".layer4.Bottleneck0"),
                                # ((10, 6, 4), ".layer4.Bottleneck0"),
                                # ((10, 4, 2), ".layer4.Bottleneck0"),
                                # ((10, 4, 6), ".layer4.Bottleneck0"),
                                # ((10, 2, 2), ".layer4.Bottleneck0"),
                                # #
                                # ((10, 2, 6), ".layer4.Bottleneck2"),
                                # ((10, 6, 6), ".layer4.Bottleneck2"),
                                # ((10, 6, 2), ".layer4.Bottleneck2"),
                                # ((10, 4, 4), ".layer4.Bottleneck2"),
                                # ((10, 2, 4), ".layer4.Bottleneck2"),
                                # ((10, 6, 4), ".layer4.Bottleneck2"),
                                # ((10, 4, 2), ".layer4.Bottleneck2"),
                                # ((10, 4, 6), ".layer4.Bottleneck2"),
                                # ((10, 2, 2), ".layer4.Bottleneck2"),
                        ]:
    expstr = "%s-%d-%d-%d"%(shorten_layername(targetlayer), *targetunit)
    prefix = "%s-%d-%d-%d"%(shorten_layername(targetlayer), *targetunit) #"L3Btn5-10-3-3"
    expdir = join(outdir, expstr)
    os.makedirs(expdir, exist_ok=True)
    scorer = TorchScorer("resnet50")
    scorer.select_unit(("resnet50", targetlayer, *targetunit), allow_grad=True)
    #%% Use an evolution to train the model
    feattsr_all = []
    resp_all = []
    optimizer = CholeskyCMAES(4096, population_size=None, init_sigma=3.0)
    z_arr = np.zeros((1, 4096))  # optimizer.init_x
    pbar = tqdm(range(60))
    for i in pbar:
        imgs = G.visualize(torch.tensor(z_arr).float().cuda())
        resp = scorer.score(imgs, )
        z_arr_new = optimizer.step_simple(resp, z_arr)
        z_arr = z_arr_new

        with torch.no_grad():
            featnet(scorer.preprocess(imgs, input_scale=1.0))

        del imgs
        # print(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
        pbar.set_description(f"{i}: {resp.mean():.2f}+-{resp.std():.2f}")
        feattsr = featFetcher[regresslayer]
        feattsr_all.append(feattsr.cpu().numpy())
        resp_all.append(resp)

    imgs = G.visualize(torch.tensor(z_arr).float().cuda())
    save_imgrid(imgs[:5], join(expdir, f"{prefix}-proto-imgs.png"), nrow=5)
    resp_all = np.concatenate(resp_all, axis=0)
    feattsr_all = np.concatenate(feattsr_all, axis=0)
    np.save(join(expdir, f"{expstr}_evol_resp.npy"), resp_all)
    #%% Compute Correlation and Covariance tensor
    respnorm = (resp_all - resp_all.mean(axis=0, keepdims=True)) / resp_all.std(axis=0, keepdims=True)
    featnorm = (feattsr_all - feattsr_all.mean(axis=0, keepdims=True))
    covtsr = np.einsum("ijkl, i->jkl", featnorm, respnorm) / respnorm.shape[0]
    cctsr = covtsr / featnorm.std(axis=0, keepdims=False)
    cctsr[np.isnan(cctsr)] = 0
    # t value correspond to the pearson correlation
    Ttsr = cctsr / np.sqrt(1 - cctsr**2) * np.sqrt(resp_all.shape[0] - 2)
    del featnorm, respnorm, imgs
    #%%
    Hmat, Hmaps, ccfactor, Stat = tsr_posneg_factorize(covtsr,
                           bdr=0, Nfactor=3, do_save=True, figdir=expdir)

    #%% numpy feature tensor transforms
    Xtfms = {"spmask3": lambda tsr: np.einsum("BCHW,HWF->BFC", tsr, Hmaps).reshape(tsr.shape[0], -1),
             "featvec3": lambda tsr: np.einsum("BCHW,CF->BFHW", tsr, ccfactor).reshape(tsr.shape[0], -1),
             "factor3": lambda tsr: np.einsum("BCHW,CF,HWF->BF", tsr, ccfactor, Hmaps).reshape(tsr.shape[0], -1),
             "pca": lambda tsr: pca.transform(tsr.reshape(tsr.shape[0], -1)),
             "srp": lambda tsr: srp.transform(tsr.reshape(tsr.shape[0], -1)),}
    # "facttsr3": lambda tsr: np.einsum("BCHW,CF,HWF->B", tsr, ccfactor, Hmaps).reshape(tsr.shape[0], -1),  }
    #%%
    Xdict = {k: Xtfms[k](feattsr_all) for k in Xtfms}
    #%%
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=1.0)
    # poissreg = PoissonRegressor(alpha=1.0, max_iter=500)
    # kr_rbf = KernelRidge(alpha=1.0, kernel="rbf", gamma=None, )
    regressors = [ridge, lasso]
    regressor_names = ["Ridge", "Lasso"]
    result_df, fit_models = sweep_regressors(Xdict, resp_all, regressors, regressor_names, )
    #%%
    result_df.to_csv(join(expdir, "evol_regress_results.csv"))
    pkl.dump(fit_models, open(join(expdir, "evol_regress_models.pkl"), "wb"))
    #%%
    gradmap_dict = estimate_RF_for_fit_models(fit_models, regresslayer, Hmaps, ccfactor,
                                   expdir, prefix=prefix, )
    try:
        gradAmpmap = grad_RF_estimate(scorer.model, targetlayer, targetunit, input_size=(3,227,227),
                         device="cuda", show=True, reps=200, batch=1)
    except ValueError:
        gradAmpmap = GAN_grad_RF_estimate(G, scorer.model, targetlayer, targetunit, input_size=(3,227,227),
                         device="cuda", show=True, reps=20, batch=5)
    fitdict = fit_2dgauss(gradAmpmap, prefix, outdir=expdir, plot=True)
    # Xlim, Ylim = gradmap2RF_square(gradAmpmap, absthresh=None, relthresh=0.01, square=True))
    #%% Evaluate RF fit
    for (FeatReducer, regressor, ) in fit_models:
        gradAmpmap_model, fitdict_model = gradmap_dict[(FeatReducer, regressor, )]
        cval, pval = spearmanr(gradAmpmap.flatten(), gradAmpmap_model.flatten())
        cval_fit, pval_fit = pearsonr(fitdict["fitmap"].flatten(), fitdict_model["fitmap"].flatten())
        # cval_fit, pval_fit = pearsonr(fitdict["fitmap"].flatten(), gradAmpmap_model.flatten())
        print(f"{regressor} {FeatReducer}: {cval:.3f} {pval:.1e} fit {cval_fit:.3f} {pval_fit:.1e}")
    pkl.dump(gradmap_dict, open(join(expdir, "model_gradmap_rfdict.pkl"), "wb"))
    pkl.dump((gradAmpmap, fitdict), open(join(expdir, "groundtruth_gradmap_rfdict.pkl"), "wb"))
    df_rfstat, _, _, _ = summarize_rf_cmp(gradAmpmap, fitdict, gradmap_dict, expstr, expdir=[expdir, sumdir])

    #%% Get "ground truth" scores from ImageNet validation set
    dataset = create_imagenet_valid_dataset(imgpix=227, normalize=False)
    data_loader = DataLoader(Subset(dataset, range(0, 1000)), batch_size=40,
                  shuffle=False, num_workers=0)
    #%  Get "ground truth" scores from ImageNet validation set
    target_scores_natval = []
    for i, (imgs, _) in tqdm(enumerate(data_loader)):
        imgs = imgs.cuda()
        with torch.no_grad():
            score_batch = scorer.score(imgs)  # scorer.score(denormalizer(imgs))
        target_scores_natval.append(score_batch)

    target_scores_natval = np.concatenate(target_scores_natval, axis=0)
    #%%
    INdata = create_imagenet_valid_dataset(imgpix=227)
    natvalXfeat = calc_reduce_features_dataset(INdata, Xtfms, net, regresslayer, idx_range=range(0, 1000))
    #%%
    df_INet, eval_dict_INet, y_pred_INet = evaluate_prediction(fit_models, natvalXfeat, target_scores_natval,
                                                               label="-layer3-ImageNet", savedir=expdir)
    np.save(join(expdir, f"{expstr}_ImageNet_resp_gt.npy"), natvalXfeat)
    plt.close("all")
#%%

sumdir = join(outdir, "summary")
for (targetunit, targetlayer) in [((10, 3, 3), ".layer3.Bottleneck5"),
                                  ((10, 9, 9), ".layer3.Bottleneck5"),
                                  ((20, 9, 3), ".layer3.Bottleneck5"),
                                  ((15, 3, 10), ".layer3.Bottleneck5"),
                                  ((15, 3, 6), ".layer3.Bottleneck1"),
                                  ((15, 10, 6), ".layer2.Bottleneck2"),]:
    expstr = "%s-%d-%d-%d"%(shorten_layername(targetlayer), *targetunit)
    expdir = join(outdir, expstr)
    gradmap_dict = pkl.load(open(join(expdir, "model_gradmap_rfdict.pkl"), "rb"))
    (gradAmpmap, fitdict) = pkl.load(open(join(expdir, "groundtruth_gradmap_rfdict.pkl"), "rb"))
    print("\n", expstr)
    df_rfstat, _, _, _ = summarize_rf_cmp(gradAmpmap, fitdict, gradmap_dict, expstr, expdir=sumdir)
    # for (FeatReducer, regressor, ) in fit_models:
    #     gradAmpmap_model, fitdict_model = gradmap_dict[(FeatReducer, regressor, )]
    #     cval, pval = pearsonr(gradAmpmap.flatten(), gradAmpmap_model.flatten())
    #     cval_fit, pval_fit = pearsonr(fitdict["fitmap"].flatten(), fitdict_model["fitmap"].flatten())
    #     mask1 = fitdict["fitmap"] > 0.3*fitdict["amplitude"] + fitdict['offset']
    #     mask2 = fitdict_model["fitmap"] > 0.3*fitdict_model["amplitude"] + fitdict_model['offset']
    #     iou = IoU(mask1, mask2)
    #     # cval_fit, pval_fit = pearsonr(fitdict["fitmap"].flatten(), gradAmpmap_model.flatten())
    #     print(f"{regressor} {FeatReducer}: {cval:.3f} {pval:.1e} fit {cval_fit:.3f} {pval_fit:.1e} IOU {iou:.3f}")
#%%

#%%
figh, axs = plt.subplots(1, 1, figsize=(5, 5))
plt.subplot(1, 1, 1)
# plt.contour(mask1, cmap="gray", label="Ground truth RF")
# plt.contour(mask2, cmap="summer", label="Model RF")
plt.contour(fitdict["fitmap"], cmap="gray", label="Ground truth RF")
plt.contour(fitdict_model["fitmap"], cmap="summer", label="Model RF")
plt.suptitle(f"{expstr} {regressor} {FeatReducer}\n Pearson corr raw {cval:.3f} fit {cval_fit:.3f} IOU {iou:.3f}")
plt.axis("image")
plt.gca().invert_yaxis()
plt.savefig(join(expdir, f"RF_reconstruct_cmp-{FeatReducer}-{regressor}.png"))
plt.savefig(join(expdir, f"RF_reconstruct_cmp-{FeatReducer}-{regressor}.pdf"))
plt.show()
#%%
#%% Estimate RF of one model
clf_ = fit_models[('srp', 'Lasso')].best_estimator_
Wtsr = (clf_.coef_@srp.components_).reshape(1, 1024, 15, 15)
Wtsr_th = torch.tensor(Wtsr).float().cuda()
gradAmpmap_srp = grad_population_RF_estimate(scorer.model, ".layer3.Bottleneck5", Wtsr_th, input_size=(3,227,227),
                     device="cuda", show=True, reps=100, batch=1)
#%%
#%%
# titlestr = f"{Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}"
fig, axs = plt.subplots(1, 2, figsize=(8, 4.5))
sns.heatmap(Wtsr.mean(axis=0), ax=axs[r0], )
axs[0].axis("equal")
axs[0].set_title("mean weight")
sns.heatmap(np.abs(Wtsr).mean(axis=0), ax=axs[1], )
axs[1].axis("equal")
axs[1].set_title("mean abs weight")
# plt.suptitle(titlestr)
# plt.tight_layout()
plt.show()
