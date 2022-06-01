"""
Visualize RF for factor regression models.
"""
import pickle as pkl
from os.path import join
expdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress\Beto_26"
modelfile = r'.layer3.Bottleneck5_factor-regress_models.pkl'
data = pkl.load(open(join(expdir, modelfile), "rb"))
#%%
from featvis_lib import load_featnet
featnet, net = load_featnet("resnet50_linf8")
#%%
import numpy as np
from featvis_lib import CorrFeatScore, tsr_posneg_factorize, rectify_tsr, pad_factor_prod
def load_covtsrs(Animal, Expi, layer, ):
    data = np.load(join(fr"S:\corrFeatTsr\{Animal}_Exp{Expi:d}_Evol_nobdr_res-robust_corrTsr.npz"), allow_pickle=True)
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
    return covtsr, Ttsr, cctsr


def load_NMF_factors(Animal, Expi, layer, NF=3):
    # This is from resnet50_linf8 (res-robust)
    rect_mode = "Tthresh"; thresh = (None, 3); bdr = 1
    explabel = f"{Animal}_Exp{Expi:d}_resnet50_linf8_{layer}_corrfeat_rect{rect_mode}"
    covtsr, Ttsr, cctsr = load_covtsrs(Animal, Expi, layer)
    Hmat, Hmaps, ccfactor, FactStat = tsr_posneg_factorize(rectify_tsr(covtsr, rect_mode, thresh, Ttsr=Ttsr),
                                                           bdr=bdr, Nfactor=NF, do_plot=False)
    padded_Hmaps = np.pad(Hmaps[:, :, :], ((bdr, bdr), (bdr, bdr), (0, 0)), mode="constant")
    return padded_Hmaps, Hmaps, ccfactor, FactStat

#%%
import torch

from grad_RF_estim import grad_RF_estimate, gradmap2RF_square, fit_2dgauss
from grad_RF_estim import grad_population_RF_estimate
def estimate_RF_for_fit_models(net, fit_models, targetlayer, Hmaps, ccfactor, savedir,
                               prefix="", reps=100, batch=1, show=True,
                               input_size=(3, 227, 227)):
    space_dim = Hmaps.shape[:2]
    chan_dim = ccfactor.shape[0]
    gradmap_dict = {}
    for (FeatReducer, regressor, ) in fit_models:
        print(f"Processing {FeatReducer} {regressor}")
        clf_ = fit_models[(FeatReducer, regressor, )].best_estimator_
        if FeatReducer == "srp":
            Wtsr = (clf_.coef_ @ srp.components_).reshape(1, chan_dim, *space_dim)
        elif FeatReducer == "pca":
            Wtsr = (clf_.coef_ @ pca.components_).reshape(1, chan_dim, *space_dim)
        elif FeatReducer == "spmask3":
            ccfactor_fit = clf_.coef_.reshape(-1, chan_dim, )
            Wtsr = np.einsum("FC,HWF->CHW", ccfactor_fit, Hmaps).reshape(1, chan_dim, *space_dim)
        elif FeatReducer == "featvec3":
            Hmap_fit = clf_.coef_.reshape([-1, *space_dim])
            Wtsr = np.einsum("CF,FHW->CHW", ccfactor, Hmap_fit).reshape(1, chan_dim, *space_dim)
        elif FeatReducer == "factor3":
            weight_fit = clf_.coef_
            Wtsr = np.einsum("CF,HWF,F->CHW", ccfactor, Hmaps, weight_fit).reshape(1, chan_dim, *space_dim)
        else:
            raise ValueError("FeatReducer (Xtfm) not recognized")
        Wtsr_th = torch.tensor(Wtsr).float().cuda()
        gradAmpmap = grad_population_RF_estimate(net, targetlayer, Wtsr_th,
                                                 input_size=input_size, device="cuda", reps=reps, batch=batch,
                                                 label=f"{prefix}-{regressor}-{FeatReducer}", show=show, figdir=savedir,)
        fitdict = fit_2dgauss(gradAmpmap, f"{prefix}-{regressor}-{FeatReducer}", outdir=savedir, plot=True)
        gradmap_dict[(FeatReducer, regressor, )] = (gradAmpmap, fitdict)

    return gradmap_dict

outdir = r"E:\OneDrive - Harvard University\CCN2022_shortpaper\figures\FeatureAttrib"
regresslayer = ".layer3.Bottleneck5"
Animal, Expi = "Beto", 26
#%%

padded_Hmaps, Hmaps, ccfactor, FactStat = load_NMF_factors(Animal, Expi, regresslayer.split('.')[1])
#%%
import matplotlib.pyplot as plt
plt.imshow(padded_Hmaps)
plt.show()
#%%
gradmap_dict = estimate_RF_for_fit_models(net, data, regresslayer, padded_Hmaps, ccfactor,
                               outdir, prefix=f"{Animal}-Exp{Expi}", input_size=(3, 224, 224))

#%%
featlayer = '.layer3.Bottleneck5'
y_pred_manif = pkl.load(open(join(expdir, "eval_predvec_factreg-.layer3.Bottleneck5-Manif.pkl"),"rb"))
y_pred_gabor = pkl.load(open(join(expdir, "eval_predvec_factreg-.layer3.Bottleneck5-Gabor.pkl"),"rb"))
y_pred_pasu = pkl.load(open(join(expdir, "eval_predvec_factreg-.layer3.Bottleneck5-Pasu.pkl"),"rb"))
y_pred_natref = pkl.load(open(join(expdir, "eval_predvec_factreg-.layer3.Bottleneck5-EvolRef.pkl"),"rb"))
#%%
from scipy.io import loadmat
from data_loader import load_score_mat
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
Animal="Beto"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = \
loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)[
    'EStats']
score_manif, imgfullpath_manif = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)],stimdrive="N")
score_pasu, imgfullpath_pasu = load_score_mat(EStats, MStats, Expi, "Pasu_avg", wdws=[(50, 200)], stimdrive="N")
score_gabor, imgfullpath_gabor = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)],stimdrive="N")
score_natref, imgfullpath_natref = load_score_mat(EStats, MStats, Expi, "EvolRef_avg", wdws=[(50, 200)],stimdrive="N")
#%%
Xtype, regrname = 'spmask3', 'Ridge'
# SN = df_natmerge.loc[Xtype, regrname]
# SA = df_merge.loc[Xtype, regrname]
# statstr_nat = f"no Manif: rho_p {SN.rho_p:.3f} R2 {SN.D2:.3f} imgN {SN.imgN}"
# statstr_all = f"all: rho_p {SA.rho_p:.3f} R2 {SA.D2:.3f} imgN {SA.imgN}"
y_manif = y_pred_manif[(Xtype, regrname)]
y_gabor = y_pred_gabor[(Xtype, regrname)] if len(score_gabor) > 0 else []
y_pasu = y_pred_pasu[(Xtype, regrname)] if len(score_pasu) > 0 else []
y_natref = y_pred_natref[(Xtype, regrname)] if len(score_natref) > 0 else []
plt.figure(figsize=[6, 5.5])
plt.gca().axline((100, 100), slope=1, ls=":", color="k")
plt.scatter(y_manif, score_manif, label="manif", alpha=0.3)
plt.scatter(y_gabor, score_gabor, label="gabor", alpha=0.3)
plt.scatter(y_pasu, score_pasu, label="pasu", alpha=0.3)
plt.scatter(y_natref, score_natref, label="natref", alpha=0.3)
# plt.title(f"{Animal}-Exp{Expi:02d} {Xtype}-{regrname}\n{statstr_nat}\n{statstr_all}")
plt.title(f"{Animal}-Exp{Expi:02d} {Xtype}-{regrname}")
plt.xlabel("Predicted Spike Rate")
plt.ylabel("Observed Spike Rate")
plt.legend()
plt.savefig(join(outdir, f"{Xtype}_{regrname}_{featlayer}_factreg_pred_performance.png"))
plt.savefig(join(outdir, f"{Xtype}_{regrname}_{featlayer}_factreg_pred_performance.pdf"))
plt.show()

