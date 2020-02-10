'''
Code to cross validate model between 2 experimental data
'''

import os
from os.path import join
from glob import glob
from load_neural_data import ExpTable, ExpData
from time import time
from skimage.transform import resize #, rescale, downscale_local_mean
from skimage.io import imread, imread_collection
from skimage.color import gray2rgb
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
import numpy as np
Result_Dir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Tuning_Interpretation"
DataStore_Dir = r"D:\Tuning_Interpretation"
#%%
for Expi in range(1, 46):
    ftr = (ExpTable.Expi == Expi) & ExpTable.expControlFN.str.contains("generate") & ExpTable.Exp_collection.str.contains("Manifold")
    print(ExpTable.comments[ftr].str.cat())
    EData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
    EData.load_mat()
    Expi = ExpTable.Expi[ftr].to_numpy()[0]
    print("Process Experiment %d" % Expi)
    IsEvolution = ExpTable.expControlFN[ftr].str.contains("generate").to_numpy()[0]
    EData.find_generated()  # fit the model only to generated images.
    Exp_Dir = join(Result_Dir, "Exp%d_Chan%d_%s" % (Expi, EData.pref_chan, "Evol" if IsEvolution else "Man"))
    DS_Dir = join(DataStore_Dir, "Exp%d_Chan%d_%s" % (Expi, EData.pref_chan, "Evol" if IsEvolution else "Man"))
    Exp_str = "%s Exp%d Pref Chan%d" % ("Evolution" if IsEvolution else "Manifold", Expi, EData.pref_chan)
    os.makedirs(Exp_Dir, exist_ok=True)
    os.makedirs(DS_Dir, exist_ok=True)

    with np.load(join(DS_Dir, "feat_tsr.npz")) as data:
        out_feats_all = data["feat_tsr"]

    pref_ch_idx = (EData.spikeID == EData.pref_chan).nonzero()[1]
    psths = EData.rasters[:, :, pref_ch_idx[0]]
    scores = (psths[EData.gen_rows_idx, 50:200].mean(axis=1) - psths[EData.gen_rows_idx, 0:40].mean(axis=1)).squeeze()
    t1 = time()
    RdgCV = RidgeCV(alphas=[1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1E4]).fit( \
        out_feats_all.reshape((out_feats_all.shape[0], -1)), scores)
    LssCV = LassoCV(alphas=[1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1E4]).fit(\
        out_feats_all.reshape((out_feats_all.shape[0], -1)), scores)
    print("Selected Ridge alpha %.1f" % RdgCV.alpha_)
    print("Selected Lasso alpha %.1f" % LssCV.alpha_)
    t2 = time()
    print("%.1f sec spent on model fitting." % (t2 - t1))
    # % Prediction
    pred_score = RdgCV.predict(out_feats_all.reshape((out_feats_all.shape[0], -1)))
    Lss_pred_score = LssCV.predict(out_feats_all.reshape((out_feats_all.shape[0], -1)))

    RdgweightTsr = np.reshape(RdgCV.coef_, out_feats_all.shape[1:])
    LssweightTsr = np.reshape(LssCV.coef_, out_feats_all.shape[1:])

    np.savez(join(Exp_Dir, "RLM_weight_store.npz"), LssCoef=LssweightTsr, LssBias=LssCV.intercept_,
             LssAlpha=LssCV.alpha_, RdgCoef=RdgweightTsr, RdgBias=RdgCV.intercept_,
             RdgAlpha=RdgCV.alpha_, WeightShape=out_feats_all.shape[1:])

    plt.figure()
    plt.plot(scores, alpha=0.5, label="Neuro Rsp")
    plt.plot(pred_score, alpha=0.5, label="Ridge Model Pred")
    plt.plot(Lss_pred_score, alpha=0.5, label="Lasso Model Pred")
    plt.title("Ridge (alpha=%d) and LASSO model (alpha=%d)\n%s" % (RdgCV.alpha_, LssCV.alpha_, Exp_str))
    plt.xlabel("image presentation")
    plt.ylabel("score")
    plt.legend()
    plt.savefig(join(Exp_Dir, "RidgeLasso_Prediction_cmp.png"))
    plt.show()
    plt.close("all")

    RdgwightMap = np.abs(RdgweightTsr).sum(axis=2)
    alpha_msk = RdgwightMap / np.max(RdgwightMap)
    figh = plt.figure(figsize=[5, 4])
    plt.matshow(RdgwightMap, figh.number)
    plt.colorbar()
    plt.title("Ridge Regression (alpha=%d)\n%s" % (RdgCV.alpha_, Exp_str))
    figh.show()
    figh.savefig(join(Exp_Dir, "HeatMask_Ridge%d.png" % RdgCV.alpha_))

    LsswightMap = np.abs(LssweightTsr).sum(axis=2)
    alpha_msk = LsswightMap / np.max(LsswightMap)
    figh2 = plt.figure(figsize=[5, 4])
    plt.matshow(LsswightMap, figh2.number)
    plt.colorbar()
    plt.title("Lasso Regression (alpha=%d)\n%s" % (LssCV.alpha_, Exp_str))
    figh2.show()
    figh2.savefig(join(Exp_Dir, "HeatMask_Lasso%d.png" % LssCV.alpha_))
