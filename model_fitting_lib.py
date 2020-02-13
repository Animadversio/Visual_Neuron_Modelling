"""Formalize and making the experiment processing less verbose"""
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

class model_fitter:
    def __init__(self):
        self.Exp_Dir = ""
        self.out_feats_all = None
        self.feat_proc = False

    def load_experiment(self, ftr=None, Expi=None, exptype=None):
        """ftr should be a binary filter same length of the ExpTable"""
        if ftr is None:
            if exptype == "generate":
                ftr = (ExpTable.Expi == Expi) & ExpTable.expControlFN.str.contains(
                    "generate") & ExpTable.Exp_collection.str.contains("Manifold")
            elif exptype == "selectivity":
                ftr = (ExpTable.Expi == Expi) & ExpTable.expControlFN.str.contains(
                    "selectivity") & ExpTable.Exp_collection.str.contains("Manifold")
            else:
                raise Exception("`exptype` argument accept only { \"generate\", \"selectivity\"} result.")
        self.EData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
        self.EData.load_mat()
        self.Expi = ExpTable.Expi[ftr].to_numpy()[0]

        self.IsEvolution = ExpTable.expControlFN[ftr].str.contains("generate").to_numpy()[0]
        self.IsManifold = ExpTable.expControlFN[ftr].str.contains("selectivity").to_numpy()[0]
        if self.IsEvolution:
            self.pref_chan = self.EData.pref_chan
            self.EData.find_generated()  # generate index for generated images.
        if self.IsManifold:
            self.pref_chan = int(ExpTable[ftr].pref_chan.array[0])
        print("Process %s Experiment %d" % ("Evolution" if self.IsEvolution else "Manifold", self.Expi))
        print(ExpTable.comments[ftr].str.cat())

        self.Exp_Dir = join(Result_Dir, "Exp%d_Chan%d_%s" % (self.Expi, self.pref_chan, "Evol" if self.IsEvolution else "Man"))
        self.DS_Dir = join(DataStore_Dir, "Exp%d_Chan%d_%s" % (self.Expi, self.pref_chan, "Evol" if self.IsEvolution else "Man"))
        os.makedirs(self.Exp_Dir, exist_ok=True)
        os.makedirs(self.DS_Dir, exist_ok=True)
        self.Exp_str = "Manifold Exp%d Pref Chan%d" % (self.Expi, self.pref_chan)

    def proc_images(self, network, layer, batch=1, savefeat=True):
        """use CNN to process images, allows different backends"""
        self.out_feats_all
        self.feat_tsr_shape = self.out_feats_all.shape[1:]
        self.feat_proc = True
        if savefeat:
            np.savez(join(self.DS_Dir, "feat_tsr.npz"), feat_tsr=self.out_feats_all,
                     ephysFN=self.EData.ephysFN, stimuli_path=self.EData.stimuli)

    def fit_model(self, ridge_alpha=None,lasso_alpha=None,save=True):
        """Fit general regularized linear model for neural response"""
        if self.feat_proc is False: # load the features from processed datastore.
            with np.load(join(self.DS_Dir, "feat_tsr.npz")) as data:
                self.out_feats_all = data["feat_tsr"]

        pref_ch_idx = (self.EData.spikeID == self.EData.pref_chan).nonzero()[1]
        psths = self.EData.rasters[:, :, pref_ch_idx[0]]
        if self.IsEvolution: # only choose to fit model based on the evolved images.
            scores = (psths[self.EData.gen_rows_idx, 50:200].mean(axis=1) - psths[self.EData.gen_rows_idx, 0:40].mean(
                axis=1)).squeeze()
        else: # fit model based on all responses.
            scores = (psths[:, 50:200].mean(axis=1) - psths[:, 0:40].mean(axis=1)).squeeze()
        t1 = time()
        self.RdgCV = RidgeCV(alphas=[1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1E4]).fit( \
            self.out_feats_all.reshape((self.out_feats_all.shape[0], -1)), scores)
        print("Selected Ridge alpha %.1f" % self.RdgCV.alpha_)
        self.LssCV = LassoCV(alphas=[1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1E4]).fit( \
            self.out_feats_all.reshape((self.out_feats_all.shape[0], -1)), scores)
        print("Selected Lasso alpha %.1f" % self.LssCV.alpha_)
        t2 = time()
        print("%.1f sec spent on model fitting." % (t2 - t1))
        if save:
            self.save_model()

    def save_model(self):
        # assume we have self.feat_tsr_shape now!
        RdgweightTsr = np.reshape(self.RdgCV.coef_, self.feat_tsr_shape)
        LssweightTsr = np.reshape(self.LssCV.coef_, self.feat_tsr_shape)
        np.savez(join(self.Exp_Dir, "RLM_weight_store.npz"), LssCoef=LssweightTsr, LssBias=self.LssCV.intercept_,
                 LssAlpha=self.LssCV.alpha_, RdgCoef=RdgweightTsr, RdgBias=self.RdgCV.intercept_,
                 RdgAlpha=self.RdgCV.alpha_, WeightShape=self.feat_tsr_shape)

    def load_model(self):
        with np.load(join(self.Exp_Dir, "RLM_weight_store.npz")) as data:
            LssweightTsr = data["LssCoef"]
            self.LssCV.coef_ = LssweightTsr.reshape((-1,))
            self.LssCV.alpha_ = data["LssAlpha"]
            self.LssCV.intercept_ = data["LssBias"]
            RdgweightTsr = data["LssCoef"]
            self.RdgCV.coef_ = RdgweightTsr.reshape((-1,))
            self.RdgCV.alpha_ = data["RdgAlpha"]
            self.RdgCV.intercept_ = data["RdgBias"]
            self.feat_tsr_shape = data["WeightShape"]

    def cross_validate(self):

    def visualize(self):
        # Plot Heatmap

        # Plot Manifold if it's Manifold experiment.

    def visualize_mask_on_img(self):
