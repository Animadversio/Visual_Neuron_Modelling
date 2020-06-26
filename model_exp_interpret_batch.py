'''
Code to Fit a model and see how that generalize through to correlated experiments
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
import numpy as np
Result_Dir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Tuning_Interpretation"
DataStore_Dir = r"D:\Tuning_Interpretation"
#%% Load a Generation (Evolution Experiment)
ftr = (ExpTable.Expi == 11) & ExpTable.expControlFN.str.contains("generate")
print(ExpTable.comments[ftr].str.cat())
EData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
EData.load_mat()
Expi = ExpTable.Expi[ftr].to_numpy()[0]
# Use this flag to determine how to name the folder / how to load the data
IsEvolution = ExpTable.expControlFN[ftr].str.contains("generate").to_numpy()[0]
EData.find_generated()
Exp_Dir = join(Result_Dir, "Exp%d_Chan%d_%s" % (Expi, EData.pref_chan, "Evol" if IsEvolution else "Man"))
DS_Dir = join(DataStore_Dir, "Exp%d_Chan%d_%s" % (Expi, EData.pref_chan, "Evol" if IsEvolution else "Man"))
Exp_str = "%s Exp%d Pref Chan%d" % ("Evolution" if IsEvolution else "Manifold", Expi, EData.pref_chan)
#%% Load a Selectivity (Manifold Experiment)
ftr = (ExpTable.Expi == 11) & ExpTable.expControlFN.str.contains("selectivity")
print(ExpTable.comments[ftr].str.cat())
MData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
MData.load_mat()
Expi_M = ExpTable.Expi[ftr].to_numpy()[0]
IsManifold = ExpTable.expControlFN[ftr].str.contains("selectivity").to_numpy()[0]
assert Expi_M == Expi  # confirm the experimental number coincide with each other.

Exp_Dir = join(Result_Dir, "Exp%d_Chan%d_EM_CV"% (Expi_M, EData.pref_chan))
os.makedirs(Exp_Dir, exist_ok=True)
os.makedirs(DS_Dir, exist_ok=True)
#%% Fit with one set of data and validate with another set.
from alexnet.alexnet import MyAlexnet
net = MyAlexnet()
import tensorflow as tf

init = tf.initialize_all_variables()
config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)
# TF GPU test code
im1 = (imread("alexnet\laska.png")[:,:,:3]).astype(np.float32)
im1 = im1 - np.mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
output = sess.run(net.conv4, feed_dict={net.x: [im1,im1]})
#%% Set BGR Image net mean parameters
RGBmean = np.array([0.485, 0.456, 0.406])
RGBstd = np.array([0.229, 0.224, 0.225])
BGRmean = RGBmean[::-1]
BGRstd = RGBstd[::-1]
#%% Feeding image through CNN to get features (Numpy input pipeline)
assert IsEvolution
EData.find_generated() # fit the model only to generated images.
fnlst = glob(EData.stimuli+"\\*")
stimpaths = [[nm for nm in fnlst if imgfn in nm][0] for imgfn in EData.gen_fns]

t0 = time()
Bnum = 10
print("%d images to fit the model, estimated batch number %d."%(len(stimpaths), np.ceil(len(stimpaths)/Bnum)))
out_feats_all = np.empty([], dtype=np.float32)
idx_csr = 0
BS_num = 0
while idx_csr < len(stimpaths):
    idx_ub = min(idx_csr + Bnum, len(stimpaths))
    imgs = imread_collection(stimpaths[idx_csr:idx_ub])
    # oneline the preprocessing step
    ppimgs = [(gray2rgb(resize(img, (227, 227), order=1, anti_aliasing=True))[np.newaxis, :, :, ::-1] - BGRmean) / BGRstd for img in imgs]
    input_tsr = np.concatenate(tuple(ppimgs), axis=0)
    output = sess.run(net.conv3, feed_dict={net.x: input_tsr})
    out_feats_all = np.concatenate((out_feats_all, output), axis=0) if out_feats_all.shape else output
    idx_csr = idx_ub
    BS_num += 1
    print("Finished %d batch, take %.1f s" % (BS_num, time() - t0))
t1 = time()
# Temporially safe files
np.savez("Efeat_tsr2.npz", feat_tsr=out_feats_all, ephysFN=EData.ephysFN, stimuli_path=EData.stimuli)
print("%.1f s" % (t1 - t0))  # Tensorflow 115.1s for 10 sample batch! Faster than torch
#%
# with np.load("Efeat_tsr.npz") as data:
#    out_feats_all = data["feat_tsr"]
#%%
assert IsManifold
fnlst = glob(MData.stimuli+"\\*")
stimpaths = [[nm for nm in fnlst if imgfn in nm][0] for imgfn in MData.imgnms]
t0 = time()
Bnum = 15
print("%d images to fit the model, estimated batch number %d."%(len(stimpaths), np.ceil(len(stimpaths)/Bnum)))
out_feats_all_M = np.empty([], dtype=np.float32)
idx_csr = 0
BS_num = 0
while idx_csr < len(stimpaths):
    idx_ub = min(idx_csr + Bnum, len(stimpaths))
    imgs = imread_collection(stimpaths[idx_csr:idx_ub])
    # ppimgs = [gray2rgb(resize(img, (227, 227), order=1, anti_aliasing=True))[np.newaxis, :] for img in imgs]
    ppimgs = [(gray2rgb(resize(img, (227, 227), order=1, anti_aliasing=True))[np.newaxis, :, :, ::-1] - BGRmean) / BGRstd
              for img in imgs]
    input_tsr = np.concatenate(tuple(ppimgs), axis=0)
    output = sess.run(net.conv3, feed_dict={net.x: input_tsr})
    out_feats_all_M = np.concatenate((out_feats_all_M, output), axis=0) if out_feats_all_M.shape else output
    idx_csr = idx_ub
    BS_num += 1
    print("Finished %d batch, take %.1f s" % (BS_num, time() - t0))
t1 = time()
# Temporially safe files
np.savez("Mfeat_tsr2.npz", feat_tsr=out_feats_all_M, ephysFN=MData.ephysFN, stimuli_path=MData.stimuli)
print("%.1f s" % (t1 - t0))
#%%%%%%%% Starts the model fitting part
with np.load("Efeat_tsr2.npz") as data:
    out_feats_all = data["feat_tsr"]
#%% Compute scores and fit it towards features
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV
pref_ch_idx = (EData.spikeID == EData.pref_chan).nonzero()[1]
psths = EData.rasters[:, :, pref_ch_idx[0]]
scores = (psths[EData.gen_rows_idx, 50:200].mean(axis=1) - psths[EData.gen_rows_idx, 0:40].mean(axis=1)).squeeze()
t1 = time()
RdgCV = RidgeCV(alphas=[1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1E4]).fit(out_feats_all.reshape((out_feats_all.shape[0], -1)), scores)
LssCV = LassoCV(alphas=[1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1E4]).fit(out_feats_all.reshape((out_feats_all.shape[0], -1)), scores)
print("Selected Ridge alpha %.1f" % RdgCV.alpha_)
print("Selected Lasso alpha %.1f" % LssCV.alpha_)
t2 = time()
print("%.1f sec spent on model fitting." % t2-t1)
#%% Prediction
pred_score = RdgCV.predict(out_feats_all.reshape((out_feats_all.shape[0], -1)))
Lss_pred_score = LssCV.predict(out_feats_all.reshape((out_feats_all.shape[0], -1)))
#%% Plot score traces
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
#%%
RdgweightTsr = np.reshape(RdgCV.coef_, out_feats_all.shape[1:])
RdgwightMap = np.abs(RdgweightTsr).sum(axis=2)
alpha_msk = RdgwightMap / np.max(RdgwightMap)
figh = plt.figure(figsize=[5, 4])
plt.matshow(RdgwightMap, figh.number)
plt.colorbar()
plt.title("Ridge Regression (alpha=%d)\n%s" % (RdgCV.alpha_, Exp_str))
figh.show()
figh.savefig(join(Exp_Dir, "HeatMask_Ridge%d.png"%RdgCV.alpha_))
#%%
LssweightTsr = np.reshape(LssCV.coef_, out_feats_all.shape[1:])
LsswightMap = np.abs(LssweightTsr).sum(axis=2)
alpha_msk = LsswightMap / np.max(LsswightMap)
figh2 = plt.figure(figsize=[5, 4])
plt.matshow(LsswightMap, figh2.number)
plt.colorbar()
plt.title("Lasso Regression (alpha=%d)\n%s" % (LssCV.alpha_,Exp_str))
figh2.show()
figh2.savefig(join(Exp_Dir, "HeatMask_Lasso%d.png"%LssCV.alpha_))
#%%
np.savez(join(Exp_Dir, "RLM_weight_store.npz"), LssCoef=LssweightTsr, LssBias=LssCV.intercept_, LssAlpha=LssCV.alpha_,\
    RdgCoef=RdgweightTsr, RdgBias=RdgCV.intercept_, RdgAlpha=RdgCV.alpha_,WeightShape=out_feats_all.shape[1:])
#%%
with np.load("RLM_store.npz", allow_pickle=True) as data:
    tmp = data["LssCV"]
#%%
#%% Trying to Cross Validate with Manifold Experiments
with np.load("Mfeat_tsr3.npz") as data:
    out_feats_all_M = data["feat_tsr"]
pref_ch_idx = (MData.spikeID == EData.pref_chan).nonzero()[1]
psths = MData.rasters[:, :, pref_ch_idx[0]]
scores_M = (psths[:, 50:200].mean(axis=1) - psths[:, 0:40].mean(axis=1)).squeeze()
pred_score_M = RdgCV.predict(out_feats_all_M.reshape((out_feats_all_M.shape[0], -1)))
Lss_pred_score_M = LssCV.predict(out_feats_all_M.reshape((out_feats_all_M.shape[0], -1)))
#%%
plt.figure()
plt.plot(scores_M, alpha=0.5, label="Manifold Exp")
plt.plot(pred_score_M, alpha=0.5, label="Ridge Model Pred")
plt.plot(Lss_pred_score_M, alpha=0.5, label="Lasso Model Pred")
plt.title("Ridge (alpha=%d) LASSO model (alpha=%d) cross experiment validation\n" % (RdgCV.alpha_, LssCV.alpha_))
plt.legend()
plt.savefig(join(Exp_Dir, "Evol2Manif_Allrsp_CV.png"))
plt.show()
#%% Sort the Experimental Data And Response into a Regular Grid
# MData.imgnms
ang_step = 18
Reps = 11
score_mat = np.zeros((11, 11, Reps))
bsl_mat = np.zeros((11, 11, Reps)) # make sure the 3 dimension has larger size than repitition number!
score_mat.fill(np.nan)
bsl_mat.fill(np.nan)
Rdg_pred_score_mat = np.zeros((11, 11))
Lss_pred_score_mat = np.zeros((11, 11))
for i in range(-5, 6):
    for j in range(-5, 6):
        cur_fn = 'PC2_%d_PC3_%d' % (i*ang_step, j*ang_step)
        img_idx = np.flatnonzero(np.core.defchararray.find(MData.imgnms, cur_fn)!=-1)
        psths = MData.rasters[img_idx, :, pref_ch_idx[0]]
        score_mat[i+5, j+5, :len(img_idx)] = psths[:, 50:150].mean(axis=1) - psths[:, 1:40].mean(axis=1)
        bsl_mat[i+5, j+5, :len(img_idx)] = psths[:, 1:40].mean(axis=1)
        Rdg_pred_score_mat[i+5, j+5] = pred_score_M[img_idx].mean()
        Lss_pred_score_mat[i+5, j+5] = Lss_pred_score_M[img_idx].mean()

Manif_score = np.nanmean(score_mat, axis=2)  # Manifold plot derived from the real monkey data!
# %% Note this page is all fitted on Conv3 data
plt.figure(figsize=[8, 8])
ax1 = plt.subplot(221)
plt.pcolor(Manif_score)
plt.axis("image")
plt.title("Neural Data (Trial Mean)" )
plt.colorbar()
ax2 = plt.subplot(222)
plt.pcolor(Rdg_pred_score_mat)
plt.title("Ridge Model (alpha=%d)\n CrossValidated Prediction about Manifold result" % RdgCV.alpha_)
plt.axis("image")
plt.colorbar()
ax3 = plt.subplot(223)
plt.pcolor(Lss_pred_score_mat)
plt.title("Lasso Model (alpha=%d) CrossValidated\n Prediction about Manifold result" % LssCV.alpha_)
plt.axis("image")
plt.colorbar()
plt.suptitle("Linear Model ~ AlexNet Conv3  Fitted on %s model\n Cross Validated on Manifold Experiments"%Exp_str)
plt.savefig(join(Exp_Dir, "Evol2Manif_rsp_CV.png"))
plt.show()
#%%
ang_step = 18
Reps = 11
score_mat = np.zeros((4, 51, Reps))
score_mat.fill(np.nan)
Rdg_pred_score_mat = np.zeros((4, 51))
Lss_pred_score_mat = np.zeros((4, 51))
for i in range(4):
    for j in range(51):
        cur_fn = 'pasu_%02d_ori_%02d_wg_f' % (j + 1, 2 * i + 1)
        img_idx = np.flatnonzero(np.core.defchararray.find(MData.imgnms, cur_fn)!=-1)
        psths = MData.rasters[img_idx, :, pref_ch_idx[0]]
        score_mat[i, j, :len(img_idx)] = psths[:, 50:150].mean(axis=1) - psths[:, 1:40].mean(axis=1)
        Rdg_pred_score_mat[i, j] = pred_score_M[img_idx].mean()
        Lss_pred_score_mat[i, j] = Lss_pred_score_M[img_idx].mean()
Pasu_score = np.nanmean(score_mat, axis=2)  # Manifold plot derived from the real monkey data!
#%%
plt.figure(figsize=[10, 6])
ax1 = plt.subplot(311)
plt.pcolor(Pasu_score)
plt.title("Neural Data (Trial Mean)" )
plt.axis("image")
plt.colorbar()
ax2 = plt.subplot(312)
plt.pcolor(Rdg_pred_score_mat)
plt.title("Ridge Model (alpha=%d)\n CrossValidated Prediction about Pasu result" % RdgCV.alpha_)
plt.axis("image")
plt.colorbar()
ax3 = plt.subplot(313)
plt.pcolor(Lss_pred_score_mat)
plt.title("Lasso Model (alpha=%d) CrossValidated\n Prediction about Pasu result" % LssCV.alpha_)
plt.axis("image")
plt.colorbar()
plt.suptitle("Linear Model ~ AlexNet Conv3  Fitted on %s model\n Cross Validated on Manifold Experiments" % Exp_str)
plt.savefig(join(Exp_Dir, "Evol2Manif_rsp_Pasu_CV.png"))
plt.show()

#%%
# plt.figure()
# plt.scatter(scores, pred_score)
# plt.xlabel("Response Scores")
# plt.ylabel("Predicted Scores")
# plt.title("Ridge model (alpha=%d) Prediction\n%s" % (RdgCV.alpha_, Exp_str))
# plt.savefig(join(Exp_Dir, "Ridge_Prediction.png"))
# plt.show()
# #%%
# plt.figure()
# plt.scatter(scores, Lss_pred_score)
# plt.xlabel("Response Scores")
# plt.ylabel("Predicted Scores")
# plt.title("LASSO model (alpha=%d) Prediction\n%s" % (LssCV.alpha_, Exp_str))
# plt.savefig(join(Exp_Dir, "Lasso_Prediction.png"))
# plt.show()