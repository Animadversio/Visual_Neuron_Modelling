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
import numpy as np
Result_Dir = r"C:\Users\ponce\OneDrive - Washington University in St. Louis\Tuning_Interpretation"
#%% Load a Generation (Evolution Experiment)
ftr = (ExpTable.Expi == 12) & ExpTable.expControlFN.str.contains("generate")
print(ExpTable.comments[ftr].str.cat())
EData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
EData.load_mat()
Expi = ExpTable.Expi[ftr].to_numpy()[0]
# Use this flag to determine how to name the folder / how to load the data
IsEvolution = ExpTable.expControlFN[ftr].str.contains("generate").to_numpy()[0]
#%% Load a Selectivity (Manifold Experiment)
ftr = (ExpTable.Expi == 12) & ExpTable.expControlFN.str.contains("selectivity")
print(ExpTable.comments[ftr].str.cat())
MData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
MData.load_mat()
Expi_M = ExpTable.Expi[ftr].to_numpy()[0]
IsManifold = ExpTable.expControlFN[ftr].str.contains("selectivity").to_numpy()[0]
assert Expi_M == Expi  # confirm the experimental number coincide with each other.

Exp_Dir = join(Result_Dir, "Exp%d_Chan%d_EM_CV")
os.makedirs(Exp_Dir, exist_ok=True)

#%% Fit with one set of data and validate with another set.
import tensorflow as tf
from alexnet.alexnet import MyAlexnet
net = MyAlexnet()

init = tf.initialize_all_variables()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
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
    output = sess.run(net.conv4, feed_dict={net.x: input_tsr})
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
#%
assert IsManifold
fnlst = glob(MData.stimuli+"\\*")
stimpaths = [[nm for nm in fnlst if imgfn in nm][0] for imgfn in MData.imgnms]
t0 = time()
Bnum = 10
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
    output = sess.run(net.conv4, feed_dict={net.x: input_tsr})
    out_feats_all_M = np.concatenate((out_feats_all_M, output), axis=0) if out_feats_all_M.shape else output
    idx_csr = idx_ub
    BS_num += 1
    print("Finished %d batch, take %.1f s" % (BS_num, time() - t0))
t1 = time()
# Temporially safe files
np.savez("Mfeat_tsr2.npz", feat_tsr=out_feats_all_M, ephysFN=MData.ephysFN, stimuli_path=MData.stimuli)
print("%.1f s" % (t1 - t0))
#%% Compute scores and fit it towards features
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV
pref_ch_idx = (EData.spikeID == EData.pref_chan).nonzero()[1]
psths = EData.rasters[:, :, pref_ch_idx[0]]
scores = (psths[EData.gen_rows_idx, 50:200].mean(axis=1) - psths[EData.gen_rows_idx, 0:40].mean(axis=1)).squeeze()
RdgCV = RidgeCV(alphas=[1e-2, 1e-1, 1, 1e1, 1e2, 1e3]).fit(out_feats_all.reshape((out_feats_all.shape[0], -1)), scores)
print("Selected alpha %.1f" % RdgCV.alpha_)
#%%
pref_ch_idx = (MData.spikeID == EData.pref_chan).nonzero()[1]
psths = MData.rasters[:, :, pref_ch_idx[0]]
Mscores = (psths[:, 50:200].mean(axis=1) - psths[:, 0:40].mean(axis=1)).squeeze()
pred_score = RdgCV.predict(out_feats_all_M.reshape((out_feats_all_M.shape[0], -1)))

#%%
import matplotlib.pylab as plt
#%%
plt.figure()
plt.plot(Mscores, alpha=0.6)
plt.plot(pred_score, alpha=0.6)
plt.show()
#%%
plt.figure()
plt.scatter(Mscores, pred_score)
plt.show()