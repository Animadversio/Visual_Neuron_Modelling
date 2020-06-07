"""Script to process image stimuli for each experiments."""
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
#%% Set BGR Image net mean parameters
RGBmean = np.array([0.485, 0.456, 0.406])
RGBstd = np.array([0.229, 0.224, 0.225])
BGRmean = RGBmean[::-1]
BGRstd = RGBstd[::-1]
#%% Fit with one set of data and validate with another set.
from alexnet.alexnet import MyAlexnet
net = MyAlexnet()
import tensorflow as tf

init = tf.initialize_all_variables()
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)
# TF GPU test code
im1 = (imread("alexnet\laska.png")[:,:,:3]).astype(np.float32)
im1 = im1 - np.mean(im1)
im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
output = sess.run(net.conv4, feed_dict={net.x: [im1,im1]})
#%% Load a Generation (Evolution Experiment)
for Expi in range(1, 46):
    ftr = (ExpTable.Expi == Expi) & ExpTable.expControlFN.str.contains("generate") & ExpTable.Exp_collection.str.contains("Manifold")
    print(ExpTable.comments[ftr].str.cat())
    EData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
    EData.load_mat()
    Expi = ExpTable.Expi[ftr].to_numpy()[0]
    # Use this flag to determine how to name the folder / how to load the data
    IsEvolution = ExpTable.expControlFN[ftr].str.contains("generate").to_numpy()[0]
    EData.find_generated()  # fit the model only to generated images.
    Exp_Dir = join(Result_Dir, "Exp%d_Chan%d_%s" % (Expi, EData.pref_chan, "Evol" if IsEvolution else "Man"))
    DS_Dir = join(DataStore_Dir, "Exp%d_Chan%d_%s" % (Expi, EData.pref_chan, "Evol" if IsEvolution else "Man"))
    Exp_str = "%s Exp%d Pref Chan%d" % ("Evolution" if IsEvolution else "Manifold", Expi, EData.pref_chan)
    os.makedirs(Exp_Dir, exist_ok=True)
    os.makedirs(DS_Dir, exist_ok=True)

    fnlst = glob(EData.stimuli + "\\*")
    stimpaths = [[nm for nm in fnlst if imgfn in nm][0] for imgfn in EData.gen_fns]
    t0 = time()
    Bnum = 1
    print("%d images to fit the model, estimated batch number %d." % (len(stimpaths), np.ceil(len(stimpaths) / Bnum)))
    out_feats_all = np.empty([], dtype=np.float32)
    idx_csr = 0
    BS_num = 0
    while idx_csr < len(stimpaths):
        idx_ub = min(idx_csr + Bnum, len(stimpaths))
        imgs = imread_collection(stimpaths[idx_csr:idx_ub])
        # oneline the preprocessing step
        ppimgs = [
            (gray2rgb(resize(img, (227, 227), order=1, anti_aliasing=True))[np.newaxis, :, :, ::-1] - BGRmean) / BGRstd
            for img in imgs]
        input_tsr = np.concatenate(tuple(ppimgs), axis=0)
        output = sess.run(net.conv3, feed_dict={net.x: input_tsr})
        out_feats_all = np.concatenate((out_feats_all, output), axis=0) if out_feats_all.shape else output
        idx_csr = idx_ub
        BS_num += 1
        print("Finished %d batch, take %.1f s" % (BS_num, time() - t0))
    t1 = time()
    # Temporially safe files
    np.savez(join(DS_Dir, "feat_tsr.npz"), feat_tsr=out_feats_all, ephysFN=EData.ephysFN, stimuli_path=EData.stimuli)
    print("%.1f s" % (t1 - t0))  # Tensorflow 115.1s for 10 sample batch! Faster than torch
    del out_feats_all
#%% Load a Selectivity (Manifold Experiment)
for Expi in range(3, 46):
    ftr = (ExpTable.Expi == Expi) & ExpTable.expControlFN.str.contains("selectivity") & ExpTable.Exp_collection.str.contains("Manifold")
    print(ExpTable.comments[ftr].str.cat())
    MData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
    MData.load_mat()
    Expi_M = ExpTable.Expi[ftr].to_numpy()[0]
    print("Processing Manifold Exp %d" % Expi_M)
    IsManifold = ExpTable.expControlFN[ftr].str.contains("selectivity").to_numpy()[0]
    assert Expi_M == Expi  # confirm the experimental number coincide with each other.
    Exp_Dir = join(Result_Dir, "Exp%d_Chan%d_%s" % (Expi, ExpTable[ftr].pref_chan.array[0], "Man" if IsManifold else "Evol"))
    DS_Dir = join(DataStore_Dir, "Exp%d_Chan%d_%s" % (Expi, ExpTable[ftr].pref_chan.array[0], "Man" if IsManifold else "Evol"))
    os.makedirs(Exp_Dir, exist_ok=True)
    os.makedirs(DS_Dir, exist_ok=True)
    assert IsManifold
    fnlst = glob(MData.stimuli+"\\*")
    stimpaths = [[nm for nm in fnlst if imgfn in nm][0] for imgfn in MData.imgnms]
    t0 = time()
    Bnum = 1
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
    np.savez(join(DS_Dir, "feat_tsr.npz"), feat_tsr=out_feats_all_M, ephysFN=MData.ephysFN, stimuli_path=MData.stimuli)
    print("%.1f s" % (t1 - t0))
    del out_feats_all_M
#%%

#%%%%%%%% Starts the model fitting part
with np.load("Efeat_tsr2.npz") as data:
    out_feats_all = data["feat_tsr"]