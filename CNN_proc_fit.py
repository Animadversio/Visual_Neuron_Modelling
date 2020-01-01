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
#%%  Select the experiment to load, by filtering the experiment excel table
# ftr = (ExpTable.Expi == 12) & ExpTable.expControlFN.str.contains("generate")
# print(ExpTable.comments[ftr].str.cat())
# EData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
# EData.load_mat()
# #%%
ftr = (ExpTable.Expi == 11) & ExpTable.expControlFN.str.contains("generate")
print(ExpTable.comments[ftr].str.cat())
EData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
EData.load_mat()
Expi = ExpTable.Expi[ftr].to_numpy()[0]
# Use this flag to determine how to name the folder / how to load the data
IsEvolution = ExpTable.expControlFN[ftr].str.contains("generate").to_numpy()[0]
Exp_Dir = join(Result_Dir, "Exp%d_Chan%d_%s" % (Expi, EData.pref_chan, "Evol" if IsEvolution else "Man"))
os.makedirs(Exp_Dir, exist_ok=True)
#%%
ftr = (ExpTable.Expi == 12) & ExpTable.expControlFN.str.contains("selectivity")
print(ExpTable.comments[ftr].str.cat())
MData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
MData.load_mat()
Exp_Dir = join(Result_Dir, "Exp12_Chan20_Man")
Exp_Dir = join(Result_Dir, "Exp12_Chan20_Evol")
#%%
import tensorflow as tf
from alexnet.alexnet import MyAlexnet
net = MyAlexnet()
#%% Demo code
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
init = tf.initialize_all_variables()
sess = tf.Session(config=config)
sess.run(init)
#t = time.time()
#output = sess.run(net.conv4, feed_dict = {net.x:[im1,im2]})
output = sess.run(net.conv4, feed_dict = {net.x: input_tsr})

#%%
stimnames = [join(MData.stimuli, imgfn)+".jpg" for imgfn in MData.imgnms]
imgpaths_ds = tf.data.Dataset.from_tensor_slices(stimnames)
def load_preprocess(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image /= 255.0  # normalize to [0,1] range
    return image
image_ds = imgpaths_ds.map(load_preprocess)
#%% tensorflow input pipeline
img_raw = tf.io.read_file(stimnames[0])
img_tensor = tf.image.decode_image(img_raw)
print(img_tensor.shape)
print(img_tensor.dtype)
#%% Experiment with Tensorflow input pipeline (obsolete. finally using numpy input pipeline)
stimnames = [join(MData.stimuli, imgfn)+".jpg" if "gab_ori_" not in imgfn else join(MData.stimuli, imgfn)+".bmp"
             for imgfn in MData.imgnms]
imgpaths_ds = tf.data.Dataset.from_tensor_slices(stimnames[:30])
def load_preprocess(path):
    image = tf.io.read_file(path)
    image = tf.cond(
        tf.image.is_jpeg(image),
        lambda: tf.image.decode_jpeg(image, channels=3),
        lambda: tf.image.grayscale_to_rgb(tf.image.decode_bmp(image, channels=1)))
    image = tf.image.resize(image, [227, 227])
    image /= 255.0  # normalize to [0,1] range
    return image

image_ds = imgpaths_ds.map(load_preprocess)
Batchs = image_ds.batch(10)
iterator = Batchs.make_initializable_iterator()#Batchs.make_one_shot_iterator()
next_element = iterator.get_next()
with tf.Session() as sess:
    #ids, img_batch = next(iter(Batchs))
    sess.run(iterator.initializer)
    while True:
        try:
            # value = sess.run(next_element)
            output = sess.run(net.conv4, feed_dict={net.x: next_element})
            print(output)
        except tf.errors.OutOfRangeError:
            break
    # imgs = sess.run(next_element)

    # for ids, img_batch in Batchs:
    #     output = sess.run(net.conv4, feed_dict = {net.x:img_batch})
    #     print(output)
#%%
# stimnames = [join(MData.stimuli, imgfn)+".jpg" if "gab_ori_" not in imgfn else join(MData.stimuli, imgfn)+".bmp"
#              for imgfn in MData.imgnms]
# some stimuli ends in bmp some in jpg
#%

# Batch_size = 10
# imgs = imread_collection(stimpaths[:10])
# # input_tsr = np.empty([0, 227, 227, 3])
# # ppimgs = [gray2rgb(resize(img, (227, 227),order=1, anti_aliasing=True)) for img in imgs]
# ppimgs = [gray2rgb(resize(img, (227, 227),order=1, anti_aliasing=True))[np.newaxis,:] for img in imgs]
# input_tsr = np.concatenate(tuple(ppimgs), axis=0)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# with tf.compat.v1.Session(config=config) as sess:
#     sess.run(tf.global_variables_initializer())
#     output = sess.run(net.conv4, feed_dict={net.x: input_tsr})
# this is not good.....But don't know why
#%%
init = tf.initialize_all_variables()
sess = tf.Session(config=tf.ConfigProto())
sess.run(init)
#%% Feeding image through CNN to get features (Numpy input pipeline)
if IsEvolution:
    EData.find_generated() # fit the model only to generated images.
    stimpaths = [glob(join(EData.stimuli, imgfn+"*"))[0] for imgfn in EData.gen_fns]
else:
    stimpaths = [glob(join(EData.stimuli, imgfn + "*"))[0] for imgfn in EData.imgnms]
t0 = time()
Bnum = 10
print("%d images to fit the model, estimated batch number %d."%(len(stimpaths), np.ceil(len(stimpaths)/Bnum)))
out_feats_all = np.empty([], dtype=np.float32)
idx_csr = 0
BS_num = 0
while idx_csr < len(stimpaths):
    idx_ub = min(idx_csr + Bnum, len(stimpaths))
    imgs = imread_collection(stimpaths[idx_csr:idx_ub])
    ppimgs = [gray2rgb(resize(img, (227, 227), order=1, anti_aliasing=True))[np.newaxis, :] for img in imgs]
    input_tsr = np.concatenate(tuple(ppimgs), axis=0)
    output = sess.run(net.conv4, feed_dict={net.x: input_tsr})
    out_feats_all = np.concatenate((out_feats_all, output), axis=0) if out_feats_all.shape else output
    idx_csr = idx_ub
    BS_num += 1
    print("Finished %d batch, take %.1f s" % (BS_num, time() - t0))
t1 = time()
# temporially safe files
np.savez("Efeat_tsr2.npz", feat_tsr = out_feats_all, ephysFN=EData.ephysFN, stimuli_path=EData.stimuli)
# np.savez("Efeat_tsr.npz", feat_tsr = out_feats_all, ephysFN=EData.ephysFN, stimuli_path=EData.stimuli)
print("%.1f s" % (t1 - t0))  # Tensorflow 115.1s for 10 sample batch! Faster than torch
# 187.5 s for 2000 images
#%%
# data = np.load("Mfeat_tsr.npz")
# out_feats_all = data["feat_tsr"]
# np.savez("Mfeat_tsr.npz", feat_tsr = out_feats_all, ephysFN=MData.ephysFN, stimuli_path=MData.stimuli)
# 203.1430070400238 s for 3000 samples # 344s for larger batch
#%%
MData.close()
EData.close()
#%% Compute scores and fit it towards features
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV
pref_ch_idx = (MData.spikeID == MData.pref_chan).nonzero()[1]
psths = MData.rasters[:, :, pref_ch_idx[0]]
scores = (psths[:, 50:200].mean(axis=1) - psths[:, 0:40].mean(axis=1)).squeeze()
# pref_ch_idx = (EData.spikeID == EData.pref_chan).nonzero()[1]
# psths = EData.rasters[:, :, pref_ch_idx[0]]
# scores = (psths[EData.gen_rows_idx, 50:200].mean(axis=1) - psths[EData.gen_rows_idx, 0:40].mean(axis=1)).squeeze()
RdgCV = RidgeCV(alphas=[1e-2, 1e-1, 1, 1e1, 1e2, 1e3]).fit(out_feats_all.reshape((out_feats_all.shape[0],-1)), scores)
#%%
import matplotlib.pylab as plt
weightTsr = np.reshape(RdgCV.coef_, out_feats_all.shape[1:])
wightMap = np.abs(weightTsr).sum(axis=2)
figh = plt.figure(figsize=[5, 4])
plt.matshow(wightMap, figh.number)
plt.colorbar()
figh.show()
figh.savefig(join(Exp_Dir, "Heatmap_Ridge%d.png"%RdgCV.alpha_))
#%%
from utils import build_montages
import lucid.optvis.objectives as objectives
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.transform as transform
import lucid.modelzoo.vision_models as models
model = models.AlexNet()
model.load_graphdef()
#%%
row_num, col_num = weightTsr.shape[:2]
img_arr = []
param_f = lambda: param.image(128, fft=True, decorrelate=True)
transforms = [
    transform.pad(16),
    transform.jitter(8),
    transform.random_scale([n/100. for n in range(80, 120)]),
    #transform.random_rotate(range(-10,10) + range(-5,5) + 10*range(-2,2)),
    transform.jitter(2)
]
t0 = time()
for i in range(row_num):
    for j in range(col_num):
        weightVec = weightTsr[i:i+1, j:j+1, :].mean(axis=(0, 1)).astype(np.float32)
        # img = FVis.visualize(sz=50, layer=feat[8], filter=filters, weights=weightVec/weightVec.std(),
        #              blur=10, opt_steps=11, upscaling_steps=7, upscaling_factor=1.2, print_losses=True)
        obj = objectives.direction_cossim("conv4_1", vec=weightVec)
        img = render.render_vis(model, obj, param_f=param_f)[0] #, transforms=transforms
        img_arr.append(img[0, :])
        plt.imsave(join(Exp_Dir, "layer_" + "conv4" + "_Lucid_(%d, %d).jpg" % (i+1, j+1)), img[0, :])
        print("Finished plotting unit (%d, %d), take %.1f s" % (i+1, j+1, time() - t0))
#%
montages = build_montages(img_arr, (128, 128), (row_num, col_num))
plt.figure(figsize=[10, 10]);plt.imshow(montages[0])
plt.imsave(join(Exp_Dir, "layer_" + "conv4" + "_Lucid_VisMontage.jpg"), montages[0])
plt.show()
