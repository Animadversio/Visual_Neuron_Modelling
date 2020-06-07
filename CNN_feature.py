from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from load_neural_data import ExpTable, ExpData
import numpy as np
from glob import glob
from time import time
from os.path import join
#%%
keras_test = False
tf_test = False
if keras_test:
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_conv3').output)
    img_path = r'alexnet\laska.png'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    block4_pool_features = model.predict(x)
if tf_test:
    from alexnet.alexnet import MyAlexnet
    import tensorflow as tf
    net = MyAlexnet()
    init = tf.initialize_all_variables()
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)
    # TF GPU test code
    im1 = (imread("alexnet\laska.png")[:, :, :3]).astype(np.float32)
    im1 = im1 - np.mean(im1)
    im1[:, :, 0], im1[:, :, 2] = im1[:, :, 2], im1[:, :, 0]
    output = sess.run(net.conv4, feed_dict={net.x: [im1, im1]})
#%%
RGBmean = np.array([0.485, 0.456, 0.406])
RGBstd = np.array([0.229, 0.224, 0.225])
BGRmean = RGBmean[::-1]
BGRstd = RGBstd[::-1]
#%%
class CNNfeature:
    def __init__(self, backend="keras"):
        self.backend = backend
        if backend == "keras":
            from keras.preprocessing import image
            from keras.models import Model

        if backend == "tf":
            import tensorflow as tf

        if backend == "torch":
            import torch

    def set_model_param(self, model="VGG16", layer='block4_conv3'):
        if self.backend == "keras":
            if model == "VGG16":
                from keras.applications.vgg16 import VGG16
                from keras.applications.vgg16 import preprocess_input
                base_model = VGG16(weights='imagenet')
                self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer).output)
                self.preprocess = preprocess_input

        if self.backend == "tf":
            if model == "AlexNet":
                from alexnet.alexnet import MyAlexnet
                self.net = MyAlexnet()
                init = tf.initialize_all_variables()
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                sess = tf.Session(config=config)
                sess.run(init)
                layer_dict = {"conv1": self.net.conv1, "conv2": self.net.conv2, "conv3": self.net.conv3,
                              "conv4": self.net.conv4, "conv5": self.net.conv5, "fc6": self.net.fc6,
                              "fc7": self.net.fc7, "fc8": self.net.fc8}
                self.output_layer = layer_dict[layer]

    def preprocess(self, img):
        if self.backend == "keras":
            if model == "VGG16":
                from keras.applications.vgg16 import preprocess_input
                return preprocess_input(img)
        if self.backend == "tf":
            if model == "AlexNet":
                from skimage.transform import resize
                from skimage.color import gray2rgb
                ppimg = (gray2rgb(resize(img, (227, 227), order=1, anti_aliasing=True))[np.newaxis, :, :, ::-1] - BGRmean) / BGRstd
                return ppimg
    def process(self, ppimgs):
        if self.backend == "keras":
            return self.model.predict(ppimgs)

        if self.backend == "tf":
            with tf.Session() as sess:
                return sess.run(self.output_layer, feed_dict={self.net.x: ppimgs})
#%%
DataStore_Dir = r"D:\Tuning_Interpretation"
# https://stackoverflow.com/questions/44100837/disable-keras-batch-normalization-standardization
for Expi in range(11, 46):
    t00 = time()
    ftr = (ExpTable.Expi == Expi) & ExpTable.expControlFN.str.contains("generate") & ExpTable.Exp_collection.str.contains("Manifold")
    print(ExpTable.comments[ftr].str.cat())
    EData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
    EData.load_mat()
    EData.find_generated()
    DS_Dir = join(DataStore_Dir, "Exp%d_Chan%d_Evol" % (Expi, EData.pref_chan))
    fnlst = glob(EData.stimuli + "\\*")
    stimpaths = [[nm for nm in fnlst if imgfn in nm][0] for imgfn in EData.gen_fns]
    t0 = time()
    print("Mat file loading time %.1f s" % (t0 - t00))
    Bnum = 15
    print("%d images to fit the model, estimated batch number %d." % (len(stimpaths), np.ceil(len(stimpaths) / Bnum)))
    feature_all = np.empty([], dtype=np.float32)
    idx_csr = 0
    BS_num = 0
    while idx_csr < len(stimpaths):
        idx_ub = min(idx_csr + Bnum, len(stimpaths))
        ppimgs = []
        for img_path in stimpaths[idx_csr:idx_ub]:
            # should be taken care of by the CNN part
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            ppimgs.append(x[np.newaxis, :, :, :].copy())
        input_tsr = np.concatenate(tuple(ppimgs), axis=0)
        # should be taken care of by the CNN part
        features = model.predict(input_tsr)
        feature_all = np.concatenate((feature_all, features), axis=0) if feature_all.shape else features
        idx_csr = idx_ub
        BS_num += 1
        print("Finished %d batch, take %.1f s" % (BS_num, time() - t0))
    t1 = time()
    np.savez(join(DS_Dir, "VGG_feat_tsr.npz"), feat_tsr=feature_all, ephysFN=EData.ephysFN, stimuli_path=EData.stimuli)
    print("%.1f s" % (t1 - t0))
    del feature_all
#%%
for Expi in range(1, 46):
    ftr = (ExpTable.Expi == Expi) & ExpTable.expControlFN.str.contains("generate") & ExpTable.Exp_collection.str.contains("Manifold")
    print(ExpTable.comments[ftr].str.cat())
    EData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
    EData.load_mat()
    EData.find_generated()
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
    print("%.1f s" % (t1 - t0))