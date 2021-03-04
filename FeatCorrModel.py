mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
from scipy.io import loadmat
from skimage.io import imread, imread_collection
from os.path import join
import numpy as np
import matplotlib.pylab as plt
#%%
Animal = "Alfa"
# ManifDyn = loadmat(join(mat_path, Animal + "_ManifPopDynamics.mat"), struct_as_record=False, squeeze_me=True)['ManifDyn']
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True)['EStats']
#%% Note idx in it is recorded in matlab fashion not python
Expi = 3
idx_vect = MStats[Expi-1].manif.idx_grid.reshape([-1])
imgnm_vect = []
for idx in idx_vect:
    imgnm_vect.append(np.unique(MStats[Expi-1].imageName[idx-1])[0])
    print(np.unique(MStats[Expi-1].imageName[idx-1])[0])
#%
from glob import glob
Pasupath = r"N:\Stimuli\2019-Manifold\pasupathy-wg-f-4-ori"
Gaborpath = r"N:\Stimuli\2019-Manifold\gabor"
imgnms = glob(MStats[Expi-1].meta.stimuli+"\\*") + glob(Pasupath+"\\*") + glob(Gaborpath+"\\*")
imgfullpath_vect = [[path for path in imgnms if imgnm in path][0] for imgnm in imgnm_vect]
#%% 
score_mat = np.frompyfunc(lambda psth: np.mean(psth[0,50:,:]),1,1)(MStats[2].manif.psth)
score_mat = score_mat.astype(np.float)
score_vect = score_mat.reshape(-1)
#%%
plt.matshow(score_mat)
plt.axis('image')
plt.show()
#%%
layername_dict ={"vgg16":['conv1', 'conv1_relu',
                         'conv2', 'conv2_relu', 'pool1',
                         'conv3', 'conv3_relu',
                         'conv4', 'conv4_relu', 'pool2',
                         'conv5', 'conv5_relu',
                         'conv6', 'conv6_relu',
                         'conv7', 'conv7_relu', 'pool3',
                         'conv8', 'conv8_relu',
                         'conv9', 'conv9_relu',
                         'conv10', 'conv10_relu', 'pool4',
                         'conv11', 'conv11_relu',
                         'conv12', 'conv12_relu',
                         'conv13', 'conv13_relu', 'pool5',
                         'fc1', 'fc1_relu', 'dropout1',
                         'fc2', 'fc2_relu', 'dropout2',
                         'fc3'],
                 "densenet121":['conv1',
                                 'bn1',
                                 'bn1_relu',
                                 'pool1',
                                 'denseblock1', 'transition1',
                                 'denseblock2', 'transition2',
                                 'denseblock3', 'transition3',
                                 'denseblock4',
                                 'bn2',
                                 'fc1']}
from torchvision import models
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import  SGD, Adam
from lucent.optvis import render, param, transform, objectives
from matver_CNN import getMatVGG, lucent_layernames
VGG = models.vgg16(pretrained=True)
preprocess = transforms.Compose([transforms.ToTensor(),
            #transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#%% Preload images
ppimgs = []
for img_path in imgfullpath_vect:
    # should be taken care of by the CNN part
    curimg = imread(img_path)
    # curimg = resize(curimg, (256, 256))
    x = preprocess(curimg)
    ppimgs.append(x.unsqueeze(0))
input_tsr = torch.cat(tuple(ppimgs), dim=0)
#%% Passing image through network
imgN = input_tsr.shape[0]
basenet = "vgg16"
layername = 'conv8'
layer_idx = layername_dict[basenet].index(layername)
FeatNet = VGG.features[:layer_idx + 1].cuda()
#%%
feat_tsr = torch.tensor([]);
csr = 0; BSz = 42
while csr < imgN:
    cend = min(csr + BSz, imgN)
    with torch.no_grad():
        part_tsr = FeatNet(input_tsr[csr:cend,:,:,:].cuda()).cpu()
    feat_tsr = torch.cat((feat_tsr, part_tsr), 0)
    csr = cend
feat_tsr = feat_tsr.numpy()
tsr_shape = feat_tsr.shape[1:]
feat_tsr = feat_tsr.reshape(imgN,-1)
#%%
# np.corrcoef won't work
# cc_tsr = np.corrcoef(x=feat_tsr.view(121,-1).numpy(), y=score_vect)
if len(score_vect.shape) == 1:
    score_vect = score_vect[:, np.newaxis]
innerProd = np.einsum("ik,ij->kj", score_vect, feat_tsr, )  # [time by features]
scorS = score_vect.mean(axis=0)
scorSSq = (score_vect**2).mean(axis=0)
featS = feat_tsr.mean(axis=0)
featSSq = (feat_tsr**2).mean(axis=0)
scorStd = score_vect.std(axis=0)
featStd = feat_tsr.std(axis=0)
#%%
featStdtsr = featStd.reshape(tsr_shape)
#%
corrtsr = (innerProd / imgN - scorS[:, np.newaxis] @ featS[np.newaxis,:]) / (scorStd[:, np.newaxis] @ featStd[np.newaxis,:])
corrtsr = corrtsr.reshape((score_vect.shape[1],)+tsr_shape)
#%%
L1cc_map = np.abs(corrtsr).mean(axis=(0,1))
import matplotlib.pylab as plt
plt.figure(figsize=[5,4])
plt.matshow(L1cc_map)
plt.axis('image'); plt.axis('off')
plt.title("%s Manif Exp %d\n VGG16 %s features"%(Animal,Expi,layername))
plt.colorbar()
plt.show()
#%%

#%%
Std_map = np.abs(featStd.reshape(-1,*tsr_shape)).mean(axis=(0,1))
plt.figure(figsize=[5,5])
plt.matshow(Std_map)
plt.axis('image'); plt.axis('off')
plt.colorbar()
plt.show()
#%%
FeatNet.cpu()
#
#%% Use Lucent to interpret the correlation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
VGG.to(device).eval()
#%%
obj = objectives.channel("features_%d"%(layer_idx), 100)
images = render.render_vis(VGG, obj)
#%%
maxind = np.unravel_index(L1cc_map.argmax(), L1cc_map.shape)
dirvect = corrtsr[0,:,maxind[0],maxind[1]]
channel = lambda n: objectives.channel("features_%d"%(layer_idx), n)
obj = sum([dirvect[iCh] * channel(iCh) for iCh in range(512)])
#%%
maxind = np.unravel_index(L1cc_map.argmax(), L1cc_map.shape)
dirvect = corrtsr[0, :, maxind[0]-2:maxind[0]+2, maxind[1]-2:maxind[1]+2].mean((1, 2))
stdvect = featStdtsr[:, maxind[0]-2:maxind[0]+2, maxind[1]-2:maxind[1]+2].mean((1, 2))
channel = lambda n: objectives.channel("features_%d"%(layer_idx), n)
obj = sum([dirvect[iCh] * stdvect[iCh] * channel(iCh) for iCh in range(512)])
#%%
maxind = np.unravel_index(L1cc_map.argmax(), L1cc_map.shape)
dirvect = corrtsr[0,:,maxind[0]-1:maxind[0]+1,maxind[1]-1:maxind[1]+1].mean((1,2))
iCh = dirvect.argmax()
obj = channel(iCh)
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef"
from os.path import join
images = render.render_vis(VGG, obj)
plt.imsave(join(savedir, r"Alfa-Manif-Exp3-%04d_std.PNG"%np.random.randint(1000)),images[0][0,:])

param_f = lambda: param.image(128, fft=True, decorrelate=False)
images2 = render.render_vis(VGG, obj, param_f, transforms=[], show_inline=False)
plt.imsave(join(savedir, r"Alfa-Manif-Exp3-%04d_stdFFT.PNG"%np.random.randint(1000)),images2[0][0,:])
#%%
cppn_param_f = lambda: param.cppn(128)
# We initialize an optimizer with lower learning rate for CPPN
obj = objectives.channel("features_%d"%(layer_idx), 100)
cppn_opt = lambda params: torch.optim.Adam(params, 5e-2)
images = render.render_vis(VGG, obj, cppn_param_f, cppn_opt, transforms=[], show_inline=False)
#%%

#%% Visualize a local population!
maxind = np.unravel_index(L1cc_map.argmax(), L1cc_map.shape)
dirvect = corrtsr[0,:,maxind[0]-2:maxind[0]+2,maxind[1]-2:maxind[1]+2].mean((1,2))
iCh = dirvect.argmax()
#%%
param_f = lambda: param.image(256, fft=False, decorrelate=False)
obj = objectives.neuron("features_%d"%(layer_idx),iCh, x=maxind[1], y=maxind[0])

images = render.render_vis(VGG, obj, show_inline=False, thresholds=(512,))
plt.imsave(join(savedir,r"Alfa-Manif-Exp3-%04d_localvis_tfm.PNG"%np.random.randint(1000)),images[0][0,:])
#%% With all transformation (single unit objective seems incompatible with lots of transform)
all_transforms = [
    # transform.pad(16),
    transform.jitter(8),
    # transform.random_scale([n/100. for n in range(80, 120)]),
    # transform.random_rotate(list(range(-10,10)) + list(range(-5,5)) + 10*list(range(-2,2))),
    transform.jitter(2),
]
cppn_param_f = lambda: param.cppn(256)
# We initialize an optimizer with lower learning rate for CPPN
cppn_opt = lambda params: torch.optim.Adam(params, 5e-3)
obj = objectives.channel("features_%d"%(layer_idx),iCh)#, x=maxind[1], y=maxind[0]
images = render.render_vis(VGG, obj, cppn_param_f, cppn_opt, transforms=all_transforms, show_inline=False, thresholds=(512,))
plt.imsave(join(savedir,r"Alfa-Manif-Exp3-%04d_localvis_tfm.PNG"%np.random.randint(1000)),images[0][0,:])
#%%
from lucent.optvis.objectives_util import _extract_act_pos
def neuron_weight(layer, weight=None, x=None, y=None, batch=None):
    """ Linearly weighted channel activation at one location as objective
    weight: a torch Tensor vector same length as channel.
    """
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x, y)
        if weight is None:
            return -layer_t.mean()
        else:
            return -(layer_t.squeeze() * weight).mean()
    return inner

def channel_weight(layer, weight, batch=None):
    """ Linearly weighted channel activation as objective
    weight: a torch Tensor vector same length as channel. """
    def inner(model):
        layer_t = model(layer)
        return -(layer_t * weight.view(1, -1, 1, 1)).mean()
    return inner

def localgroup_weight(layer, weight=None, x=None, y=None, wx=1, wy=1, batch=None):
    """ Linearly weighted channel activation around some spot as objective
    weight: a torch Tensor vector same length as channel. """
    def inner(model):
        layer_t = model(layer)
        if weight is None:
            return -(layer_t[:, :, y:y + wy, x:x + wx]).mean()
        else:
            return -(layer_t[:, :, y:y+wy, x:x+wx] * weight.view(1,-1,1,1)).mean()
    return inner
#%%
maxind = np.unravel_index(L1cc_map.argmax(), L1cc_map.shape)
dirvect = corrtsr[0, :, maxind[0], maxind[1]]
obj = neuron_weight("features_%d"%(layer_idx), x=maxind[1], y=maxind[0], weight=torch.from_numpy(dirvect).cuda())#,iCh
# obj = channel_weight("features_%d"%(layer_idx), weight=torch.from_numpy(dirvect).cuda())
images = render.render_vis(VGG, obj, transforms=[], show_inline=False, thresholds=(512,))
plt.imsave(join(savedir,r"Alfa-Manif-Exp3-%04d_chan_vect.PNG"%np.random.randint(1000)),images[0][0,:])
#%% std will not rescue it.
stdvect = featStdtsr[:, maxind[0]-2:maxind[0]+2, maxind[1]-2:maxind[1]+2].mean((1, 2))
weight = torch.from_numpy(dirvect*stdvect).cuda()
obj = localgroup_weight("features_%d"%(layer_idx), x=maxind[1]-2, y=maxind[0]-2, wx=4, wy=4, weight=weight)
images = render.render_vis(VGG, obj, transforms=[], show_inline=False, thresholds=(512,))
plt.imsave(join(savedir,r"Alfa-Manif-Exp3-%04d_locpop_vectstd.PNG"%np.random.randint(1000)),images[0][0,:])
#%%
stdvect = featStdtsr[:, maxind[0]-2:maxind[0]+2, maxind[1]-2:maxind[1]+2].mean((1, 2))
weight = dirvect*stdvect
threshold = np.percentile(weight, (10, 90))
weight[np.logical_and(weight>threshold[0], weight<threshold[1])]=0
weight = torch.from_numpy(weight).cuda()
obj = localgroup_weight("features_%d"%(layer_idx), x=maxind[1]-2, y=maxind[0]-2, wx=4, wy=4, weight=weight)
images = render.render_vis(VGG, obj, transforms=[], show_inline=False, thresholds=(512,))
plt.imsave(join(savedir,r"Alfa-Manif-Exp3-%04d_locpop_vectstd_sparse.PNG"%np.random.randint(1000)),images[0][0,:])

#%%
dirvect = corrtsr[0, :, maxind[0]-2:maxind[0]+2, maxind[1]-2:maxind[1]+2].mean((1,2))
stdvect = featStdtsr[:, maxind[0]-2:maxind[0]+2, maxind[1]-2:maxind[1]+2].mean((1, 2))
weight = dirvect*stdvect
threshold = np.percentile(weight, (0, 95))
weight[np.logical_and(weight > threshold[0], weight < threshold[1])]=0
weight = torch.from_numpy(weight).cuda()
obj = localgroup_weight("features_%d"%(layer_idx), x=maxind[1]-2, y=maxind[0]-2, wx=4, wy=4, weight=weight)
all_transforms = [
    transform.pad(16),
    transform.jitter(8),
    transform.random_scale([n/100. for n in range(80, 120)]),
    transform.random_rotate(list(range(-10,10)) + list(range(-5,5)) + 10*list(range(-2,2))),
    transform.jitter(2),
]
param_f = lambda: param.image(256, fft=True, decorrelate=False)
images = render.render_vis(VGG, obj, param_f=param_f, transforms=all_transforms, show_inline=False, thresholds=(1024,))
plt.imsave(join(savedir,r"Alfa-Manif-Exp3-%04d_locpop_vectstd_sparsePos_tfm_FFT.PNG"%np.random.randint(1000)),images[0][0,:])

#%%
plt.plot(dirvect)
plt.plot(stdvect)
plt.plot(stdvect*dirvect)
plt.show()
#%% Visualize the correlation coef dist.
# plt.plot(dirvect)
# plt.plot(stdvect)
plt.plot(sorted(stdvect*dirvect))
plt.ylabel("corr coef * std(feat)")
plt.xlabel("sorted feature ")
plt.savefig(join(savedir, r"Alfa-Manif-Exp3-sorted_weight.png"))
plt.show()
plt.plot(sorted(dirvect))
plt.ylabel("corr coef")
plt.xlabel("sorted feature ")
plt.savefig(join(savedir, r"Alfa-Manif-Exp3-sorted_cc.png"))
plt.show()
#%% Load the fit the data
ccdata = loadmat(r"E:\OneDrive - Washington University in St. Louis\CNNFeatCorr\Beto_Evol_Exp11_conv5_3.mat")
cc_tsr = ccdata['cc_tsr']
#%%
from matplotlib import use as usebackend
import matplotlib.pylab as plt
# usebackend("Qt4Agg")
#%%
plt.figure(figsize=[5, 5])
plt.matshow(np.mean(np.abs(cc_tsr), axis=2)[:,:,14])
# plt.scatter([36], [32], c='red', s=50)
plt.axis('image'); plt.axis('off')
plt.colorbar()
plt.show()
# np.sum(np.abs(cc_tsr),axis=2)[:,:,9]
#%%
corrheatmap = np.mean(np.abs(cc_tsr), axis=2)[:, :, 11]
sortidx = np.argsort(-corrheatmap, axis=None)
np.unravel_index(sortidx[:10], corrheatmap.shape)
#%%
center = [32,36]
#%%
plt.plot(np.mean(cc_tsr[32-3:32+3,36-3:36+3, :, 11],axis=(0,1)))
plt.show()
#%%
from lucent.optvis import render, param, transform, objectives
from matver_CNN import getMatVGG, lucent_layernames
matvgg = getMatVGG()
layernames = lucent_layernames(matvgg)
matvgg.cuda()
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef"
from os.path import join
#%%
# maxind = np.unravel_index(L1cc_map.argmax(), L1cc_map.shape)
dirvect = np.mean(cc_tsr[32-3:32+3,36-3:36+3,:,11],axis=(0,1))
layer_idx = 10
# obj = neuron_weight("features_%d"%(layer_idx), x=maxind[1], y=maxind[0], weight=torch.from_numpy(dirvect).cuda())#,iCh
# obj = channel_weight("features_%d"%(layer_idx), weight=torch.from_numpy(dirvect).cuda())
obj = objectives.channel("features_%d"%(layer_idx),n_channel=dirvect.argmax())
images = render.render_vis(matvgg, obj, transforms=[], show_inline=False, preprocess=False, thresholds=(512,))
plt.imsave(join(savedir, r"Beto-Evol-Exp11-%04d_chan_vect.PNG"%np.random.randint(1000)),images[0][0,:])
#%%
RGBmean = torch.tensor([123.6800, 116.7790, 103.9390]).cuda().reshape([1, 3, 1, 1])
def matVGG_preprocess(x):
    return 255 * x - RGBmean
#%%
# obj = objectives.channel("features_%d" % (layer_idx), n_channel=dirvect.argmax())
weight_vec = dirvect
threshold = np.percentile(dirvect,90)
weight_vec[weight_vec < threshold] = 0
weight_vec = torch.tensor(weight_vec).cuda()
# obj = neuron_weight("features_%d" % (layer_idx), weight=weight_vec, x=36, y=32, batch=None)
obj = localgroup_weight("features_%d" % (layer_idx), weight=weight_vec, x=34, y=30, wx=5, wy=5, batch=None)
images = render.render_vis(matvgg, obj, show_inline=False, preprocess=False, transforms=[matVGG_preprocess], thresholds=(256,))
#%%
perm_idx = torch.randperm(256)
obj = neuron_weight("features_%d" % (layer_idx), weight=weight_vec[perm_idx], x=36, y=32, batch=None)
images = render.render_vis(matvgg, obj, show_inline=False, preprocess=False, transforms=[matVGG_preprocess], thresholds=(128,))
#%%
from lucent.optvis.transform import pad, jitter, random_rotate, random_scale
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),
        matVGG_preprocess]
obj = objectives.channel("features_%d" % (2), n_channel=12, batch=None)
#%%
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),
        matVGG_preprocess]
obj = objectives.channel("classifier_%d" % (6), n_channel=949, batch=None)
param_f = lambda: param.image(224, fft=False, decorrelate=True)
images = render.render_vis(matvgg, obj, param_f, show_inline=False, preprocess=False, transforms=tfms, thresholds=(64, 128, 192, 256,),verbose=True)
#%% VGG pytorch as reference
VGG = models.vgg16(pretrained=True)
#%%
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),]
images = render.render_vis(VGG.cuda(), obj, show_inline=False, preprocess=True, transforms=[], thresholds=(256,))
#%% CPPN parametrization to find optimal image for the lemon neurons
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),]
cppn_param_f = lambda: param.cppn(128)
# We initialize an optimizer with lower learning rate for CPPN
cppn_opt = lambda params: torch.optim.Adam(params, 5e-3) # seems 5E-3 is the sweet spot for
# obj = objectives.channel("features_%d"%(layer_idx),iCh)#, x=maxind[1], y=maxind[0]
obj = objectives.channel("classifier_%d" % (6), n_channel=949, batch=None)
images = render.render_vis(VGG, obj, cppn_param_f, cppn_opt, transforms=tfms, show_inline=False, thresholds=(64,128,192,256),verbose=True)
#%% Note you should not reuse tfms across exp or extra staff will attach to it and generate artifacts!
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),]
param_f = lambda: param.image(128, fft=True, decorrelate=True)
fft_opt = lambda params: torch.optim.Adam(params, 10e-2)
obj = objectives.channel("classifier_%d" % (6), n_channel=951, batch=None)
images = render.render_vis(VGG, obj, param_f, fft_opt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256), verbose=True)
