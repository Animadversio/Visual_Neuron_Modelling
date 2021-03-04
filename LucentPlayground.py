#%%
from torchvision import models
from torchvision import transforms
import torch
from lucent.optvis.transform import pad, jitter, random_rotate, random_scale
from lucent.optvis import render, param, transform, objectives
from matver_CNN import getMatVGG, lucent_layernames
from GAN_utils import upconvGAN
from imageio import imsave
from os.path import join
import numpy as np
import matplotlib.pylab as plt
def simple_montage(images, shape=(2,2)):
    row1 = np.concatenate(tuple(images[:2, ]), axis=1)
    row2 = np.concatenate(tuple(images[2:, ]), axis=1)
    whol = np.concatenate((row1, row2), axis=0)
    return whol
#%%
matvgg = getMatVGG()
layernames = lucent_layernames(matvgg)
matvgg.cuda()
#%%
VGG = models.vgg16(pretrained=True)
layernames = lucent_layernames(VGG)
VGG.cuda()
#%% Note DenseNet169
DSNet = models.densenet169(pretrained=True).cuda()
DSNet.eval()
layernames = lucent_layernames(DSNet)
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef\CNN_ref\misc"
#%%
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),]
param_f = lambda: param.image(224, fft=True, decorrelate=True)
fft_opt = lambda params: torch.optim.Adam(params, 5e-2) # 5E-2 looks good
iCh = 372
obj = objectives.channel('classifier', n_channel=iCh, batch=None)
images = render.render_vis(DSNet, obj, param_f, fft_opt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256, 384), verbose=True) #, 384, 512
imsave(join(savedir, r"densenet169-fc-ch%04d-FFT-%04d.PNG"%(iCh, np.random.randint(1000))), images[-1][0,:])
#%% Try out batch visualization and optimization
batch = 2
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),]
# param_f = lambda: param.image(224, fft=True, decorrelate=True)
batch_param_f = lambda: param.image(224, fft=True, batch=batch) # not using FFT make the image plain and harder to optimize....
fft_opt = lambda params: torch.optim.Adam(params, 5e-2) # 5E-2 looks good
iCh = 949
obj = objectives.channel('classifier', n_channel=iCh) - 1e2 * objectives.diversity("features_denseblock3")
images = render.render_vis(DSNet, obj, batch_param_f, fft_opt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256), verbose=True) #, 384, 512
RND = np.random.randint(1000)
mtgimage = np.concatenate(tuple(images[-1][:, :]), axis=1)
imsave(join(savedir, r"densenet169-fc-ch%04d-DivDB3-%04d.PNG" % (iCh, RND)), mtgimage)
# RND = np.random.randint(1000)
# for B in range(batch):
#     imsave(join(savedir, r"densenet169-fc-ch%04d-FFTDivDB3-%04d-%d.PNG" % (iCh, RND, B)), images[-1][B, :])
#%%
batch = 3
batch_param_f = lambda: param.image(224, fft=True, batch=batch)
fft_opt = lambda params: torch.optim.Adam(params, 5e-2)
for iCh in range(275, 400):
    tfms = [pad(12, mode="constant", constant_value=.5),
            jitter(8),
            random_scale([1 + (i - 5) / 50. for i in range(11)]),
            random_rotate(list(range(-10, 11)) + 5 * [0]),
            jitter(4),]
    obj = objectives.channel('classifier', n_channel=iCh) - 1e2 * objectives.diversity("features_denseblock3")
    images = render.render_vis(DSNet, obj, batch_param_f, fft_opt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256), verbose=True)
    RND = np.random.randint(1000)
    mtgimage = np.concatenate(tuple(images[-1][:, :]), axis=1)
    imsave(join(savedir, r"densenet169-fc-ch%04d-FFTDivDB3-%04d.PNG" % (iCh, RND)), mtgimage)

#%% Add a relatively large diversity objective to do multi-faceted visualization for each target
batch = 4
fft_opt = lambda params: torch.optim.Adam(params, 5e-2) # 5E-2 looks good
for iCh in range(275, 400):
    # param_f = lambda: param.image(128, batch=2, fft=True, decorrelate=False)
    batch_param_f = lambda: param.image(128, batch=batch)
    obj = objectives.channel('classifier', n_channel=iCh, batch=batch) - .5e2 * objectives.diversity("features_denseblock3")
    tfms = [pad(12, mode="constant", constant_value=.5),
            jitter(8),
            random_scale([1 + (i - 5) / 50. for i in range(11)]),
            random_rotate(list(range(-10, 11)) + 5 * [0]),
            jitter(4),]
    images = render.render_vis(DSNet, obj, batch_param_f, fft_opt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256, 384), verbose=True) #, 384, 512
    RND = np.random.randint(1000)
    for B in range(batch):
        imsave(join(savedir, r"densenet169-fc-ch%04d-FFTdiv-%04d-%d.PNG" % (iCh, RND, B)), images[-1][B, :])
        imsave(join(savedir, r"densenet169-fc-ch%04d-FFTdiv-%04d-%d-1.PNG" % (iCh, RND, B)), images[-2][B, :])
#%%
MBLnet = models.mobilenet_v2(pretrained=True).cuda()
MBLnet.eval()
layernames = lucent_layernames(MBLnet)
#%%
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),]
param_f = lambda: param.image(128, fft=True, decorrelate=True)
fft_opt = lambda params: torch.optim.Adam(params, 5e-2)  # 5E-2 looks good
obj = objectives.channel('classifier_1', n_channel=1, batch=None)  # optimizing the last layer is quite feasible ()
# obj = objectives.channel('features_8_conv_2', n_channel=5, batch=None)
images = render.render_vis(MBLnet, obj, param_f, fft_opt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256, 384, 512), verbose=True)

#%% Use GAN as parametrization
G = upconvGAN("fc7")
G.requires_grad_(False)
G.cuda()
def GANparam(batch=1, sd=1):
    code = (torch.randn((batch, G.codelen)) * sd).to("cuda").requires_grad_(True)
    imagef = lambda:  G.visualize(code)
    return [code], imagef
#%%
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),]
GANparam_f = lambda: GANparam(batch=2, sd=1)
GANopt = lambda params: torch.optim.Adam(params, 5e-2)  # 5E-2 looks good
obj = objectives.channel('classifier_6', n_channel=950, batch=None)  # optimizing the
images = render.render_vis(VGG, obj, GANparam_f, GANopt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256, 384), verbose=True)
#%% Diversity Visualization with GAN
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),]
iCh = 949
GANparam_f = lambda: GANparam(batch=4, sd=1)
GANopt = lambda params: torch.optim.Adam(params, 5e-2)  # 5E-2 looks good
obj = objectives.channel('classifier_6', n_channel=iCh, batch=None) #- .25e1 * objectives.diversity('features_15')  # optimizing the
images = render.render_vis(VGG, obj, GANparam_f, GANopt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256, 384), verbose=True)
img_mtg = simple_montage(images[-1])
# obj = objectives.channel('classifier', n_channel=iCh) - 1e2 * objectives.diversity("features_denseblock3")
#%%
#%% Diversity Visualization with GAN pool5
G = upconvGAN("pool5")
G.requires_grad_(False)
G.cuda()
def GANparam(batch=1, sd=1):
    code = (torch.randn((batch, G.codelen, 6, 6)) * sd).to("cuda").requires_grad_(True)
    imagef = lambda:  G.visualize(code)
    return [code], imagef
#%
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),]
iCh = 950
GANparam_f = lambda: GANparam(batch=4, sd=4)
GANopt = lambda params: torch.optim.Adam(params, 5e-2)  # 5E-2 looks good
obj = objectives.channel('classifier_6', n_channel=iCh, batch=None) #- .25e1 * objectives.diversity('features_15')  # optimizing the
images = render.render_vis(VGG, obj, GANparam_f, GANopt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256, 384), verbose=True)
img_mtg = simple_montage(images[-1])

#%% Diversity Visualization with GAN fc8
G = upconvGAN("fc8").requires_grad_(False).cuda()
def GANparam(batch=1, sd=1):
    code = (torch.randn((batch, G.codelen)) * sd).to("cuda").requires_grad_(True)
    imagef = lambda:  G.visualize(code)
    return [code], imagef
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),]
iCh = 459
GANparam_f = lambda: GANparam(batch=4, sd=0.2)
GANopt = lambda params: torch.optim.Adam(params, 5e-2)  # 5E-2 looks good
obj = objectives.channel('classifier_6', n_channel=iCh, batch=None) - .1e1 * objectives.diversity('features_15')  # optimizing the
images = render.render_vis(VGG, obj, GANparam_f, GANopt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256,), verbose=True)
img_mtg = simple_montage(images[-1])

#%%
Categ_list = [1, *list(range(109, 130)), 340, 345, *list(range(356,385)), *list(range(456, 481)), 486, 629, *list(range(646, 664)), 949, 950, 951, 967, 985, 986]
savedir = r"E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef\CNN_ref\fc7gan"
G = upconvGAN("fc7")
G.requires_grad_(False).cuda()
def GANparam(batch=1, sd=1):
    code = (torch.randn((batch, G.codelen)) * sd).to("cuda").requires_grad_(True)
    imagef = lambda:  G.visualize(code)
    return [code], imagef
divL = .2e1
for iCh in Categ_list:
    RND = torch.randint(1000, (1,)).item()
# iCh = 356
    tfms = [pad(12, mode="constant", constant_value=.5),
            jitter(8),
            random_scale([1 + (i - 5) / 50. for i in range(11)]),
            random_rotate(list(range(-10, 11)) + 5 * [0]),
            jitter(4),]
    GANparam_f = lambda: GANparam(batch=4, sd=1)
    GANopt = lambda params: torch.optim.Adam(params, 5e-2)  # 5E-2 looks good
    obj = objectives.channel('classifier_6', n_channel=iCh, batch=None) - divL * objectives.diversity('features_15')  #  optimizing the
    images = render.render_vis(VGG, obj, GANparam_f, GANopt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256, ), verbose=True)
    img_mtg = simple_montage(images[-1])
    imsave(join(savedir, r"vgg16-fc-ch%04d-fc7GAN-tfmdiv%.1f-%04d.PNG" % (iCh, divL, RND)), img_mtg)

#
G = upconvGAN("fc6")
G.requires_grad_(False).cuda()
def GANparam(batch=1, sd=1):
    code = (torch.randn((batch, G.codelen)) * sd).to("cuda").requires_grad_(True)
    imagef = lambda:  G.visualize(code)
    return [code], imagef
divL = .2e1
for iCh in Categ_list:
    RND = torch.randint(1000, (1,)).item()
# iCh = 356
    tfms = [pad(12, mode="constant", constant_value=.5),
            jitter(8),
            random_scale([1 + (i - 5) / 50. for i in range(11)]),
            random_rotate(list(range(-10, 11)) + 5 * [0]),
            jitter(4),]
    GANparam_f = lambda: GANparam(batch=4, sd=1)
    GANopt = lambda params: torch.optim.Adam(params, 5e-2)  # 5E-2 looks good
    obj = objectives.channel('classifier_6', n_channel=iCh, batch=None) - divL * objectives.diversity('features_15')  #  optimizing the
    images = render.render_vis(VGG, obj, GANparam_f, GANopt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256, ), verbose=True)
    img_mtg = simple_montage(images[-1])
    imsave(join(savedir, r"vgg16-fc-ch%04d-fc6GAN-tfmdiv%.1f-%04d.PNG" % (iCh, divL, RND)), img_mtg)

#
G = upconvGAN("pool5")
G.requires_grad_(False).cuda()
def GANparam(batch=1, sd=1):
    code = (torch.randn((batch, G.codelen, 6, 6)) * sd).to("cuda").requires_grad_(True)
    imagef = lambda:  G.visualize(code)
    return [code], imagef
divL = .2e1
for iCh in Categ_list: # iCh = 356
    RND = torch.randint(1000, (1,)).item()
    tfms = [pad(12, mode="constant", constant_value=.5),
            jitter(8),
            random_scale([1 + (i - 5) / 50. for i in range(11)]),
            random_rotate(list(range(-10, 11)) + 5 * [0]),
            jitter(4),]
    GANparam_f = lambda: GANparam(batch=4, sd=5)
    GANopt = lambda params: torch.optim.Adam(params, 5e-2)  # 5E-2 looks good
    obj = objectives.channel('classifier_6', n_channel=iCh, batch=None) - divL * objectives.diversity('features_15')  #  optimizing the
    images = render.render_vis(VGG, obj, GANparam_f, GANopt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256, ), verbose=True)
    img_mtg = simple_montage(images[-1])
    imsave(join(savedir, r"vgg16-fc-ch%04d-pl5GAN-tfmdiv%.1f-%04d.PNG" % (iCh, divL, RND)), img_mtg)
#%%
G = upconvGAN("fc8")
G.requires_grad_(False).cuda()
def GANparam(batch=1, sd=1):
    code = (torch.randn((batch, G.codelen)) * sd).to("cuda").requires_grad_(True)
    imagef = lambda:  G.visualize(code)
    return [code], imagef
divL = .2e1
for iCh in Categ_list:
    RND = torch.randint(1000, (1,)).item()
# iCh = 356
    tfms = [pad(12, mode="constant", constant_value=.5),
            jitter(8),
            random_scale([1 + (i - 5) / 50. for i in range(11)]),
            random_rotate(list(range(-10, 11)) + 5 * [0]),
            jitter(4),]
    GANparam_f = lambda: GANparam(batch=4, sd=1)
    GANopt = lambda params: torch.optim.Adam(params, 5e-2)  # 5E-2 looks good
    obj = objectives.channel('classifier_6', n_channel=iCh, batch=None) - divL * objectives.diversity('features_15')  #  optimizing the
    images = render.render_vis(VGG, obj, GANparam_f, GANopt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256, ), verbose=True)
    img_mtg = simple_montage(images[-1])
    imsave(join(savedir, r"vgg16-fc-ch%04d-fc8GAN-tfmdiv%.1f-%04d.PNG" % (iCh, divL, RND)), img_mtg)

#%%
import pickle, urllib
imageNet_category = pickle.load(urllib.request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))
#%%
# from GAN_utils import multiZupconvGAN
mG = multiZupconvGAN(blendlayer="conv4_1", name="fc7",).requires_grad_(False).cuda()
def mzGANparam(batch=1, z_num=30, sd=1):
    code = (torch.randn((batch, z_num, 4096)) * sd).to("cuda").requires_grad_(True)
    alpha_z = (torch.full((batch, z_num, mG.c_num), 1 / z_num)).to("cuda").requires_grad_(True)
    imagef = lambda:  mG.visualize(code, alpha_z)
    return [code, alpha_z], imagef
#%%
tfms = [pad(12, mode="constant", constant_value=.5),
        jitter(8),
        random_scale([1 + (i - 5) / 50. for i in range(11)]),
        random_rotate(list(range(-10, 11)) + 5 * [0]),
        jitter(4),]
iCh = 459
mzGANparam_f = lambda: mzGANparam(batch=1, z_num=1, sd=0.6)
GANopt = lambda params: torch.optim.Adam(params, 5e-2)  # 5E-2 looks good
obj = objectives.channel('classifier_6', n_channel=iCh, batch=None) #- .1e1 * objectives.diversity('features_15')  # optimizing the
images = render.render_vis(VGG, obj, mzGANparam_f, GANopt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256,), verbose=True)
# img_mtg = simple_montage(images[-1])
plt.imshow(images[-1].squeeze());plt.axis("off"); plt.show()
