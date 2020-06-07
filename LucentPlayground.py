#%%
from torchvision import models
from torchvision import transforms
import torch
from lucent.optvis.transform import pad, jitter, random_rotate, random_scale
from lucent.optvis import render, param, transform, objectives
from matver_CNN import getMatVGG, lucent_layernames
from imageio import imsave
from os.path import join
import numpy as np
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
savedir = r"E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef\CNN_ref\misc"
#%% Note DenseNet169
DSNet = models.densenet169(pretrained=True).cuda()
DSNet.eval()
layernames = lucent_layernames(DSNet)
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
#%%
VGG = models.vgg16(pretrained=True)
layernames = lucent_layernames(VGG)
VGG.cuda()
#%% Use GAN as parametrization
from GAN_utils import upconvGAN
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
#%% Diversity Visualization with GAN
G = upconvGAN("pool5")
G.requires_grad_(False)
G.cuda()
def GANparam(batch=1, sd=1):
    code = (torch.randn((batch, G.codelen, 6, 6)) * sd).to("cuda").requires_grad_(True)
    imagef = lambda:  G.visualize(code)
    return [code], imagef
#%%
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
#%% This can work~
G = upconvGAN("pool5")
# G.G.load_state_dict(torch.hub.load_state_dict_from_url(r"https://drive.google.com/uc?export=download&id=1vB_tOoXL064v9D6AKwl0gTs1a7jo68y7",progress=True))
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
model_urls = {"pool5" : "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145337&authkey=AFaUAgeoIg0WtmA",
    "fc6": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145339&authkey=AC2rQMt7Obr0Ba4",
    "fc7": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145338&authkey=AJ0R-daUAVYjQIw",
    "fc8": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145340&authkey=AKIfNk7s5MGrRkU"}
G = upconvGAN("fc7")
G.G.load_state_dict(torch.hub.load_state_dict_from_url(model_urls["fc7"], progress=True))
torchhome = torch.hub._get_torch_home()
ckpthome = join(torchhome, "checkpoints")
os.makedirs(ckpthome, exist_ok=True)
torch.hub.download_url_to_file(model_urls["fc7"], join(ckpthome, "upconvGAN_fc7.pt"), hash_prefix=None, progress=True)