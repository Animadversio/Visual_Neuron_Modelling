import torch
import sys
sys.path.append(r"E:\DL_Projects\Vision\deep-image-prior")
from models import skip
INPUT = 'noise'
input_depth = 32
pad = 'reflection'
net = skip(input_depth, 3, num_channels_down = [16, 32, 64, 128, 128, 128],
                           num_channels_up =   [16, 32, 64, 128, 128, 128],
                           num_channels_skip = [0, 4, 4, 4, 4, 4],
                           filter_size_down = [5, 3, 5, 5, 3, 5], filter_size_up = [5, 3, 5, 3, 5, 3],
                           upsample_mode='bilinear', downsample_mode='avg',
                           need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(torch.cuda.FloatTensor)
import numpy as np
s = sum(np.prod(list(p.size())) for p in net.parameters())
print('Number of params: %d' % s)
# Number of params: 2948567
torch.save(net.state_dict(),r"C:\Users\binxu\.cache\torch\hub\checkpoints\tmp.pt")
# Size on disk 11,583 KB
#%%
def simple_montage(images, shape=(2,2)):
    if shape == (2, 2):
        row1 = np.concatenate(tuple(images[:2, ]), axis=1)
        row2 = np.concatenate(tuple(images[2:, ]), axis=1)
        whol = np.concatenate((row1, row2), axis=0)
        return whol
    elif shape == (2, 3):
        row1 = np.concatenate(tuple(images[:3, ]), axis=1)
        row2 = np.concatenate(tuple(images[3:, ]), axis=1)
        whol = np.concatenate((row1, row2), axis=0)
        return whol
    elif shape == (1, 2):
        return np.concatenate(tuple(images[:, ]), axis=1)
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
VGG = models.vgg16(pretrained=True)
layernames = lucent_layernames(VGG)
VGG.cuda()
#%%
Categ_list = [1, *list(range(109, 130)), 340, 345, *list(range(356,385)), *list(range(456, 481)), 486, 629, *list(range(646, 664)), 949, 950, 951, 967, 985, 986]
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef\CNN_ref\DeepImagePrior"
def DIPr_param(imsize=256, batch=1, sd=1):
    net = skip(input_depth, 3, num_channels_down=[16, 32, 64, 128, 128, 128],
               num_channels_up=[16, 32, 64, 128, 128, 128],
               num_channels_skip=[0, 4, 4, 4, 4, 4],
               filter_size_down=[5, 3, 5, 5, 3, 5], filter_size_up=[5, 3, 5, 3, 5, 3],
               upsample_mode='bilinear', downsample_mode='avg',
               need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(torch.cuda.FloatTensor)
    net.requires_grad_(True)

    noise = (torch.randn((batch, 32, imsize, imsize)) * sd).to("cuda").requires_grad_(False)
    imagef = lambda:  net(noise)
    return list(net.parameters()), imagef
#%

for iCh in Categ_list[1:]:
    DivW = .5e1
    tfms = [pad(12, mode="constant", constant_value=.5),
            jitter(8),
            random_scale([1 + (i - 5) / 50. for i in range(11)]),
            random_rotate(list(range(-10, 11)) + 5 * [0]),
            jitter(4),]
    DIPrparam_f = lambda: DIPr_param(batch=2, sd=1)
    DIPopt = lambda params: torch.optim.Adam(params, lr=0.001)  # 5E-2 looks good
    obj = objectives.channel('classifier_6', n_channel=iCh) - DivW * objectives.diversity('features_15')  # optimizing the
    images = render.render_vis(VGG, obj, DIPrparam_f, DIPopt, transforms=tfms, show_inline=False, thresholds=(64, 128, 256, ), verbose=True)
    RND = np.random.randint(1000)
    img_mtg = simple_montage(images[-1], shape=(1, 2))
    imsave(join(savedir, r"vgg16-fc-ch%04d-DIPr-slr-Div%.1f-tfm-%04d.PNG" % (iCh, DivW, RND)), img_mtg)
#%%
# imsave(join("E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef\CNN_ref\DeepImagePrior", r"vgg16-fc-ch%04d-DIPr-slr-tfm.PNG" % (iCh)), images[-1][0,:])
# img_mtg = simple_montage(images[-3])
# imsave(join("E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef\CNN_ref\DeepImagePrior", r"vgg16-fc-ch%04d-DIPr-3.PNG" % (iCh)), img_mtg)
import pickle, urllib
imageNet_category = pickle.load(urllib.request.urlopen('https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'))
imageNet_category = pickle.load(open(r"resource/imagenet_class.pkl", "rb"))
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef\CNN_ref\DeepImagePrior"
net = skip(input_depth, 3, num_channels_down=[16, 32, 64, 128, 128, 128],
               num_channels_up=[16, 32, 64, 128, 128, 128],
               num_channels_skip=[0, 4, 4, 4, 4, 4],
               filter_size_down=[5, 3, 5, 5, 3, 5], filter_size_up=[5, 3, 5, 3, 5, 3],
               upsample_mode='bilinear', downsample_mode='avg',
               need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(torch.cuda.FloatTensor)
net.requires_grad_(True)
def DIPrRAND_param(imsize=256, batch=1, sd=1):
    imagef = lambda:  net((torch.randn((batch, 32, imsize, imsize)) * sd).to("cuda").requires_grad_(False))
    return list(net.parameters()), imagef

DivW = 1e1
iCh = 629
DIPrRAND_param_f = lambda: DIPrRAND_param(batch=6, sd=1)
DIPopt = lambda params: torch.optim.Adam(params, lr=0.001)  # 5E-2 looks good
obj = objectives.channel('classifier_6', n_channel=iCh) - DivW * objectives.diversity('features_15')  # optimizing the
images = render.render_vis(VGG, obj, DIPrRAND_param_f, DIPopt, transforms=[], show_inline=False, thresholds=(0,   64,  128,  192,  256,  320,  384,  448,  512,  576,  640, 704,  768,  832,  896,  960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048), verbose=True)
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef\CNN_ref\DeepImagePrior\vggGAN"
for i, img_group in enumerate(images):
    imsave(join(savedir, "vggGAN%04dlipstick_%d.png"%(iCh, i)), simple_montage(img_group, (2, 3)))

#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef\CNN_ref\DeepImagePrior"
net = skip(input_depth, 3, num_channels_down=[16, 32, 64, 128, 128, 128],
               num_channels_up=[16, 32, 64, 128, 128, 128],
               num_channels_skip=[0, 4, 4, 4, 4, 4],
               filter_size_down=[5, 3, 5, 5, 3, 5], filter_size_up=[5, 3, 5, 3, 5, 3],
               upsample_mode='bilinear', downsample_mode='avg',
               need_sigmoid=True, pad=pad, act_fun='LeakyReLU').type(torch.cuda.FloatTensor)
net.requires_grad_(True)

def DIPrRAND_param(imsize=256, batch=1, sd=1):
    imagef = lambda:  net((torch.randn((batch, 32, imsize, imsize)) * sd).to("cuda").requires_grad_(False))
    return list(net.parameters()), imagef

DivW = .5e1
iCh = 629
DIPrRAND_param_f = lambda: DIPrRAND_param(batch=6, sd=1)
DIPopt = lambda params: torch.optim.Adam(params, lr=0.001, weight_decay=0.001)  # 5E-2 looks good
obj = objectives.channel('classifier_6', n_channel=iCh) - DivW * objectives.diversity('features_28')  # optimizing the
images = render.render_vis(VGG, obj, DIPrRAND_param_f, DIPopt, transforms=[], show_inline=True, thresholds=list(range(0, 2050, 64)), verbose=True)
#%%
savedir = r"E:\OneDrive - Washington University in St. Louis\InterpretCorrCoef\CNN_ref\DeepImagePrior\vggGAN"
for i, img_group in enumerate(images):
    imsave(join(savedir, "vggGAN%04dlipstick_HDiv%.1f_%d.png" % (iCh, DivW, i)), simple_montage(img_group, (2, 3)))
#%%
