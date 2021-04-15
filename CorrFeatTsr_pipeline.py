# Alias the disks for usage
# !subst N: E:\Network_Data_Sync
# !subst S: E:\Network_Data_Sync
# !subst O: "E:\OneDrive - Washington University in St. Louis"
#%%
# %load_ext autoreload
# %autoreload 2
#%%
from scipy.io import loadmat
from skimage.io import imread, imread_collection
from os.path import join
from glob import glob
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pylab as plt
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from skimage.transform import resize
from kornia.filters import median_blur, gaussian_blur2d
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
Pasupath = r"N:\Stimuli\2019-Manifold\pasupathy-wg-f-4-ori"
Gaborpath = r"N:\Stimuli\2019-Manifold\gabor"
#% Loading the tool kit from the lib
from CorrFeatTsr_lib import visualize_cctsr, visualize_cctsr_embed, Corr_Feat_Machine, Corr_Feat_pipeline
# Define image loading and pre precessing functions
from CorrFeatTsr_lib import loadimg_preprocess, loadimg_embed_preprocess, preprocess
# Load full path to images and psths
from data_loader import load_score_mat
ccdir = "S:\corrFeatTsr"
ckpt_path = r"E:\Cluster_Backup\torch"

#%% Final Experiment Pipeline
VGG = models.vgg16(pretrained=True)
VGG.requires_grad_(False).eval()
VGG.features.cuda()
figdir = join("S:\corrFeatTsr", "VGGsummary")
expsuffix = "_nobdr"
layers2plot = ["conv5_3", "conv4_3",  "conv3_3", "conv2_2", ]
Animal = "Alfa"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
for Expi in range(1, len(EStats)+1):
    imgsize = EStats[Expi-1].evol.imgsize
    imgpos = EStats[Expi-1].evol.imgpos
    pref_chan = EStats[Expi-1].evol.pref_chan
    imgpix = int(imgsize * 40)
    titstr = "Driver Chan %d, %.1f deg [%s]"%(pref_chan, imgsize, tuple(imgpos))
    featFetcher = Corr_Feat_Machine()
    featFetcher.register_hooks(VGG, ["conv2_2", "conv3_3","conv4_3", "conv5_3"])
    featFetcher.init_corr()
    # Load Evol data and fit 
    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)], stimdrive="S")
    Corr_Feat_pipeline(VGG.features, featFetcher, score_vect, imgfullpath_vect,
        lambda x:loadimg_preprocess(x, borderblur=True, imgpix=imgpix), online_compute=True,
        batchsize=121, savedir=ccdir, savenm="%s_Exp%d_Evol%s" % (Animal, Expi, expsuffix), )
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "Evol%s"%expsuffix, titstr, figdir=figdir)
    # Load Manifold data and fit 
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50,200)], stimdrive="S")
    Corr_Feat_pipeline(VGG.features, featFetcher, scorecol_M, imgfullpath_vect_M,
           lambda x: loadimg_preprocess(x, borderblur=True, imgpix=imgpix), online_compute=True,
       batchsize=121, savedir=ccdir, savenm="%s_Exp%d_EM%s" % (Animal, Expi, expsuffix), )
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "EM_sgtr%s"%expsuffix, titstr, figdir=figdir)
    featFetcher.clear_hook()
#%% Final Experiment Pipeline for AlexNet.
Anet = models.alexnet(pretrained=True)
Anet.requires_grad_(False).eval()
Anet.features.cuda()
featNet = Anet.features
figdir = join("S:\corrFeatTsr", "Alexsummary")
expsuffix = "_nobdr_alex"
layers2plot = ["conv2", "conv3", "conv4", "conv5"]
Animal = "Beto"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
for Expi in range(1, len(EStats)+1):
    imgsize = EStats[Expi-1].evol.imgsize
    imgpos = EStats[Expi-1].evol.imgpos
    pref_chan = EStats[Expi-1].evol.pref_chan
    imgpix = int(imgsize * 40)
    titstr = "Driver Chan %d, %.1f deg [%s]"%(pref_chan, imgsize, tuple(imgpos))
    featFetcher = Corr_Feat_Machine()
    featFetcher.register_hooks(Anet, ["conv2", "conv3","conv4", "conv5"], netname="alexnet")
    featFetcher.init_corr()
    # Load Evol data and fit 
    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)], stimdrive="S")
    Corr_Feat_pipeline(Anet.features, featFetcher, score_vect, imgfullpath_vect,
        lambda x:loadimg_preprocess(x, borderblur=True, imgpix=imgpix), online_compute=True,
        batchsize=121, savedir=ccdir, savenm="%s_Exp%d_Evol%s" % (Animal, Expi, expsuffix), )
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "Evol%s"%expsuffix, titstr, figdir=figdir)
    # Load Manifold data and fit 
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50,200)], stimdrive="S")
    Corr_Feat_pipeline(Anet.features, featFetcher, scorecol_M, imgfullpath_vect_M,
           lambda x: loadimg_preprocess(x, borderblur=True, imgpix=imgpix), online_compute=True,
       batchsize=121, savedir=ccdir, savenm="%s_Exp%d_EM%s" % (Animal, Expi, expsuffix), )
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "EM_sgtr%s"%expsuffix, titstr, figdir=figdir)
    featFetcher.clear_hook()

#%%
ckpt_path = r"E:\Cluster_Backup\torch"
net = models.resnet50(pretrained=True)
net.requires_grad_(False).cuda()
net.load_state_dict(torch.load(join(ckpt_path, "imagenet_linf_8_pure.pt")))
featnet = net
figdir = join("S:\corrFeatTsr", "resnet-robust_summary")
expsuffix = "_nobdr_res-robust"
layers2plot = ["layer1", "layer2", "layer3", "layer4", ]

Animal = "Beto"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
for Expi in range(1, len(EStats)+1):
    imgsize = EStats[Expi-1].evol.imgsize
    imgpos = EStats[Expi-1].evol.imgpos
    pref_chan = EStats[Expi-1].evol.pref_chan
    imgpix = int(imgsize * 40)
    titstr = "Driver Chan %d, %.1f deg [%s]"%(pref_chan, imgsize, tuple(imgpos))
    featFetcher = Corr_Feat_Machine()
    featFetcher.register_hooks(net, ["layer1", "layer2", "layer3", "layer4", ], netname="resnet50_linf")
    featFetcher.init_corr()
    # Load Evol data and fit
    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)], stimdrive="S")
    Corr_Feat_pipeline(featnet, featFetcher, score_vect, imgfullpath_vect,
        lambda x:loadimg_preprocess(x, borderblur=True, imgpix=imgpix), online_compute=True,
        batchsize=121, savedir=ccdir, savenm="%s_Exp%d_Evol%s" % (Animal, Expi, expsuffix), )
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "Evol%s"%expsuffix, titstr, figdir=figdir)
    # Load Manifold data and fit
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50,200)], stimdrive="S")
    Corr_Feat_pipeline(featnet, featFetcher, scorecol_M, imgfullpath_vect_M,
           lambda x: loadimg_preprocess(x, borderblur=True, imgpix=imgpix), online_compute=True,
       batchsize=121, savedir=ccdir, savenm="%s_Exp%d_EM%s" % (Animal, Expi, expsuffix), )
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "EM_sgtr%s"%expsuffix, titstr, figdir=figdir)
    featFetcher.clear_hook()

#%% Older version VGG16 pipeline without wrap
VGG = models.vgg16(pretrained=True)
VGG.requires_grad_(False)
VGG.features.cuda()
online_compute = True
batchsize = 121
figdir = join("S:\corrFeatTsr", "VGGsummary")
layers2plot = ["conv5_3", "conv4_3",  "conv3_3", "conv2_2", ]
for Expi in range(1, len(EStats)+1):
    imgsize = EStats[Expi-1].evol.imgsize
    imgpos = EStats[Expi-1].evol.imgpos
    pref_chan = EStats[Expi-1].evol.pref_chan
    imgpix = int(imgsize * 40)
    titstr = "Driver Chan %d, %.1f deg [%s]"%(pref_chan, imgsize, tuple(imgpos))

    featFetcher = Corr_Feat_Machine()
    featFetcher.register_hooks(VGG, ["conv2_2", "conv3_3","conv4_3", "conv5_3"])
    featFetcher.init_corr()

    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50,200)])
    imgN = len(imgfullpath_vect)
    score_tsr = torch.tensor(score_vect).float() # torchify the score vector
    csr = 0
    pbar = tqdm(total=imgN)
    while csr < imgN:
        cend = min(csr + batchsize, imgN)
        input_tsr = loadimg_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix)
        # input_tsr = loadimg_embed_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
        # Pool through VGG
        with torch.no_grad():
            part_tsr = VGG.features(input_tsr.cuda()).cpu()
        featFetcher.update_corr(score_tsr[csr:cend])
        # update bar!
        pbar.update(cend-csr)
        csr = cend
    pbar.close()
    featFetcher.calc_corr()
    np.savez(join("S:\corrFeatTsr","%s_Exp%d_Evol_corrTsr.npz"%(Animal,Expi)), **featFetcher.make_savedict())
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "Evol", titstr, figdir=figdir)

    # Load manifold experiment, single trial response
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50,200)])
    imgN_M = len(imgfullpath_vect_M)
    csr = 0
    pbar = tqdm(total=imgN_M)
    while csr < imgN_M:
        cend = min(csr + batchsize, imgN_M)
        input_tsr = loadimg_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix)
        # input_tsr = loadimg_embed_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
        # Pool through VGG
        with torch.no_grad():
            part_tsr = VGG.features(input_tsr.cuda()).cpu()
        featFetcher.update_corr_rep(scorecol_M[csr:cend])
        # update bar!
        pbar.update(cend-csr)
        csr = cend
    pbar.close()

    featFetcher.calc_corr()
    np.savez(join("S:\corrFeatTsr","%s_Exp%d_EM_corrTsr.npz"%(Animal,Expi)), **featFetcher.make_savedict())
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "EM_sgtr", titstr, figdir=figdir )
    featFetcher.clear_hook()

#%% Final version, ResNet edition
net = models.resnet50(pretrained=True)
net.requires_grad_(False).cuda()

online_compute = True
batchsize = 121
figdir = join("S:\corrFeatTsr", "ResNetsummary")
netname = "resnet50"
layers2plot = ["layer4", "layer3",  "layer2", "layer1", ]
for Expi in range(5, len(EStats)+1):
    imgsize = EStats[Expi-1].evol.imgsize
    imgpos = EStats[Expi-1].evol.imgpos
    pref_chan = EStats[Expi-1].evol.pref_chan
    imgpix = int(imgsize * 40)
    titstr = "Driver Chan %d, %.1f deg [%s]"%(pref_chan, imgsize, tuple(imgpos))

    featFetcher = Corr_Feat_Machine()
    featFetcher.register_hooks(net, ["layer4", "layer3",  "layer2", "layer1", ], netname=netname)
    featFetcher.init_corr()

    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50,200)])
    imgN = len(imgfullpath_vect)
    score_tsr = torch.tensor(score_vect).float() # torchify the score vector
    csr = 0
    pbar = tqdm(total=imgN)
    while csr < imgN:
        cend = min(csr + batchsize, imgN)
        input_tsr = loadimg_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix)
        # input_tsr = loadimg_embed_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
        # Pool through VGG
        with torch.no_grad():
            part_tsr = net(input_tsr.cuda()).cpu()
        featFetcher.update_corr(score_tsr[csr:cend])
        # update bar!
        pbar.update(cend-csr)
        csr = cend
    pbar.close()
    featFetcher.calc_corr()
    np.savez(join("S:\corrFeatTsr","%s_Exp%d_Evol_corrTsr_%s.npz"%(Animal,Expi,netname)), **featFetcher.make_savedict())
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "Evol", titstr, figdir=figdir)

    # Load manifold experiment, single trial response
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50,200)])
    imgN_M = len(imgfullpath_vect_M)
    csr = 0
    pbar = tqdm(total=imgN_M)
    while csr < imgN_M:
        cend = min(csr + batchsize, imgN_M)
        input_tsr = loadimg_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix)
        # input_tsr = loadimg_embed_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
        # Pool through VGG
        with torch.no_grad():
            part_tsr = net(input_tsr.cuda()).cpu()
        featFetcher.update_corr_rep(scorecol_M[csr:cend])
        # update bar!
        pbar.update(cend-csr)
        csr = cend
    pbar.close()

    featFetcher.calc_corr()
    np.savez(join("S:\corrFeatTsr","%s_Exp%d_EM_corrTsr_%s.npz"%(Animal,Expi,netname)), **featFetcher.make_savedict())
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "EM_sgtr", titstr, figdir=figdir)
    featFetcher.clear_hook()
#%% Load the featDict and see! 
Expi = 3
featDict = np.load(join("S:\corrFeatTsr","%s_Exp%d_EM_corrTsr.npz"%(Animal,Expi)), allow_pickle=True)

#%% Plot Embedded version of image
online_compute = True
batchsize = 100
figdir = join("S:\corrFeatTsr", "VGGsummary")
layers2plot = ["conv5_3", "conv4_3",  "conv3_3", "conv2_2", ]
for Expi in range(27,len(EStats)+1):  # len(EStats)+1):
    imgsize = EStats[Expi-1].evol.imgsize
    imgpos = EStats[Expi-1].evol.imgpos
    pref_chan = EStats[Expi-1].evol.pref_chan
    imgpix = int(imgsize * 40)
    titstr = "Driver Chan %d, %.1f deg [%s]"%(pref_chan, imgsize, tuple(imgpos))

    featFetcher = Corr_Feat_Machine()
    featFetcher.register_hooks(VGG, ["conv2_2", "conv3_3","conv4_3", "conv5_3"], netname="vgg16")
    featFetcher.init_corr()

    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50,200)])
    imgN = len(imgfullpath_vect)
    score_tsr = torch.tensor(score_vect).float() # torchify the score vector
    csr = 0
    pbar = tqdm(total=imgN)
    while csr < imgN:
        cend = min(csr + batchsize, imgN)
        # input_tsr = loadimg_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix)
        input_tsr = loadimg_embed_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix, fullimgsz=224)
        # Pool through VGG
        with torch.no_grad():
            part_tsr = VGG.features(input_tsr.cuda()).cpu()
        featFetcher.update_corr(score_tsr[csr:cend])
        pbar.update(cend-csr) # update bar!
        csr = cend
    pbar.close()
    featFetcher.calc_corr()
    np.savez(join("S:\corrFeatTsr","%s_Exp%d_Evol_embed_corrTsr.npz"%(Animal,Expi)), **featFetcher.make_savedict())
    # figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "Evol_embed", titstr, figdir=figdir)
    figh = visualize_cctsr_embed(featFetcher, layers2plot, ReprStats, Expi, Animal, "Evol_embed", titstr, figdir=figdir, imgpix=imgpix, fullimgsz=224)
    # Load manifold experiment, single trial response
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50,200)])
    imgN_M = len(imgfullpath_vect_M)
    csr = 0
    pbar = tqdm(total=imgN_M)
    while csr < imgN_M:
        cend = min(csr + batchsize, imgN_M)
        # input_tsr = loadimg_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix)
        input_tsr = loadimg_embed_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix, fullimgsz=224)
        # Pool through VGG
        with torch.no_grad():
            part_tsr = VGG.features(input_tsr.cuda()).cpu()
        featFetcher.update_corr_rep(scorecol_M[csr:cend])
        # update bar!
        pbar.update(cend-csr)
        csr = cend
    pbar.close()

    featFetcher.calc_corr()
    np.savez(join("S:\corrFeatTsr","%s_Exp%d_EM_embed_corrTsr.npz"%(Animal,Expi)), **featFetcher.make_savedict())
    # figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "EM_sgtr_embed", titstr, figdir=figdir )
    figh = visualize_cctsr_embed(featFetcher, layers2plot, ReprStats, Expi, Animal, "EM_sgtr_embed", titstr, figdir=figdir, imgpix=imgpix, fullimgsz=224)
    featFetcher.clear_hook()
#%%
featFetcher

# #%%
# featFetcher = Corr_Feat_Machine()
# featFetcher.register_hooks(VGG, ["conv3_3","conv4_3", "conv5_3"])#, "conv4_3", "conv5_3"
# featFetcher.init_corr()
# VGG.features(input_tsr.cuda())
# featFetcher.update_corr(score_tsr[-25:])
# featFetcher.calc_corr()
# #%%
# online_compute = True
# imgN = len(imgfullpath_vect)
# feat_tsr = torch.tensor([])
# score_tsr = torch.tensor(score_vect).float()
# innerProd = None
# featS = None
# featSSq = None
# csr = 0; batchsize = 100
# pbar = tqdm(total=imgN)
# while csr < imgN:
#     cend = min(csr + batchsize, imgN)
#     # Prepare the input image batch!
#     ppimgs = []
#     for img_path in (imgfullpath_vect[csr:cend]): # should be taken care of by the CNN part
#         curimg = imread(img_path)
#         x = preprocess(curimg)
#         ppimgs.append(x.unsqueeze(0))
#     input_tsr = torch.cat(tuple(ppimgs), dim=0)
#     # input_tsr = median_blur(input_tsr, (3, 3)) # this looks good but very slow, no use
#     input_tsr = gaussian_blur2d(input_tsr, (5, 5), sigma=(3, 3))
#     input_tsr = F.interpolate(input_tsr, size=[imgpix, imgpix], align_corners=True, mode='bilinear')
#     input_tsr = F.interpolate(input_tsr, size=[224, 224], align_corners=True, mode='bilinear')
#     # Pool through VGG
#     with torch.no_grad():
#         part_tsr = FeatNet(input_tsr.cuda()).cpu()
#     innerProd_tmp = torch.einsum("i,ijkl->jkl", score_tsr[csr:cend], part_tsr, )  # [time by features]
#     innerProd = innerProd_tmp if innerProd is None else innerProd + innerProd_tmp
#     featS = part_tsr.sum(dim=0) if featS is None else part_tsr.sum(dim=0) + featS
#     featSSq = (part_tsr ** 2).sum(dim=0) if featSSq is None else (part_tsr ** 2).sum(dim=0) + featSSq
#     # update bar!
#     pbar.update(cend-csr)
#     csr = cend
# pbar.close()
# #%%
# featM = featS / imgN
# featMSq = featSSq / imgN
# featStd = (featMSq - featM**2).sqrt()
# innerProd_M = innerProd / imgN
# scorM = score_tsr.mean(dim=0)
# scorMSq = (score_tsr ** 2).mean(dim=0)
# scorStd = (scorMSq - scorM**2).sqrt()
# #%%
# cctsr = (innerProd_M - scorM * featM) / featStd / scorStd
# #%%
# Ttsr = np.sqrt(imgN - 2) * cctsr / (1 - cctsr**2).sqrt()
# #%%
# plt.matshow(Ttsr.abs().mean(dim=0).data.numpy())
# plt.colorbar()
# plt.show()