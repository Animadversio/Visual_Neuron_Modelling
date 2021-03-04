# Alias the disks for usage
# !subst N: E:\Network_Data_Sync
# !subst S: E:\Network_Data_Sync
# !subst O: "E:\OneDrive - Washington University in St. Louis"
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

# net = models.resnet50(pretrained=True)
# net.requires_grad_(False).cuda()

#%%
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
Pasupath = r"N:\Stimuli\2019-Manifold\pasupathy-wg-f-4-ori"
Gaborpath = r"N:\Stimuli\2019-Manifold\gabor"
Animal = "Alfa"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
# ManifDyn = loadmat(join(mat_path, Animal + "_ManifPopDynamics.mat"), struct_as_record=False, squeeze_me=True)['ManifDyn']
#%% Routine for loading images
# loadbatch = 50
# ppimgs = []
# for img_path in tqdm(imgfullpath_vect[-loadbatch:]):
#     # should be taken care of by the CNN part
#     curimg = imread(img_path)
#     x = preprocess(curimg)
#     ppimgs.append(x.unsqueeze(0))
# input_tsr = torch.cat(tuple(ppimgs), dim=0)
# # input_tsr = median_blur(input_tsr, (3, 3)) # this looks good but very slow
# input_tsr = gaussian_blur2d(input_tsr, (5, 5), sigma=(3, 3))
# input_tsr = F.interpolate(input_tsr, size=[imgpix, imgpix], align_corners=True, mode='bilinear')
# input_tsr = F.interpolate(input_tsr, size=[224, 224], align_corners=True, mode='bilinear')
# #%%
# #%% Note there are some high freq noise signal in the image so may be misleading to the CNN
# ToPILImage()(torch.clamp(input_tsr[-1]*0.2+0.4, 0, 1)).show()
# ToPILImage()(input_tsr[-1]).show()
# ToPILImage()(torch.clamp(median_blur(input_tsr[-2:-1], (3, 3))[0]*0.2+0.4, 0, 1)).show()
# ToPILImage()(median_blur(input_tsr[-2:-1], (3, 3))[0]).show()
# We want to remove the checkerboard noise!

#%%
from CorrFeatTsr_lib import visualize_cctsr, visualize_cctsr_embed, Corr_Feat_Machine, Corr_Feat_pipeline
#%% Load image names and psths
def load_score_mat(EStats, MStats, Expi, ExpType, wdws=[(50,200)]):
    """
    test_code
    ```
    Expi = 3
    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50,200)])
    score_vect_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50,200)])
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50,200)])
    ```
    :param EStats:
    :param MStats:
    :param Expi:
    :param ExpType:
    :param wdws:
    :return:
    """
    if ExpType=="Evol":
        psth = EStats[Expi-1].evol.psth
        if psth[0].ndim == 3:
            nunit = psth[0].shape[0]
        else:
            nunit = 1
        psthmat = np.concatenate(tuple(np.reshape(P,[nunit,200,-1]) for P in psth), axis=2)
        assert nunit==1
        score_vect = np.mean(psthmat[0, 50:, :], axis=(0)).astype(np.float)
        # score_vect = scorevec.reshape(-1)
        idxvec = np.concatenate(tuple(np.reshape(I,(-1)) for I in EStats[Expi-1].evol.idx_seq))
        imgnm_vect = EStats[Expi-1].imageName[idxvec-1]  # -1 for the index starting from 0 instead of 1
        stimpath = EStats[Expi - 1].meta.stimuli
        stimpath = stimpath.replace(r"\\storage1.ris.wustl.edu\crponce\Active", r"N:")
        imgnms_col = glob(stimpath+"\\*") + glob(Pasupath+"\\*") + glob(Gaborpath+"\\*")
        imgfullpath_vect = [[path for path in imgnms_col if imgnm in path][0] for imgnm in imgnm_vect]
        return score_vect, imgfullpath_vect
    elif "Manif" in ExpType:
        ui = EStats[Expi - 1].evol.unit_in_pref_chan  # unit id in the pref chan
        if MStats[Expi - 1].manif.psth.shape == (11, 11):
            psth = MStats[Expi - 1].manif.psth.reshape(-1)
            idx_vect = MStats[Expi - 1].manif.idx_grid.reshape([-1])
        else:
            psth = MStats[Expi - 1].manif.psth[0].reshape(-1)
            idx_vect = MStats[Expi - 1].manif.idx_grid[0].reshape([-1])
        if psth[0].ndim == 3:
            nunit = psth[0].shape[0]
        else:
            nunit = 1
        psthlist = list(np.reshape(P, [nunit, 200, -1]) for P in psth)
        scorecol = [np.mean(P[ui - 1, 50:200, :], axis=0).astype(np.float) for P in psthlist]
        idx_vect = [np.array(idx).reshape(-1) for idx in idx_vect]
        imgnm_vect = [MStats[Expi - 1].imageName[idx[0] - 1] for idx in idx_vect]
        stimpath = MStats[Expi - 1].meta.stimuli
        stimpath = stimpath.replace(r"\\storage1.ris.wustl.edu\crponce\Active", r"N:")
        imgnms = glob(stimpath + "\\*") + glob(Pasupath + "\\*") + glob(Gaborpath + "\\*")
        imgfullpath_vect = [[path for path in imgnms if imgnm in path][0] for imgnm in imgnm_vect]
        if ExpType == "Manif_avg":
            score_vect = np.array([np.mean(score) for score in scorecol]).astype(np.float)
            return score_vect, imgfullpath_vect
        elif ExpType == "Manif_sgtr":
            return scorecol, imgfullpath_vect
# ui = EStats[Expi - 1].evol.unit_in_pref_chan # unit id in the pref chan
# psth = MStats[Expi-1].manif.psth.reshape(-1)
# if psth[0].ndim == 3:
#     nunit = psth[0].shape[0]
# else:
#     nunit = 1
# psthlist = list(np.reshape(P, [nunit, 200, -1]) for P in psth)
# scorecol = [np.mean(P[ui-1, 50:200, :],axis=0).astype(np.float) for P in psthlist]
#%%
RGBmean = torch.tensor([0.485, 0.456, 0.406]).float().reshape([1,3,1,1])
RGBstd = torch.tensor([0.229, 0.224, 0.225]).float().reshape([1,3,1,1])
preprocess = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
def loadimg_preprocess(imgfullpath, imgpix=120, fullimgsz=224, borderblur=False):
    """Prepare the input image batch!
    Load the image, cat as 4d tensor, blur the image to get rid of high freq noise, interpolate to certain resolution.
    """
    ppimgs = []
    for img_path in (imgfullpath):  # should be taken care of by the CNN part
        curimg = imread(img_path)
        x = preprocess(curimg)
        ppimgs.append(x.unsqueeze(0))
    input_tsr = torch.cat(tuple(ppimgs), dim=0)
    # input_tsr = median_blur(input_tsr, (3, 3)) # this looks good but very slow, no use
    input_tsr = gaussian_blur2d(input_tsr, (5, 5), sigma=(3, 3))
    input_tsr = F.interpolate(input_tsr, size=[imgpix, imgpix], align_corners=True, mode='bilinear')
    input_tsr = F.interpolate(input_tsr, size=[fullimgsz, fullimgsz], align_corners=True, mode='bilinear')
    if borderblur:
        border = round(fullimgsz * 0.05)
        bkgrd_tsr = 0.5 * torch.ones([1, 3, fullimgsz, fullimgsz])
        msk = torch.ones([fullimgsz - 2 * border, fullimgsz - 2 * border])
        msk = F.pad(msk, [border, border, border, border], mode="constant", value=0)
        blurmsk = (1 - msk).reshape([1, 1, fullimgsz, fullimgsz]);
        blurmsk_trans = gaussian_blur2d(blurmsk, (5, 5), sigma=(3, 3))
        # bkgrdtsr = gaussian_blur2d(input_tsr*blurmsk, (5, 5), sigma=(3, 3))
        final_tsr = bkgrd_tsr * blurmsk_trans + input_tsr * (1 - blurmsk_trans)
        final_tsr = (final_tsr - RGBmean) / RGBstd
        return final_tsr
    else:
        return input_tsr

def loadimg_embed_preprocess(imgfullpath, imgpix=120, fullimgsz=224, borderblur=True):
    """Prepare the input image batch!
    Load the image, cat as 4d tensor, blur the image to get rid of high freq noise, interpolate to certain resolution, put the image embedded in a gray background.
    """
    ppimgs = []
    for img_path in (imgfullpath):  # should be taken care of by the CNN part
        curimg = imread(img_path)
        x = transforms.ToTensor()(curimg)
        ppimgs.append(x.unsqueeze(0))
    input_tsr = torch.cat(tuple(ppimgs), dim=0)
    # input_tsr = median_blur(input_tsr, (3, 3)) # this looks good but very slow, no use
    input_tsr = gaussian_blur2d(input_tsr, (5, 5), sigma=(3, 3))
    imgpix = min(imgpix, fullimgsz)
    input_tsr = F.interpolate(input_tsr, size=[imgpix, imgpix], align_corners=True, mode='bilinear')
    padbef = (fullimgsz - imgpix) // 2
    padaft = (fullimgsz - imgpix) - padbef
    input_tsr = F.pad(input_tsr, [padbef, padaft, padbef, padaft, 0, 0, 0, 0], mode="constant", value=0.5)
    if borderblur:
        border = round(imgpix*0.05)
        bkgrd_tsr = torch.ones([1,3,fullimgsz,fullimgsz])
        msk = torch.ones([imgpix-2*border, imgpix-2*border])
        msk = F.pad(msk, [padbef+border, padaft+border, padbef+border, padaft+border], mode="constant", value=0.5)
        blurmsk = (1-msk).reshape([1,1,fullimgsz,fullimgsz]);
        blurmsk_trans = gaussian_blur2d(blurmsk, (5,5), sigma=(3, 3))
        # bkgrdtsr = gaussian_blur2d(input_tsr*blurmsk, (5, 5), sigma=(3, 3))
        final_tsr = bkgrd_tsr*blurmsk_trans + input_tsr*(1 - blurmsk_trans)
        final_tsr = (final_tsr - RGBmean) / RGBstd
        return final_tsr
    else:
        input_tsr = (input_tsr - RGBmean) / RGBstd
        return input_tsr
#%%
VGG = models.vgg16(pretrained=True)
VGG.requires_grad_(False)
VGG.features.cuda()
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
    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)])
    Corr_Feat_pipeline(VGG.features, featFetcher, score_vect, imgfullpath_vect, lambda x:loadimg_preprocess(x, borderblur=True, imgpix=imgpix), online_compute=True, batchsize=121, savedir=figdir, savenm="%s_Exp%d_Evol_nobdr" % (Animal, Expi), )
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "Evol_nobdr", titstr, figdir=figdir)
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50,200)])
    Corr_Feat_pipeline(VGG.features, featFetcher, scorecol_M, imgfullpath_vect_M,
           lambda x: loadimg_preprocess(x, borderblur=True, imgpix=imgpix), online_compute=True,
                       batchsize=121, savedir=figdir, savenm="%s_Exp%d_EM_nobdr" % (Animal, Expi), )
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "EM_sgtr_nobdr", titstr, figdir=figdir)
    featFetcher.clear_hook()
#%% Final version VGG16
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
#%%
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