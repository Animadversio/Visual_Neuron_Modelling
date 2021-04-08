"""Compute feature tensors of a batch of images.(PyTorch Pipeline)"""
from torchvision.models import vgg16, alexnet
import torch
import torch.nn.functional as F
from skimage.io import imread, imread_collection
from os.path import join
from glob import glob
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
Pasupath = r"N:\Stimuli\2019-Manifold\pasupathy-wg-f-4-ori"
Gaborpath = r"N:\Stimuli\2019-Manifold\gabor"

#%% Load full path to images and psths
def load_score_mat(EStats, MStats, Expi, ExpType, wdws=[(50,200)]):
    """
    Demo code
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

from CorrFeatTsr_lib import visualize_cctsr, visualize_cctsr_embed, Corr_Feat_Machine, Corr_Feat_pipeline, loadimg_preprocess, loadimg_embed_preprocess
#%%
VGG = vgg16(pretrained=True).cuda()
VGG.requires_grad_(False)
VGG.eval()
del VGG.classifier
#%%
datadir = r"S:\FeatTsr"
Animal = "Alfa"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
#%%
Expi = 3

savedir = join(datadir, "%s_Exp%02d_E"%(Animal, Expi))
score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)])
#%%
savedir = join(datadir, "%s_Exp%02d_M"%(Animal, Expi))
score_vect_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)])
#%%
def compute_FeatTsr(imgfullpath_vect_M, net, featFetcher, batchsize=30, imgpix=224):
    featTsrs = {}
    imgN_M = len(imgfullpath_vect_M)
    csr = 0
    pbar = tqdm(total=imgN_M)
    while csr < imgN_M:
        cend = min(csr + batchsize, imgN_M)
        input_tsr = loadimg_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix)
        # input_tsr = loadimg_embed_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
        # Pool through VGG
        with torch.no_grad():
            part_tsr = net(input_tsr.cuda()).cpu()
        # featFetcher.update_corr_rep(scorecol_M[csr:cend])
        for layer, tsr in featFetcher.feat_tsr.items():
            if not layer in featTsrs:
                featTsrs[layer] = tsr.clone().half()
            else:
                featTsrs[layer] = torch.cat((featTsrs[layer], tsr.clone().half()), dim=0)
        # update bar!
        pbar.update(cend - csr)
        csr = cend
    pbar.close()
    featFetcher.clear_hook()
    for layer, tsr in featTsrs.items():
        featTsrs[layer] = tsr.numpy()
    return featTsrs

savedir = join(datadir, "%s_Exp%02d_E"%(Animal, Expi))
score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)])
featFetcher = Corr_Feat_Machine()
featFetcher.register_hooks(VGG, ["conv3_3", "conv4_3", "conv5_3"])  # "conv2_2",
from sklearn.model_selection import train_test_split
train_idx, valid_idx = train_test_split(range(len(imgfullpath_vect)), test_size=0.2, random_state=35, shuffle=False, )
imgfp_train, imgfp_valid = [imgfullpath_vect[idx] for idx in train_idx], [imgfullpath_vect[idx] for idx in valid_idx]
score_train, score_valid = score_vect[train_idx], score_vect[valid_idx]
trainN, validN = len(train_idx), len(valid_idx)
featTsrs = compute_FeatTsr(imgfp_train, VGG.features, featFetcher, batchsize=40, imgpix=224)
# np.savez(join(datadir, "%s_Exp%02d_E_VGG16_featTsr_train.npz"%(Animal, Expi)), imgpath_vect=imgfp_train, score_vect=score_train, featTsrs=featTsrs, idx=train_idx)
#%%
import pickle
pickle.dump({"imgpath_vect":imgfp_train, "score_vect":score_train, "featTsrs":featTsrs, "idx":train_idx}, open(join(datadir, "%s_Exp%02d_E_VGG16_featTsr_train.pkl"%(Animal, Expi)),mode="wb"), protocol=4)
#%%
featFetcher = Corr_Feat_Machine()
featFetcher.register_hooks(VGG, ["conv3_3", "conv4_3", "conv5_3"])
featTsrs = compute_FeatTsr(imgfp_valid, VGG.features, featFetcher, batchsize=40, imgpix=224)
np.savez(join(datadir, "%s_Exp%02d_E_VGG16_featTsr_val.npz"%(Animal, Expi)), imgpath_vect=imgfp_valid, score_vect=score_valid, featTsrs=featTsrs, idx=valid_idx)
#%%
batchsize = 30
imgpix = 224

featFetcher = Corr_Feat_Machine()
featFetcher.register_hooks(VGG, ["conv3_3", "conv4_3", "conv5_3"])  # "conv2_2",

featTsrs = {}
imgN_M = len(imgfullpath_vect_M)
csr = 0
pbar = tqdm(total=imgN_M)
while csr < imgN_M:
    cend = min(csr + batchsize, imgN_M)
    input_tsr = loadimg_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix)
    # input_tsr = loadimg_embed_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
    # Pool through VGG
    with torch.no_grad():
        part_tsr = VGG.features(input_tsr.cuda()).cpu()
    # featFetcher.update_corr_rep(scorecol_M[csr:cend])
    for layer, tsr in featFetcher.feat_tsr.items():
        if not layer in featTsrs:
            featTsrs[layer] = tsr.clone().half()
        else:
            featTsrs[layer] = torch.cat((featTsrs[layer], tsr.clone().half()), dim=0)
    # update bar!
    pbar.update(cend-csr)
    csr = cend
pbar.close()
featFetcher.clear_hook()
for layer, tsr in featTsrs.items():
    featTsrs[layer] = tsr.numpy()
#%%
np.savez(join(datadir, "%s_Exp%02d_M_VGG16_featTsr.npz"%(Animal, Expi)), imgpath_vect=imgfullpath_vect_M, score_vect=score_vect_M, featTsrs=featTsrs)
