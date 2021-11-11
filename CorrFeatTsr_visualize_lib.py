"""Visualize the correlated units for a given evolution.
Basic building blocks of a feature visualization
    CorrFeatScore: a Scorer or objective function based on neural network
    corr_visualize: Visualize feature based on pixel parametrization
    corr_GAN_visualize: Visualize feature based on GAN parametrization
All these components are heavily used in higher level api in featvis_lib

"""
# Alias the disks for usage
# !subst N: E:\Network_Data_Sync
# !subst S: E:\Network_Data_Sync
# !subst O: "E:\OneDrive - Washington University in St. Louis"
#%%
from scipy.io import loadmat
from skimage.io import imread, imread_collection
import os
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

from collections import defaultdict
from scipy.stats import t
from easydict import EasyDict
layername_dict={"alexnet":["conv1", "conv1_relu", "pool1",
                            "conv2", "conv2_relu", "pool2",
                            "conv3", "conv3_relu",
                            "conv4", "conv4_relu",
                            "conv5", "conv5_relu", "pool3",
                            "dropout1", "fc6", "fc6_relu",
                            "dropout2", "fc7", "fc7_relu",
                            "fc8",],
                "vgg16":['conv1_1', 'conv1_1_relu',
                         'conv1_2', 'conv1_2_relu', 'pool1',
                         'conv2_1', 'conv2_1_relu',
                         'conv2_2', 'conv2_2_relu', 'pool2',
                         'conv3_1', 'conv3_1_relu',
                         'conv3_2', 'conv3_2_relu',
                         'conv3_3', 'conv3_3_relu', 'pool3',
                         'conv4_1', 'conv4_1_relu',
                         'conv4_2', 'conv4_2_relu',
                         'conv4_3', 'conv4_3_relu', 'pool4',
                         'conv5_1', 'conv5_1_relu',
                         'conv5_2', 'conv5_2_relu',
                         'conv5_3', 'conv5_3_relu', 'pool5',
                         'fc6', 'fc6_relu', 'dropout6',
                         'fc7', 'fc7_relu', 'dropout7',
                         'fc8'],
                "densenet121":['conv1',
                                 'bn1',
                                 'bn1_relu',
                                 'pool1',
                                 'denseblock1', 'transition1',
                                 'denseblock2', 'transition2',
                                 'denseblock3', 'transition3',
                                 'denseblock4',
                                 'bn2',
                                 'fc1'],
                "resnet50": ["layer1", "layer2", "layer3", "layer4"]}

class CorrFeatScore:
    def __init__(self):
        self.feat_tsr = {}
        self.weight_tsr = {}
        self.mask_tsr = {}
        self.weight_N = {}
        self.hooks = []
        self.layers = []
        self.scores = {}
        self.netname = None
        self.mode = "dot"  # "corr"

    def hook_forger(self, layer, grad=True):
        # this function is important, or layer will be redefined in the same scope!
        def activ_hook(module, fea_in, fea_out):
            # print("Extract from hooker on %s" % module.__class__)
            # ref_feat = fea_out.detach().clone().cpu()
            # ref_feat.requires_grad_(False)
            self.feat_tsr[layer] = fea_out
            return None
        return activ_hook

    def register_hooks(self, net, layers, netname="vgg16"):
        if isinstance(layers, str):
            layers = [layers]

        for layer in layers:
            if netname in ["vgg16","alexnet"]:
                layer_idx = layername_dict[netname].index(layer)
                if layer_idx > 30:
                    targmodule = net.classifier[layer_idx-31]
                elif layer_idx < 30:
                    targmodule = net.features[layer_idx]
                else:
                    targmodule = net.avgpool
            elif "resnet50" in netname: # in ["resnet50", "resnet50_linf"]:
                targmodule = net.__getattr__(layer)
            else:
                raise NotImplementedError
            actH = targmodule.register_forward_hook(self.hook_forger(layer))
            self.hooks.append(actH)
            self.layers.append(layer)
        self.netname = netname

    def register_weights(self, weight_dict, mask_dict=None):
        for layer, weight in weight_dict.items():
            self.weight_tsr[layer] = torch.tensor(weight).float().cuda()
            self.weight_tsr[layer].requires_grad_(False)
            self.weight_N[layer] = (weight > 0).sum()
            if mask_dict is not None and layer in mask_dict:
                mask = mask_dict[layer]
                self.mask_tsr[layer] = torch.tensor(mask, requires_grad=False).bool().cuda()

    def corrfeat_score(self, layers=None, Nnorm=True):
        if layers is None: layers = self.layers
        if isinstance(layers, str):
            layers = [layers]
        for layer in layers:
            acttsr = self.feat_tsr[layer]
            if acttsr.ndim == 2: # fc layers
                sumdims = [1]
            elif acttsr.ndim == 4: # conv layers
                sumdims = [1, 2, 3]
            else:
                raise ValueError
            if self.mode == "dot":
                score = (self.weight_tsr[layer] * acttsr).sum(dim=sumdims)
                if Nnorm: score = score / self.weight_N[layer]
            elif self.mode == "MSE":
                score = (self.weight_tsr[layer] - acttsr).pow(2).mean(dim=sumdims)
            elif self.mode == "MSEmask":
                score = torch.sum(self.mask_tsr[layer]*(self.weight_tsr[layer] - acttsr).pow(2), dim=sumdims) / \
                        self.mask_tsr[layer].count_nonzero().float()#(~torch.isnan(self.weight_tsr[layer])).float().sum()
            elif self.mode == "L1":
                score = (self.weight_tsr[layer] - acttsr).abs().mean(dim=sumdims)
            elif self.mode == "L1mask":
                score = torch.sum(self.mask_tsr[layer]*(self.weight_tsr[layer] - acttsr).abs(), dim=sumdims) / \
                        self.mask_tsr[layer].count_nonzero().float()
            elif self.mode == "corr":
                w_mean = self.weight_tsr[layer].mean()
                w_std = self.weight_tsr[layer].std()
                act_mean = acttsr.mean(dim=sumdims, keepdim=True)
                act_std = acttsr.std(dim=sumdims, keepdim=True)
                score = ((self.weight_tsr[layer] - w_mean) / w_std * (acttsr - act_mean) / act_std).mean(dim=sumdims)
            elif self.mode == "cosine":
                w_norm = torch.linalg.norm(self.weight_tsr[layer])
                act_norm = torch.linalg.norm(acttsr, dim=sumdims, keepdim=True)
                score = ((self.weight_tsr[layer] * acttsr) / w_norm / act_norm).sum(dim=sumdims) # validate
            elif self.mode == "corrmask":
                msk = self.mask_tsr[layer]
                weightvec = self.weight_tsr[layer][msk]
                w_mean = weightvec.mean()
                w_std = weightvec.std()
                actmat = acttsr[:, msk]
                act_mean = actmat.mean(dim=1, keepdim=True)
                act_std = actmat.std(dim=1, keepdim=True)
                score = ((weightvec - w_mean) / w_std * (actmat - act_mean) / act_std).mean(dim=1)
            else:
                raise NotImplementedError("Check `mode` of `scorer` ")
            self.scores[layer] = score
        if len(layers) > 1:
            return self.scores
        else:
            return self.scores[layers[0]]

    def featvec_corrmap(self, layer:str, featvec):
        act_feattsr = self.feat_tsr[layer].cpu()
        target_vec = torch.from_numpy(featvec).reshape([1, -1, 1, 1]).float()
        cov_map = (act_feattsr * target_vec).mean(dim=1, keepdim=False) # torch.tensor (B, H, W)
        z_feattsr = (act_feattsr - act_feattsr.mean(dim=1, keepdim=True)) / act_feattsr.std(dim=1, keepdim=True)
        z_featvec = (target_vec - target_vec.mean(dim=1, keepdim=True)) / target_vec.std(dim=1, keepdim=True)
        corr_map = (z_feattsr * z_featvec).mean(dim=1) # torch.tensor (B, H, W)
        return cov_map, corr_map

    def load_from_npy(self, savedict, net, netname, thresh=0, layers=[]):
        # imgN = savedict["imgN"]
        if savedict["cctsr"].shape == ():
            cctsr = savedict["cctsr"].item()
            Ttsr = savedict["Ttsr"].item()
        else:
            cctsr = savedict["cctsr"]
            Ttsr = savedict["Ttsr"]

        weight_dict = {}
        for layer in (cctsr.keys() if len(layers) == 0 else layers):
            weight = cctsr[layer]
            weight[np.abs(Ttsr[layer]) < thresh] = 0
            weight_dict[layer] = weight
            if layer not in self.layers:
                self.register_hooks(net, layer, netname=netname)
        self.register_weights(weight_dict)

    def clear_hook(self):
        for h in self.hooks:
            h.remove()
        self.layers = []

    def __del__(self):
        self.clear_hook()
        print('Feature Correlator Destructed, Hooks deleted.')

#%%
from torch.optim import SGD, Adam
from torchvision.utils import make_grid
from GAN_utils import upconvGAN
RGBmean = torch.tensor([0.485, 0.456, 0.406]).float().reshape([1,3,1,1])
RGBstd = torch.tensor([0.229, 0.224, 0.225]).float().reshape([1,3,1,1])
def save_imgtsr(finimgs, figdir:str ="", savestr:str =""):
    """
    finimgs: a torch tensor on cpu with shape B,C,H,W. 
    """
    B = finimgs.shape[0]
    for imgi in range(B):
        ToPILImage()(finimgs[imgi,:,:,:]).save(join(figdir, "%s_%02d.png"%(savestr, imgi)))


def preprocess(img: torch.tensor):
    """ clamp range to 0, 1; Blur the image; Centralize the tensor to go into CNN."""
    img = torch.clamp(img,0,1)
    img = gaussian_blur2d(img, (5,5), sigma=(3, 3))
    img = (img - RGBmean.to(img.device)) / RGBstd.to(img.device)
    return img


def compose(transforms):
    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return inner


def corr_visualize(scorer, CNNnet, preprocess, layername, tfms=[],
    lr=0.01, imgfullpix=224, MAXSTEP=100, Bsize=4, saveImgN=None, use_adam=True, langevin_eps=0, 
    savestr="", figdir="", imshow=False, PILshow=False, verbose=True, saveimg=False, score_mode="dot", maximize=True):
    """  """
    scorer.mode = score_mode
    score_sgn = -1 if maximize else 1
    x = 0.5+0.01*torch.rand((Bsize,3,imgfullpix,imgfullpix)).cuda()
    x.requires_grad_(True)
    optimizer = Adam([x], lr=lr) if use_adam else SGD([x], lr=lr)
    tfms_f = compose(tfms)
    score_traj = []
    pbar = tqdm(range(MAXSTEP))
    for step in pbar:
        ppx = preprocess(x)
        optimizer.zero_grad()
        CNNnet(tfms_f(ppx))
        score = score_sgn * scorer.corrfeat_score(layername)
        score.sum().backward()
        x.grad = x.norm() / x.grad.norm() * x.grad
        optimizer.step()
        score_traj.append(score.detach().clone().cpu())
        if langevin_eps > 0:
            # if > 0 then add noise to become Langevin gradient descent jump minimum
            x.data.add_(torch.randn(x.shape, device="cuda") * langevin_eps)
        if verbose and step % 10 == 0:
            print("step %d, score %s"%(step, " ".join("%.1f"%s for s in score_sgn * score)))
        pbar.set_description("step %d, score %s"%(step, " ".join("%.2f" % s for s in score_sgn * score)))

    final_score = score_sgn * score.detach().clone().cpu()
    del score
    torch.cuda.empty_cache()
    if maximize:
        idx = torch.argsort(final_score, descending=True)
    else:
        idx = torch.argsort(final_score, descending=False)
    score_traj = score_sgn * torch.stack(tuple(score_traj))[:, idx]
    finimgs = x.detach().clone().cpu()[idx, :, :, :]  # finimgs are generated by z before preprocessing.
    print("Final scores %s"%(" ".join("%.2f" % s for s in final_score[idx])))
    mtg = ToPILImage()(make_grid(finimgs))
    if PILshow: mtg.show()
    mtg.save(join(figdir, "%s_pix_%s.png"%(savestr, layername)))
    np.savez(join(figdir, "%s_pix_%s.npz"%(savestr, layername)), score_traj=score_traj.numpy())
    if imshow:
        plt.figure(figsize=[Bsize*2, 2.3])
        plt.imshow(mtg)
        plt.axis("off")
        plt.show()
    if saveimg:
        os.makedirs(join(figdir, "img"), exist_ok=True)
        if saveImgN is None:
            save_imgtsr(finimgs, figdir=join(figdir, "img"), savestr="%s"%(savestr))
        else:
            save_imgtsr(finimgs[:saveImgN,:,:,:], figdir=join(figdir, "img"), savestr="%s"%(savestr))
    return finimgs, mtg, score_traj


def corr_GAN_visualize(G, scorer, CNNnet, preprocess, layername, tfms=[],
    lr=0.01, imgfullpix=224, MAXSTEP=100, Bsize=4, saveImgN=None, use_adam=True, langevin_eps=0, 
    savestr="", figdir="", imshow=False, PILshow=False, verbose=True, saveimg=False, score_mode="dot", maximize=True):
    """ Visualize the features carried by the scorer.  """
    scorer.mode = score_mode
    score_sgn = -1 if maximize else 1
    z = 0.5*torch.randn([Bsize, 4096]).cuda()
    z.requires_grad_(True)
    optimizer = Adam([z], lr=lr) if use_adam else SGD([z], lr=lr)
    tfms_f = compose(tfms)
    score_traj = []
    pbar = tqdm(range(MAXSTEP))
    for step in pbar:
        x = G.visualize(z, scale=1.0)
        ppx = preprocess(x)
        ppx = F.interpolate(ppx, [imgfullpix, imgfullpix], mode="bilinear", align_corners=True)
        optimizer.zero_grad()
        CNNnet(tfms_f(ppx))
        score = score_sgn * scorer.corrfeat_score(layername)
        score.sum().backward()
        z.grad = z.norm(dim=1, keepdim=True) / z.grad.norm(dim=1, keepdim=True) * z.grad  # this is a gradient normalizing step 
        optimizer.step()
        score_traj.append(score.detach().clone().cpu())
        if langevin_eps > 0: 
            # if > 0 then add noise to become Langevin gradient descent jump minimum
            z.data.add_(torch.randn(z.shape, device="cuda") * langevin_eps)
        if verbose and step % 10 == 0:
            print("step %d, score %s"%(step, " ".join("%.2f" % s for s in score_sgn * score)))
        pbar.set_description("step %d, score %s"%(step, " ".join("%.2f" % s for s in score_sgn * score)))

    final_score = score_sgn * score.detach().clone().cpu()
    del score
    torch.cuda.empty_cache()
    if maximize:
        idx = torch.argsort(final_score, descending=True)
    else:
        idx = torch.argsort(final_score, descending=False)
    score_traj = score_sgn * torch.stack(tuple(score_traj))[:, idx]
    finimgs = x.detach().clone().cpu()[idx, :, :, :]  # finimgs are generated by z before preprocessing.
    print("Final scores %s"%(" ".join("%.2f" % s for s in final_score[idx])))
    mtg = ToPILImage()(make_grid(finimgs))
    if PILshow: mtg.show()
    mtg.save(join(figdir, "%s_G_%s.png"%(savestr, layername)))
    np.savez(join(figdir, "%s_G_%s.npz"%(savestr, layername)), z=z.detach().cpu().numpy(), score_traj=score_traj.numpy())
    if imshow:
        plt.figure(figsize=[Bsize*2, 2.3])
        plt.imshow(mtg)
        plt.axis("off")
        plt.show()
    if saveimg:
        os.makedirs(join(figdir, "img"), exist_ok=True)
        if saveImgN is None:
            save_imgtsr(finimgs, figdir=join(figdir, "img"), savestr="%s"%(savestr))
        else:
            save_imgtsr(finimgs[:saveImgN,:,:,:], figdir=join(figdir, "img"), savestr="%s"%(savestr))
            mtg_sel = ToPILImage()(make_grid(finimgs[:saveImgN,:,:,:]))
            mtg_sel.save(join(figdir, "%s_G_%s_best.png" % (savestr, layername)))
    return finimgs, mtg, score_traj

#%%
if __name__ == "__main__":
    # Prepare the networks
    VGG = models.vgg16(pretrained=True)
    VGG.requires_grad_(False)
    VGG.features.cuda()
    G = upconvGAN("fc6").cuda()
    G.requires_grad_(False)
    #%%
    mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
    Pasupath = r"N:\Stimuli\2019-Manifold\pasupathy-wg-f-4-ori"
    Gaborpath = r"N:\Stimuli\2019-Manifold\gabor"
    Animal = "Beto"
    # ManifDyn = loadmat(join(mat_path, Animal + "_ManifPopDynamics.mat"), struct_as_record=False, squeeze_me=True)['ManifDyn']
    MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    # %% Final Batch Processing of all Exp.
    figdir = r"S:\corrFeatTsr\VGG_featvis";
    Animal = "Alfa"; Expi = 3
    for Animal in ["Alfa",]: # "Beto"
        for Expi in range(27,46+1):
            D = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_EM_corrTsr.npz"%(Animal, Expi)), allow_pickle=True)
            scorer = CorrFeatScore()
            scorer.load_from_npy(D, VGG, netname="vgg16", thresh=4, layers=[])
            savefn = "%s_Exp%d_EM_corrVis"%(Animal, Expi)
            imgs5, mtg5, score_traj5 = corr_GAN_visualize(G, scorer, VGG.features, preprocess, "conv5_3", lr=0.080, imgfullpix=224, MAXSTEP=75, Bsize=4, savestr=savefn, figdir=figdir)
            imgs4, mtg4, score_traj4 = corr_GAN_visualize(G, scorer, VGG.features, preprocess, "conv4_3", lr=0.025, imgfullpix=224, MAXSTEP=75, Bsize=4, savestr=savefn, figdir=figdir)
            imgs3, mtg3, score_traj3 = corr_GAN_visualize(G, scorer, VGG.features, preprocess, "conv3_3", lr=0.01, imgfullpix=224, MAXSTEP=75, Bsize=4, savestr=savefn, figdir=figdir)
    #%% Try to use GradCam, to understand output for a single input.

    #%% Backprop based feature visualization
    Animal = "Beto"
    Expi = 11
    # for Expi in range(27,46+1):
    D = np.load(join(r"S:\corrFeatTsr", "%s_Exp%d_EM_corrTsr.npz" % (Animal, Expi)), allow_pickle=True)
    scorer = CorrFeatScore()
    scorer.load_from_npy(D, VGG, netname="vgg16", thresh=4, layers=[])
    img = ReprStats[Expi-1].Evol.BestBlockAvgImg
    x = transforms.ToTensor()(img)
    x = preprocess(x.unsqueeze(0))
    x = F.interpolate(x, [224,224])
    x.requires_grad_(True)
    VGG.features(x.cuda())
    score = scorer.corrfeat_score("conv5_3")
    #%% Visualize feature related to these masks
    GradMask = x.grad.clone()
    GradMask /= GradMask.abs().max()
    GradMaskNP = GradMask[0].permute([1,2,0]).numpy()
    # GradMaskNP_rsz = resize(img, (224,224))
    maskimg = resize(img, (224,224))*(np.minimum(1, 3*np.abs(GradMaskNP).mean(axis=2,keepdims=True)))
    plt.imshow(maskimg)
    plt.show()
    #%%
    Animal="Alfa"; Expi = 3
    D = np.load(join("S:\corrFeatTsr","%s_Exp%d_EM_corrTsr.npz"%(Animal,Expi)), allow_pickle=True)
    scorer = CorrFeatScore()
    scorer.load_from_npy(D, VGG, netname="vgg16", thresh=4, layers=[])
    #%% Maximize to the scorer using pixel parametrization.
    imgfullpix = 224
    MAXSTEP = 50
    Bsize = 4
    x = 0.5+0.01*torch.rand((Bsize,3,imgfullpix,imgfullpix)).cuda()
    x.requires_grad_(True)
    optimizer = SGD([x], lr=0.01)
    for step in range(MAXSTEP):
        ppx = preprocess(x)
        optimizer.zero_grad()
        VGG.features(ppx)
        score = -scorer.corrfeat_score("conv3_3")
        score.sum().backward()
        x.grad = x.norm() / x.grad.norm() * x.grad
        optimizer.step()
        if step % 10 == 0:
            print("step %d, score %.s"%(step, -score.item()))

    ToPILImage()(torch.clamp(x[0], 0, 1).cpu()).show()
    del score
    torch.cuda.empty_cache()
    #%%
    def preprocess(img, res=224):
        img = F.interpolate(img, [res,res], mode="bilinear", align_corners=True)
        img = torch.clamp(img,0,1)
        img = gaussian_blur2d(img, (5,5), sigma=(3, 3))
        img = (img - RGBmean.to(img.device)) / RGBstd.to(img.device)
        return img
    #%
    imgfullpix = 224
    MAXSTEP = 50
    Bsize = 4
    G = upconvGAN("fc6").cuda()
    G.requires_grad_(False)
    z = 0.5*torch.randn([Bsize, 4096]).cuda()
    z.requires_grad_(True)
    optimizer = Adam([z], lr=0.1)
    for step in range(MAXSTEP):
        x = G.visualize(z, scale=1.0)
        ppx = preprocess(x)
        optimizer.zero_grad()
        VGG.features(ppx)
        score = -scorer.corrfeat_score("conv5_3")
        score.sum().backward()
        z.grad = z.norm() / z.grad.norm() * z.grad
        optimizer.step()
        if step % 10 ==0:
            print("step %d, score %s"%(step, " ".join("%.1f"%s for s in -score)))
    # ToPILImage()(torch.clamp(x[0],0,1).cpu()).show()
    ToPILImage()(make_grid(x).cpu()).show()
    del score
    torch.cuda.empty_cache()
    #%%
