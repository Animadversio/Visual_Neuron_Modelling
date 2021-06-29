from CorrFeatTsr_visualize_lib import CorrFeatScore, corr_visualize, corr_GAN_visualize, preprocess, ToPILImage, make_grid
from torchvision.transforms import ToTensor
from tqdm import tqdm
import os
from os.path import join
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from GAN_utils import upconvGAN
import torchvision.models as models
from torch.optim import SGD, Adam
from imageio import imread
from skimage.transform import resize

refdir = r"N:\Stimuli\2020-CosineEvol\RefCollection"
os.listdir(refdir)
#%%

G = upconvGAN('fc6').cuda().eval()
G.requires_grad_(False)
CNNnet = models.vgg16().cuda().eval()
CNNnet.requires_grad_(False)
netname = "vgg16"
#%%
imgfullpix = 256
MAXSTEP = 150

#%%
def create_rand_recordmask(recordN, acttsr, nonzero_unit=False):
    if nonzero_unit:
        activ_idx = acttsr.cpu().flatten().nonzero()
        if recordN > activ_idx.numel():
            print("Only %d units are active, select all!")
            recordN = activ_idx.numel()
            rec_idx = activ_idx.numpy()
        else:
            rec_idx = np.random.choice(activ_idx.squeeze(), size=recordN, replace=False)
    else:
        rec_idx= np.random.choice(acttsr.numel(), size=recordN, replace=False)
    rec_idx = torch.tensor(rec_idx)
    recmask = np.zeros(acttsr.numel(), dtype=np.bool)
    # recmask.fill(np.nan)
    recmask[rec_idx] = 1
    recmask = recmask.reshape(acttsr.shape)
    assert (np.count_nonzero(recmask)) == recordN
    recmasktsr = torch.tensor(recmask)
    recvectsr = acttsr[recmasktsr.cuda()]
    # assert (recacttsr).count_nonzero() == recordN
    return recmasktsr, recvectsr

figdir = r"O:\ThesisProposalMeeting\CosineCNNDemo"
targnm, targname = "objects-familiar-11.bmp", "banana"
targnm, targname = 'Blue-GrayGnatcatcher_sq.jpg', "bird"
#'07_face_human_13_sh.jpg'#'Blue-GrayGnatcatcher_sq.jpg'#)#"Shelled-Peanuts_sq.jpg"
targimg = imread(join(refdir, targnm))
targimg_rsz = resize(targimg, [imgfullpix, imgfullpix])
targimgtsr = ToTensor()(targimg_rsz).float().unsqueeze(0)
score_mode = "MSEmask"
layername = "fc7" # "fc8"
for layername in ["conv1_2", "conv2_2", "conv3_3", "conv4_3", "conv5_3", "fc6", "fc7", "fc8", ]:
    recordN = 50
    savenm = "%s_%s_%s_pop%d_%s"%(targname, netname, layername, recordN, score_mode)
    scorer = CorrFeatScore()
    scorer.register_hooks(CNNnet, layername, netname="vgg16")
    targimg_rsz = resize(targimg, [imgfullpix, imgfullpix])
    targimgtsr = ToTensor()(targimg_rsz).float().unsqueeze(0)
    CNNnet(preprocess(targimgtsr).cuda())
    acttsr = scorer.feat_tsr[layername].squeeze()
    recmasktsr, recvectsr = create_rand_recordmask(recordN, acttsr, nonzero_unit=True)
    scorer.register_weights({layername: acttsr}, {layername: recmasktsr})
    #%
    if score_mode=="corrmask":
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, CNNnet, preprocess, layername,
            lr=0.1, imgfullpix=imgfullpix, MAXSTEP=MAXSTEP, Bsize=10, saveImgN=4, use_adam=True, langevin_eps=0,
            savestr=savenm, figdir=figdir, imshow=True, verbose=True, saveimg=True, score_mode=score_mode, maximize=True)
    elif score_mode=="MSEmask":
        finimgs, mtg, score_traj = corr_GAN_visualize(G, scorer, CNNnet, preprocess, layername,
            lr=0.1, imgfullpix=imgfullpix, MAXSTEP=MAXSTEP, Bsize=10, saveImgN=4, use_adam=False, langevin_eps=0,
            savestr=savenm, figdir=figdir, imshow=True, verbose=True, saveimg=True, score_mode=score_mode, maximize=False)

#%%
for layername in ["conv1_2", "conv2_2", "conv3_3", "conv4_3", "conv5_3", "fc6", "fc7", "fc8", ]:
    recordN = 500
    savenm = "%s_%s_%s_pop%d_%s"%(targname, netname, layername, recordN, score_mode)
    scorer = CorrFeatScore()
    scorer.register_hooks(CNNnet, layername, netname="vgg16")
    CNNnet(preprocess(targimgtsr).cuda())
    acttsr = scorer.feat_tsr[layername].squeeze()
    print("%s:%d/%d %f"%(layername, recordN, acttsr.numel(), recordN/acttsr.numel()))
#%%
# corr_visualize(scorer, CNNnet, preprocess, layername,
#     lr=0.01, imgfullpix=224, MAXSTEP=100, Bsize=4, saveImgN=None, use_adam=False, langevin_eps=0,
#     savestr="", figdir="", imshow=True, verbose=True, saveimg=False, score_mode="MSEmask", maximize=False)
#%%
Bsize = 5
lr = 0.05
use_adam = False
verbose = True
langevin_eps = 0.0
scorer.mode = "MSE"
z = 0.5 * torch.randn([Bsize, 4096]).cuda()
z.requires_grad_(True)
optimizer = Adam([z], lr=lr) if use_adam else SGD([z], lr=lr)
score_traj = []
pbar = tqdm(range(MAXSTEP))
for step in pbar:
    x = G.visualize(z, scale=1.0)
    ppx = preprocess(x)
    ppx = F.interpolate(ppx, [imgfullpix, imgfullpix], mode="bilinear", align_corners=True)
    optimizer.zero_grad()
    CNNnet(ppx)
    score = scorer.corrfeat_score(layername)
    score.sum().backward()
    z.grad = z.norm(dim=1, keepdim=True) / z.grad.norm(dim=1,
                                                       keepdim=True) * z.grad  # this is a gradient normalizing step
    optimizer.step()
    score_traj.append(score.detach().clone().cpu())
    if langevin_eps > 0:
        # if > 0 then add noise to become Langevin gradient descent jump minimum
        z.data.add_(torch.randn(z.shape, device="cuda") * langevin_eps)
    if verbose and step % 10 == 0:
        print("step %d, score %s" % (step, " ".join("%.2f" % s for s in -score)))
    pbar.set_description("step %d, score %s" % (step, " ".join("%.2f" % s for s in -score)))

final_score = -score.detach().clone().cpu()
del score
torch.cuda.empty_cache()
idx = torch.argsort(final_score, descending=True)
score_traj = -torch.stack(tuple(score_traj))[:, idx]
finimgs = x.detach().clone().cpu()[idx, :, :, :]  # finimgs are generated by z before preprocessing.
print("Final scores %s" % (" ".join("%.2f" % s for s in final_score[idx])))
mtg = ToPILImage()(make_grid(finimgs))
mtg.show()
# mtg.save(join(figdir, "%s_G_%s.png" % (savestr, layername)))
# np.savez(join(figdir, "%s_G_%s.npz" % (savestr, layername)), z=z.detach().cpu().numpy(), score_traj=score_traj.numpy())


