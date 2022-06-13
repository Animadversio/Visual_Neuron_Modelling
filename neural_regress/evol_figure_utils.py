import numpy as np
from build_montages import make_grid_np
import matplotlib.pyplot as plt
from glob import glob
from scipy.io import loadmat
from os.path import join
import re
from scipy.stats import sem
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#%%
evoldir = r"N:\Stimuli\2021-EvolDecomp\2022-05-11-Beto-01\2022-05-11-13-23-29"
imgfp_vec = sorted(glob(join(evoldir, "*.bmp")))
evoldata = loadmat(join(evoldir, "Evol_ScoreImgTraj.mat"))
#%%
outdir = r"E:\OneDrive - Harvard University\CCN2022_shortpaper\figures"
#%%
threadid = 1
scorevec_thread = evoldata["score_col"][0, threadid-1][:,0]
imgfp_thread = evoldata['imgfp_col'][0, threadid-1]
imgpatt = re.compile("block(\d*)_thread")
blockvec_thread = np.array([int(imgpatt.findall(imgfn)[0]) for imgfn in imgfp_thread])
blockarr = range(min(blockvec_thread),max(blockvec_thread)+1)
meanarr = np.array([np.mean(scorevec_thread[blockvec_thread==blocki]) for blocki in blockarr])
semarr = np.array([sem(scorevec_thread[blockvec_thread==blocki]) for blocki in blockarr])
#%%
figh, ax = plt.subplots(figsize=(5,4))
plt.scatter(blockvec_thread,scorevec_thread,alpha=0.3)
plt.plot(blockarr, meanarr, 'k-')
plt.fill_between(blockarr, meanarr-semarr, meanarr+semarr,alpha=0.5)
plt.ylabel("Spike rate")
plt.xlabel("Generations")
# plt.title("Evolution Trajectory prefchan %02d, %.1f deg pos [%.1f %.1f], thread %d"%\
#           (pref_chan,imgsize,imgpos[0],imgpos[1],threadid))
figh.savefig(join(outdir, "evol_score_traj.pdf"), dpi=300)
figh.savefig(join(outdir, "evol_score_traj.png"), dpi=300)
plt.show()
#%%
imgfp_vec = sorted(glob(join(evoldir, f"*_thread{threadid-1:03d}_*.bmp")))
#%%
imgfp2show = []
for blocki in range(1, max(blockvec_thread)+1, 4):
    idxs = np.where(blockvec_thread == blocki)[0]
    idx = np.random.choice(idxs, )
    imgfp2show.append(imgfp_vec[idx])
    # imgpatt = re.compile(f"block{blocki:03d}_thread")
#%%
img_col = [plt.imread(imgfp) for imgfp in imgfp2show]
#%%
mtg = make_grid_np(img_col, nrow=7, padding=2, pad_value=0, rowfirst=True)
plt.imsave(join(outdir, "image_evol_traj_mtg.png"), mtg, dpi=300)
