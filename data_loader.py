"""Newer API for data loading. for older one see load_neural_data"""
import os
from os.path import join
import shutil
from glob import glob
from scipy.io import loadmat
import numpy as np
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