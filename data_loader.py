"""
Newer API for data loading for Evolution, Manifold dataset.
Using the already formulated EStats and MStats structs from .mat files.

for older one loading from full exp record see load_neural_data.
"""
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
def load_score_mat(EStats, MStats, Expi, ExpType, wdws=[(50,200)], stimdrive="N"):
    """
    Unified interface to load image, response pair for Evolution, Manifold experiment pair. 

    :param EStats: loaded saved mat struct. 
        EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    :param MStats: loaded saved mat struct. 
        MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
    :param Expi: Int. Experiment id, number start from 1. 
    :param ExpType: A string from these options ["Evol", "Manif_avg", "Manif_sgtr", 
        "EvolRef_avg", "EvolRef_sgtr", "Gabor_avg", "Gabor_sgtr", "Pasu_avg", "Pasu_sgtr"]
        ExpType containing `sgtr` will return scores in `scorecol` format 
        ExpType containing `avg` will return scores in `score_vect` format 

    :param wdws: a list of tuple. Not implemented yet. used to specify the window to define the scores. 
    :param stimdrive: "N", "S" etc. the drive to load the stimuli images. 


    :return: Two types of return format. 
        score_vect, imgfullpath_vect
        or 
        scorecol, imgfullpath_vect
        
        score_vect: np float 1d array of scores. Same length as `imgfullpath_vect`. 
        scorecol: a list of list of scores, each list is the score in each trial for an image. 
            Same length as `imgfullpath_vect`. 
        imgfullpath_vect: a list of string of full path to the images. can be loaded via imread. 

    Example code
    ```
    Expi = 3
    score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50,200)])
    score_vect_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50,200)])
    scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50,200)])
    ```
    """
    if ExpType == "Evol":
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
        stimpath = stimpath.replace(r"N:", stimdrive+":")
        imgnms_col = glob(stimpath+"\\*")  # + glob(Pasupath+"\\*") + glob(Gaborpath+"\\*")
        imgfullpath_vect = [[path for path in imgnms_col if imgnm in path][0] for imgnm in imgnm_vect]
        return score_vect, imgfullpath_vect
    elif "Manif" in ExpType:
        ui = EStats[Expi - 1].evol.unit_in_pref_chan  # unit id in the pref chan, matlab convention start w/ 1
        if MStats[Expi - 1].manif.psth.shape == (11, 11):
            psth = MStats[Expi - 1].manif.psth.reshape(-1)
            idx_vect = MStats[Expi - 1].manif.idx_grid.reshape([-1])
        else:
            psth = MStats[Expi - 1].manif.psth[0].reshape(-1)  # if multiple Manif is done, pick the first one PC23.
            idx_vect = MStats[Expi - 1].manif.idx_grid[0].reshape([-1])
        nunit = np.alen(MStats[Expi - 1].units.pref_chan_id)  # use alen instead of len to get len(10) work
        # if psth[0].ndim == 3:
        #     nunit = psth[0].shape[0]
        # else:
        #     nunit = 1
        psthlist = list(np.reshape(P, [nunit, 200, -1]) for P in psth)
        scorecol = [np.mean(P[ui - 1, 50:200, :], axis=0).astype(np.float) for P in psthlist]
        idx_vect = [np.array(idx).reshape(-1) for idx in idx_vect]  # in case only 1 index
        imgnm_vect = [MStats[Expi - 1].imageName[idx[0] - 1] for idx in idx_vect]# note offset the index bt
            # one, since it's matlab convention.
        stimpath = MStats[Expi - 1].meta.stimuli
        stimpath = stimpath.replace(r"\\storage1.ris.wustl.edu\crponce\Active", r"N:")
        stimpath = stimpath.replace(r"N:", stimdrive+":")
        imgnms = glob(stimpath + "\\*")  # + glob(Pasupath + "\\*") + glob(Gaborpath + "\\*")
        imgfullpath_vect = [[path for path in imgnms if imgnm in path][0] for imgnm in imgnm_vect]
        if ExpType == "Manif_avg":
            score_vect = np.array([np.mean(score) for score in scorecol]).astype(np.float)
            return score_vect, imgfullpath_vect
        elif ExpType == "Manif_sgtr":
            return scorecol, imgfullpath_vect
    elif "EvolRef" in ExpType:
        impaths = EStats[Expi - 1].ref.impaths_chr.reshape(-1)
        imgfullpath_vect = [impth.replace(r"N:", stimdrive+":") for impth in impaths]
        psth = EStats[Expi - 1].ref.psth_arr
        # ui=1; nunit = 1
        scorecol = [np.mean(P[50:200, :], axis=0).astype(np.float) for P in psth]
        if ExpType == "EvolRef_avg":
            score_vect = np.array([np.mean(score) for score in scorecol]).astype(np.float)
            return score_vect, imgfullpath_vect
        elif ExpType == "EvolRef_sgtr":
            return scorecol, imgfullpath_vect
    elif "Gabor" in ExpType:
        if MStats[Expi - 1].ref.didGabor == 0:  # Gabor is not done
            scorecol, imgfullpath_vect = [], []
        else:  # Gabor is done
            ui = EStats[Expi - 1].evol.unit_in_pref_chan
            nunit = np.alen(MStats[Expi - 1].units.pref_chan_id)  # use alen instead of len to get len(10) work
            psth = MStats[Expi - 1].ref.gab_psths.reshape(-1)
            idx_vect = MStats[Expi - 1].ref.gab_idx_grid.reshape([-1])
            # validmsk = [np.alen(I) != 0 for I in idx_vect]  # get rid of invalid entries
            validmsk = [(np.isscalar(I) or len(I) != 0) for I in idx_vect]  # newer version, without np.alen
            psth, idx_vect = psth[validmsk], idx_vect[validmsk]
            psthlist = list(np.reshape(P, [nunit, 200, -1]) for P in psth)
            scorecol = [np.mean(P[ui - 1, 50:200, :], axis=0).astype(np.float) for P in psthlist]  # avg over time
            idx_vect = [np.array(idx).reshape(-1) for idx in idx_vect]  # in case only 1 index
            imgnm_vect = [MStats[Expi - 1].imageName[idx[0] - 1] for idx in idx_vect]  # note offset the index bt
                # one, since it's matlab convention.
            stimpath = Gaborpath.replace(r"\\storage1.ris.wustl.edu\crponce\Active", r"N:")
            stimpath = stimpath.replace(r"N:", stimdrive + ":")
            imgnms = glob(stimpath + "\\*")  # + glob(Pasupath + "\\*") + glob(Gaborpath + "\\*")
            imgfullpath_vect = [[path for path in imgnms if imgnm in path][0] for imgnm in imgnm_vect]
        if ExpType == "Gabor_avg":
            score_vect = np.array([np.mean(score) for score in scorecol]).astype(np.float)
            return score_vect, imgfullpath_vect
        elif ExpType == "Gabor_sgtr":
            return scorecol, imgfullpath_vect
    elif "Pasu" in ExpType:  # Pasu is not done
        if MStats[Expi - 1].ref.didPasu == 0:  # Gabor is not done
            scorecol, imgfullpath_vect = [], []
        else:  # Pasu is done
            ui = EStats[Expi - 1].evol.unit_in_pref_chan
            nunit = np.alen(MStats[Expi - 1].units.pref_chan_id)  # use alen instead of len to get len(10) work
            psth = MStats[Expi - 1].ref.pasu_psths.reshape(-1)
            idx_vect = MStats[Expi - 1].ref.pasu_idx_grid.reshape([-1])
            # validmsk = [np.alen(I) != 0 for I in idx_vect]  # get rid of invalid entries
            validmsk = [(np.isscalar(I) or len(I) != 0) for I in idx_vect]  # newer version, without np.alen
            psth, idx_vect = psth[validmsk], idx_vect[validmsk]
            psthlist = list(np.reshape(P, [nunit, 200, -1]) for P in psth)
            scorecol = [np.mean(P[ui - 1, 50:200, :], axis=0).astype(np.float) for P in psthlist]  # avg over time
            idx_vect = [np.array(idx).reshape(-1) for idx in idx_vect]  # in case only 1 index
            imgnm_vect = [MStats[Expi - 1].imageName[idx[0] - 1] for idx in idx_vect]  # note offset the index bt
                # one, since it's matlab convention.
            stimpath = Pasupath.replace(r"\\storage1.ris.wustl.edu\crponce\Active", r"N:")
            stimpath = stimpath.replace(r"N:", stimdrive + ":")
            imgnms = glob(stimpath + "\\*")  # + glob(Pasupath + "\\*") + glob(Gaborpath + "\\*")
            imgfullpath_vect = [[path for path in imgnms if imgnm in path][0] for imgnm in imgnm_vect]
        if ExpType == "Pasu_avg":  # return mean score array
            score_vect = np.array([np.mean(score) for score in scorecol]).astype(np.float)
            return score_vect, imgfullpath_vect
        elif ExpType == "Pasu_sgtr":  # return collection of scores
            return scorecol, imgfullpath_vect

def test_load_score_mat(DRIVE="S"):
    """Test all files (esp. image stimuli are loadable.)"""
    missing_fns = []
    def record_missing(imgfullpaths):
        for pth in imgfullpaths:
            if not os.path.exists(pth):
                print(pth, "not exist")
                missing_fns.append(pth)
    for Animal in ["Alfa", "Beto"]:
        MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
        EStats = \
            loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True,
                    chars_as_strings=True)[
                'EStats']
        for Expi in range(1, len(EStats) + 1):
            scorecol, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50, 200)])
            scorecol, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "EvolRef_sgtr", wdws=[(50, 200)],
                                                          stimdrive=DRIVE)
            record_missing(imgfullpath_vect)
            scorecol, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Gabor_sgtr", wdws=[(50, 200)],
                                                        stimdrive=DRIVE)
            record_missing(imgfullpath_vect)
            scorecol, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Pasu_sgtr", wdws=[(50, 200)],
                                                        stimdrive=DRIVE)
            record_missing(imgfullpath_vect)
    assert len(missing_fns) == 0, "Check the missing stimuli. data loading not fully successful"
# ui = EStats[Expi - 1].evol.unit_in_pref_chan # unit id in the pref chan
# psth = MStats[Expi-1].manif.psth.reshape(-1)
# if psth[0].ndim == 3:
#     nunit = psth[0].shape[0]
# else:
#     nunit = 1
# psthlist = list(np.reshape(P, [nunit, 200, -1]) for P in psth)
# scorecol = [np.mean(P[ui-1, 50:200, :],axis=0).astype(np.float) for P in psthlist]
if __name__ == "__main__":
    from shutil import copyfile
    DRIVE = "S"
    missing_fns = []
    for Animal in ["Alfa", "Beto"]:
        MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
        EStats = \
        loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)[
            'EStats']
        for Expi in range(1, len(EStats)+1):
            # score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)])
            # score_vect_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_avg", wdws=[(50, 200)])
            # scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50, 200)])
            score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "EvolRef_sgtr", wdws=[(50, 200)],
                                                        stimdrive=DRIVE)
            for pth in imgfullpath_vect:
                if not os.path.exists(pth):
                    print(pth, "not exist")
                    missing_fns.append(pth)
            # score_vect_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)])
            scorecol, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Gabor_sgtr", wdws=[(50, 200)],
                                                        stimdrive=DRIVE)
            for pth in imgfullpath_vect:
                if not os.path.exists(pth):
                    print(pth, "not exist")
                    missing_fns.append(pth)
            # score_vect_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Gabor_avg", wdws=[(50, 200)])
            scorecol, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Pasu_sgtr", wdws=[(50, 200)],
                                                        stimdrive=DRIVE)
            for pth in imgfullpath_vect:
                if not os.path.exists(pth):
                    print(pth, "not exist")
                    missing_fns.append(pth)

    for fn in missing_fns:
        copyfile(fn.replace(r"S:", "N:"), fn)

