"""
Define the class that manage Monkey experiment data and the Experiment records
Updated 2022/Mar.8th
Very useful data interface in python.
"""
import h5py
import sys
import os
from os.path import join
import shutil
from glob import glob
from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt
import csv
from time import time
import pandas as pd
# ExpTable = pd.read_excel(r"D:\Network_Data_Sync\ExpSpecTable_Augment.xlsx")
rspPath = r"S:\Data-Ephys-MAT"
NetrspPath = r"N:\Data-Ephys-MAT"
#%%
class ExpData:
    """A class to handle loading matlab experimental data, and do basic processing on it"""
    def __init__(self, ephysFN, stimuli_path):
        self.ephysFN = ephysFN
        self.stimuli = stimuli_path

    def load_mat(self):
        Rspfns = sorted(glob(join(rspPath, self.ephysFN + "*")))
        if len(Rspfns) == 0:
            """Mat file not existing in local folder, sync it from network folder."""
            print("Start copying mat file from network folder.")
            NetRspfns = sorted(glob(join(NetrspPath, self.ephysFN + "*")))
            for fn in NetRspfns:
                shutil.copy2(fn, rspPath)
            print("Finish copying %d mat file from network folder. %s" %
                  (len(NetRspfns), NetRspfns))
            Rspfns = sorted(glob(join(rspPath, self.ephysFN + "*")))
        Rspfns = [fn for fn in Rspfns if "format" in fn]
        assert len(Rspfns) > 0
        rspData = h5py.File(Rspfns[0])
        self.matfile = rspData
        self.spikeID = rspData['meta']['spikeID'][:]
        self.rasters = rspData['rasters']
        self.lfps = rspData['rasters']
        if 'prefChan' in rspData['Trials']['TrialRecord']['User']:
            self.pref_chan = int(rspData['Trials']['TrialRecord']['User']['prefChan'][0, 0])
        else:
            self.pref_chan = None
        # Extremely useful code snippet to solve the problem
        imgnms_refs = np.array(rspData['Trials']['imageName']).flatten()
        self.imgnms = np.array([''.join(chr(i) for i in rspData[ref]) for ref in imgnms_refs])
        print("Total %d channels recorded." % self.spikeID.shape[1])
        print("Total %d images shown." % len(self.imgnms))
        try:
            pref_chan = self.matfile[self.matfile["Trials/TrialRecord/User"]["evoConfiguration"][0, 0]][:]
            pref_chan = int(pref_chan)
            assert pref_chan == self.pref_chan
            imgsize_deg = self.matfile[self.matfile["Trials/TrialRecord/User"]["evoConfiguration"][2, 0]][:].item()
            imgpos = self.matfile[self.matfile["Trials/TrialRecord/User"]["evoConfiguration"][1, 0]][:]
            imgpos = np.squeeze(imgpos)
            self.imgsize_deg = imgsize_deg
            self.imgpos = imgpos
        except KeyError:
            width = self.matfile[self.matfile["Trials/width"][0, 0]][:].squeeze()
            self.imgsize_deg = round(2 * width[0] / 40) / 2
            self.imgpos = self.matfile["Trials/TrialRecord/User/xy"][:, 0].copy()

        try:
            pref_unit = self.matfile[self.matfile["Trials/TrialRecord/User"]["evoConfiguration"][3, 0]][:]
            pref_unit = int(pref_unit)
            self.pref_unit = pref_unit
        except (IndexError, KeyError):
            self.pref_unit = 1

        Animal = "".join([chr(i) for i in self.matfile["Trials/MLConfig/SubjectName"][:]])
        self.Animal = Animal
        print("%s Preferred unit, Ch%d (unit%d) with image %.1f deg at [%.1f,%.1f]"%
              (Animal, self.pref_chan, self.pref_unit, self.imgsize_deg, self.imgpos[0], self.imgpos[1]))

    def find_generated(self):
        gen_rows = ['gen' in fn and 'block' in fn and not fn[:2].isnumeric() for fn in self.imgnms]
        nat_rows = [not i for i in gen_rows]
        self.gen_rows_idx = [i for i, b in enumerate(gen_rows) if b]
        self.nat_rows_idx = [i for i, b in enumerate(nat_rows) if b]
        self.gen_fns = [self.imgnms[i] for i, b in enumerate(gen_rows) if b]
        self.nat_fns = [self.imgnms[i] for i, b in enumerate(nat_rows) if b]
        self.gen_rows = gen_rows
        self.nat_rows = nat_rows
        nimgs = self.matfile["Trials/block"].size
        self.generations = [int(self.matfile[self.matfile["Trials/block"][0, i]][0, 0]) for i in range(nimgs)]
        self.generations = np.array(self.generations)

    def load_codes(self):
        # allfns = os.listdir(self.stimuli)
        matfns = sorted(glob(join(self.stimuli, "*.mat")))  # [fn if ".mat" in fn else [] for fn in allfns]
        print("Found %d code mat files, reading!" % len(matfns))
        # imgfns = sorted(glob(join(self.stimuli, "*.jpg")))  # [fn if ".mat" in fn else [] for fn in allfns]
        # % Load the Codes mat file
        codes_all = np.array([])
        img_id_all = []
        for nm in matfns:
            data = loadmat(nm)
            codes = data['codes']
            codes_all = np.vstack([codes_all, codes]) if codes_all.size else codes
            img_id = [arr[0] for arr in list(data['ids'][0])]
            img_id_all.extend(img_id)
        self.codes_all = codes_all
        self.gen_img_id_all = img_id_all

    def close(self):
        self.matfile.close()
# #%%
# ftr = (ExpTable.Expi == 12) & ExpTable.expControlFN.str.contains("generate")
# EData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
# #%%
# ftr = (ExpTable.Expi == 12) & ExpTable.expControlFN.str.contains("selectivity")
# print(ExpTable.comments[ftr].str.cat())
# MData = ExpData(ExpTable[ftr].ephysFN.str.cat(), ExpTable[ftr].stimuli.str.cat())
# MData.load_mat()
# #%%
# MData.close()
# EData.close()
#%% Dev zone
#%% Building up script
# sys.path.append(r"D:\Github\Activation-Maximization-for-Visual-System")
# from torch_net_utils import load_caffenet, load_generator, visualize
# net = load_caffenet()
# Generator = load_generator()
# #%%
# rspPath = r"D:\Network_Data_Sync\Data-Ephys-MAT"#r"N:\Data-Ephys-MAT"#
# EphsFN = "Beto64chan-30102019-001"  # "Beto64chan-11112019-006" #
# stimPath = r"N:\Stimuli\2019-Manifold\beto-191030a\backup_10_30_2019_10_15_31"
#
# allfns = os.listdir(stimPath)
# matfns = sorted(glob(join(stimPath, "*.mat")))  # [fn if ".mat" in fn else [] for fn in allfns]
# imgfns = sorted(glob(join(stimPath, "*.jpg")))  # [fn if ".mat" in fn else [] for fn in allfns]
# #%% Load the Codes mat file
# data = loadmat(matfns[1])
# codes = data['codes']
# img_id = [arr[0] for arr in list(data['ids'][0])] # list of ids
# #%%
#
# Rspfns = sorted(glob(join(rspPath, EphsFN+"*")))
# rspData = h5py.File(Rspfns[1])
# spikeID = rspData['meta']['spikeID']
# rsp = rspData['rasters']
# # Extremely useful code snippet to solve the problem
# imgnms_refs = np.array(rspData['Trials']['imageName']).flatten()
# imgnms = np.array([''.join(chr(i) for i in rspData[ref]) for ref in imgnms_refs])
# #%%
# prefchan_idx = np.nonzero(spikeID[0, :] == 26)[0] - 1
# prefrsp = rsp[:, :, prefchan_idx]  # Dataset reading takes time
# scores = prefrsp[:, 50:, :].mean(axis=1) - prefrsp[:, :40, :].mean(axis=1)
