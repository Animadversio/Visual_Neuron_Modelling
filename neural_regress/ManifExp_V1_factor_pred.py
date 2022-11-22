from os.path import join
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
modelroot = r"E:\OneDrive - Washington University in St. Louis\corrFeatTsr_FactorVis\models"
modelstr = r"resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust_CV"
#%%
Animal = "Alfa"
Expi = 20
data = pkl.load(open(join(modelroot, modelstr, f"{Animal}_Exp{Expi:02d}_factors.pkl"), "rb"))
data['AllStat']['cc_bef_norm_gabor']


#%%
df_sum = pd.read_csv(join(modelroot, modelstr, \
                 "Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF3_CV.csv"))

df_sum.cc_aft_norm_gabor[df_sum.area == "V1"]
df_sum.cc_bef_norm_gabor[df_sum.area == "V1"]