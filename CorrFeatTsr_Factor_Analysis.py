"""Post hoc analysis and summary for the CorrFeatTsr analysis"""
from featvis_lib import load_featnet, rectify_tsr, tsr_factorize, tsr_posneg_factorize, vis_feattsr, vis_featvec, \
    vis_feattsr_factor, vis_featvec_point, vis_featvec_wmaps, \
    fitnl_predscore, score_images, CorrFeatScore, preprocess, loadimg_preprocess, show_img, pad_factor_prod
import os
from os.path import join
from glob import glob
import pickle as pkl
from easydict import EasyDict
import numpy as np
import numpy.ma as ma

from scipy.stats import ttest_rel, ttest_ind, pearsonr
import torch
import matplotlib as mpl
import matplotlib.pylab as plt
from data_loader import mat_path, loadmat, load_score_mat
from GAN_utils import upconvGAN
import pandas as pd
import seaborn as sns
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#%%
figroot = "O:\corrFeatTsr_FactorVis"
sumdir = join(figroot, "summary")
exproot = join(figroot, "models")
matdir = r"O:\Mat_Statistics"
#%%
csv_list = glob(join(exproot,"*","*.csv",), recursive=False)
# os.listdir(sumdir)
#%%
csv_dict = {csv_fn.split("\\")[-2]: csv_fn for csv_fn in csv_list}
candidate_dict = {
 'alexnet-conv3_NF3_bdr1_Tthresh_3__nobdr_alex': 'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv3_NF3_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv3_Tthresh_bdr1_NF3.csv',
 'alexnet-conv4_NF3_bdr1_Tthresh_3__nobdr_alex': 'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF3_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF3.csv',
 'alexnet-conv5_NF3_bdr1_Tthresh_3__nobdr_alex': 'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv5_NF3_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv5_Tthresh_bdr1_NF3.csv',
 'resnet50-layer2_NF3_bdr3_Tthresh_3__nobdr_resnet': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer2_NF3_bdr3_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer2_Tthresh_bdr3_NF3.csv',
 'resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF3.csv',
 'resnet50-layer4_NF3_bdr1_pos__nobdr_resnet': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer4_NF3_bdr1_pos__nobdr_resnet\\Both_pred_stats_resnet50-layer4_pos_bdr1_NF3.csv',
 # 'resnet50-layer3_NF5_bdr1_Tthresh_3__nobdr_resnet': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF5_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF5.csv',
 'resnet50_linf8-layer2_NF3_bdr3_Tthresh_3__nobdr_res-robust': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer2_NF3_bdr3_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer2_Tthresh_bdr3_NF3.csv',
 'resnet50_linf8-layer2_NF3_bdr3_Tthresh_5__nobdr_res-robust': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer2_NF3_bdr3_Tthresh_5__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer2_Tthresh_bdr3_NF3.csv',
 'resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF3.csv',
 'resnet50_linf8-layer3_NF3_bdr1_Tthresh_5__nobdr_res-robust': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF3_bdr1_Tthresh_5__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF3.csv',
 'vgg16-conv2_2_NF3_bdr5_pos__nobdr': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv2_2_NF3_bdr5_pos__nobdr\\Both_pred_stats_vgg16-conv2_2_pos_bdr5_NF3.csv',
 'vgg16-conv3_3_NF3_bdr2_Tthresh_3__nobdr': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv3_3_NF3_bdr2_Tthresh_3__nobdr\\Both_pred_stats_vgg16-conv3_3_Tthresh_bdr2_NF3.csv',
 'vgg16-conv4_3_NF3_bdr1_Tthresh_3__nobdr': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF3_bdr1_Tthresh_3__nobdr\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF3.csv',
 'vgg16-conv4_3_NF3_bdr1_Tthresh_5__nobdr': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF3_bdr1_Tthresh_5__nobdr\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF3.csv',
 'vgg16-conv5_3_NF3_bdr1_Tthresh_3__nobdr': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv5_3_NF3_bdr1_Tthresh_3__nobdr\\Both_pred_stats_vgg16-conv5_3_Tthresh_bdr1_NF3.csv',
}
#%% Analyze and summarize statistics

# tab = pd.read_csv(join(sumdir, 'Both_pred_stats_vgg16-conv3_3_bdr3_NF3.csv'))
# summarize_tab(tab)
#%%
def pred_cmp_scatter(tab1, tab2, explab1, explab2, varnm="cc_bef", colorvar="area", stylevar="Animal", masktab=None):
    if masktab is None:
        masktab = tab1
    masktab["area"] = ""
    masktab["area"][masktab.pref_chan <= 32] = "IT"
    masktab["area"][(masktab.pref_chan <= 48) & (masktab.pref_chan >= 33)] = "V1"
    masktab["area"][masktab.pref_chan >= 49] = "V4"
    figh = plt.figure(figsize=(5.5,5))
    sns.scatterplot(x=tab1[varnm], y=tab2[varnm], hue=masktab[colorvar], style=masktab[stylevar])
    plt.ylabel(explab2);plt.xlabel(explab1)
    plt.gca().set_aspect('equal', adjustable='box')  # datalim
    cc = ma.corrcoef(ma.masked_invalid(tab1[varnm]), ma.masked_invalid(tab2[varnm]))[0, 1]
    # cc = np.corrcoef(tab1[varnm], tab2[varnm])
    tval, pval = ttest_rel(np.arctanh(tab1[varnm]), np.arctanh(tab2[varnm]), nan_policy='omit') # ttest: exp1 - exp2
    plt.title("Linear model prediction comparison\ncc %.3f t test(Fisher z) %.2f (%.1e)"%(cc, tval, pval))
    plt.savefig(join(sumdir, "models_pred_cmp_%s_%s.png"%(explab1, explab2)))
    plt.savefig(join(sumdir, "models_pred_cmp_%s_%s.pdf"%(explab1, explab2)))
    plt.show()
    return figh
#%%
explab1 = "vgg16-conv4_3"; explab2 = "alexnet-conv3"#"vgg16-conv3_3"
tab1 = pd.read_csv(join(sumdir, "Both_pred_stats_vgg16-conv4_3_none_bdr1_NF3.csv"))
# tab1 = pd.read_csv(join(sumdir, "Both_pred_stats_alexnet-conv2_none_bdr1_NF3.csv"))
tab2 = pd.read_csv(join(sumdir, 'Both_pred_stats_alexnet-conv3_none_bdr1_NF3.csv'))
pred_cmp_scatter(tab1, tab2, "alexnet-conv2", "alexnet-conv3")
#%%
tab1 = pd.read_csv(join(sumdir, "Both_pred_stats_alexnet-conv3_pos_bdr1_NF3.csv"))
tab2 = pd.read_csv(join(sumdir, 'Both_pred_stats_alexnet-conv3_none_bdr1_NF3.csv'))
pred_cmp_scatter(tab1, tab2, "alexnet-conv3_pos", "alexnet-conv3_none")
#%%

lab1 = list(candidate_dict)[10]
lab2 = list(candidate_dict)[12]
tab1 = pd.read_csv(candidate_dict[lab1])
tab2 = pd.read_csv(candidate_dict[lab2])
pred_cmp_scatter(tab1, tab2, lab1, lab2, varnm="cc_aft_manif", colorvar="area", stylevar="Animal")
#%%
import re
from shutil import copyfile

fdrpatt = re.compile("(.*)-(.*)_NF(\d*)_bdr(\d*)_(.*)__nobdr")
model_param_map = {lab:fdrpatt.findall(lab)[0] for lab in candidate_dict.keys()}
model_param_map[np.nan] = (np.nan, np.nan, np.nan, np.nan, np.nan)

sumtab = pd.DataFrame() # temporary table to compare scores for different models
varnm = "cc_bef_norm_manif"  # "cc_aft_norm_manif"
for lab, tab_fn in candidate_dict.items():
    tab = pd.read_csv(tab_fn)
    sumtab[lab] = tab[varnm]

max_cc = sumtab.max(axis=1)
best_model_names = sumtab.idxmax(axis=1)
best_model_param = pd.DataFrame([tuple(model_param_map[lab]) for lab in best_model_names.to_list()],
    columns=["netname", "layer", "NF", "bdr", "rect_method"])

model_summary = pd.DataFrame()
model_summary["Animal"] = tab.Animal
model_summary["Expi"] = tab.Expi
model_summary["pref_chan"] = tab.pref_chan
model_summary["area"] = tab.area
model_summary["imgsize"] = tab.imgsize
model_summary["best_cc"] = max_cc
model_summary = pd.concat((model_summary, best_model_param), axis=1)
model_summary["model_str"] = best_model_names
model_summary.to_csv(join(sumdir, "model_synops_%s.csv"%varnm))
#%
best_model_dir = join(exproot, "best_models")
for rowi, R in model_summary.iterrows():
    if pd.isna(R.model_str):
        continue
    copyfile(join(exproot, R.model_str, "%s_Exp%02d_summary.png"%(R.Animal, R.Expi)), \
             join(best_model_dir, "%s_Exp%02d_summary.png"%(R.Animal, R.Expi)))
#%% Area charateristic
for name in model_summary.netname.unique():
    msk = (model_summary.netname == name)
    print("%s %d session" % (name, msk.sum()))
    print(model_summary.loc[msk].area.value_counts())
    # print(model_summary.loc[msk].imgsize.value_counts())

#%% See the effect of Evolution success.
evolsucstab = pd.read_csv(join(matdir, "Both_EvolTrajStats.csv"))
lab1 = list(candidate_dict)[4]
lab2 = list(candidate_dict)[1]
tab1 = pd.read_csv(candidate_dict[lab1])
tab2 = pd.read_csv(candidate_dict[lab2])
evolsucstab["isSuccess"] = (evolsucstab.t_p_initmax<1E-3)
pred_cmp_scatter(tab1, tab2, lab1, lab2, varnm="cc_aft_manif", colorvar="isSuccess", stylevar="area",
                 masktab=evolsucstab)
#%%
from scipy.stats import pearsonr, spearmanr
areanummap = lambda A: {"V1": 1, "V4": 2, "IT": 3}[A]
def testProgression(tab, varnm, msk=None):
    validmsk = ~((tab.Animal == "Alfa") & (tab.Expi == 10)) & (~tab[varnm].isna())
    if msk is not None:
        validmsk = validmsk & msk
    cval, pval = spearmanr(tab[varnm][validmsk], tab["area"].apply(areanummap)[validmsk])
    return cval, pval
#%% Prediction as a function of factor numbers
nf_csv_list = {
    "NF1": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF1_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF1.csv',
    "NF2": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF2_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF2.csv',
    "NF3": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF3.csv',
    "NF5": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF5_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF5.csv',
    #"resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust":
    # 'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr0_full.csv',
}
# plot prediction as a function of factor numbers
summarytab = pd.DataFrame()
for colnm in ["Animal", "Expi", "area", "pref_chan"]:
    summarytab[colnm] = tab[colnm]

for lab, tabfn in nf_csv_list.items():
    tab = pd.read_csv(tabfn)
    summarytab[lab] = tab["cc_bef_all"]
#%%
sigma = 0.1
xjit = np.random.randn(summarytab.shape[0]) * sigma
model_names = list(nf_csv_list)
plt.figure(figsize=[10,8])
for i, lab in enumerate(model_names):
    sns.scatterplot(x=i+xjit, y=summarytab[lab], hue=summarytab.area, alpha=0.7, style=summarytab.Animal)
plt.show()
#%%
summarytab = pd.DataFrame()
for lab, tabfn in nf_csv_list.items():
    tab = pd.read_csv(tabfn)
    summarytab[lab] = tab["cc_bef_norm_all"]
best_NFnum = summarytab.idxmax(axis=1).apply(lambda lab: int(lab[-1]))
# best_NFnum["area"] = tab.area
summarytab["best_NFnum"] = best_NFnum
for colnm in ["Animal", "Expi", "area", "pref_chan"]:
    summarytab[colnm] = tab[colnm]

summarytab.groupby("area").mean()
#%%
nf_csv_list = {
 "NF1":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF1_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF1.csv',
 "NF2":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF2_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF2.csv',
 "NF3":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF3.csv',
 "NF5":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF5_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF5.csv',
 # "Full":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust'
 #        '\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr0_full.csv',
}
label2num = {"NF1":1, "NF2":2, "NF3":3, "NF5":5, "Full":1024}
summarytab = pd.DataFrame()
for lab, tabfn in nf_csv_list.items():
    tab = pd.read_csv(tabfn)
    summarytab[lab] = tab["cc_bef_norm_manif"]

# best_NFnum = summarytab.idxmax(axis=1).apply(lambda lab: label2num[lab])
# # best_NFnum["area"] = tab.area
# summarytab["best_NFnum"] = best_NFnum
# for colnm in ["Animal", "Expi", "area", "pref_chan"]:
#     summarytab[colnm] = tab[colnm]
#
# summarytab.groupby("area").mean()
#%%
# summarytab.to_xarray()/np.abs(summarytab).max(axis=1)
summarymat = np.array(summarytab)
summarymat_norm = (summarymat / np.abs(summarymat).max(axis=1, keepdims=True))

figh, ax = plt.subplots(1, 3, figsize=[7, 6])
msks = [tab.area=="V1", tab.area=="V4", tab.area=="IT"]
clrs = ["red", "green", "blue"]
for i, (clr, msk) in enumerate(zip(clrs, msks)):
    xjit = np.random.randn(1,sum(msk))*0.1
    ax[i].plot(np.array([[1,2,3,5]]).T+xjit,summarymat_norm[msk,:].T,alpha=0.5,color=clr,)
plt.show()


#%%
validmsk = ~((tab.Animal == "Alfa") & (tab.Expi == 10))
#%%
plt.figure(figsize=[4,5])
sns.violinplot(x="area", y="cc_bef_norm_evoref", data=tab, inner="box", cut=0.1, bw=0.5)#saturation=0.7
plt.show()
#%%
targspace = "gabor"
varnm = "cc_bef_norm_gabor"
plt.figure(figsize=[4, 5])
sns.violinplot(x="area", y=varnm, data=tab[validmsk], order=["V1", "V4", "IT"],\
               inner="points", cut=0.1, bw=0.5)
plt.hlines(0, *plt.xlim(), linestyles='-.', color="red")
cval, pval = testProgression(tab, varnm, )
statstr = "Spearman r=%.3f (%.1e)"%(cval, pval)
plt.title("Model Prediction for %s Space\n"%targspace+statstr)
plt.show()
#%%
plt.figure()
varnm = "cc_bef_norm_gabor"
sns.violinplot(x="Animal", y=varnm, data=tab[validmsk], order=["Alfa","Beto"],\
               inner="points", cut=0.1, bw=0.5)
plt.show()
#%%
plt.figure()
sns.violinplot(x="area", y="cc_bef_norm_pasu", data=tab, inner="points")
plt.show()
#%%
nf_csv_list = \
{"NF1":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF1_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF1.csv',
 "NF2":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF2_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF2.csv',
 "NF3":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF3_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF3.csv',
 "NF5":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF5_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF5.csv',}
summarytab = pd.DataFrame()
for lab, tabfn in nf_csv_list.items():
    tab = pd.read_csv(tabfn)
    summarytab[lab] = tab["cc_bef_norm_manif"]
summarymat = np.array(summarytab)
summarymat_norm = (summarymat / np.abs(summarymat).max(axis=1, keepdims=True))

figh, ax = plt.subplots(1, 3, figsize=[7, 6])
msks = [tab.area=="V1", tab.area=="V4", tab.area=="IT"]
clrs = ["red", "green", "blue"]
for i, (clr, msk) in enumerate(zip(clrs, msks)):
    xjit = np.random.randn(1,sum(msk))*0.1
    ax[i].plot(np.array([[1,2,3,5]]).T+xjit,summarymat_norm[msk,:].T,alpha=0.5,color=clr,)
plt.show()

best_NFnum = summarytab.idxmax(axis=1).apply(lambda lab: int(lab[-1]))
# best_NFnum["area"] = tab.area
summarytab["best_NFnum"] = best_NFnum
for colnm in ["Animal", "Expi", "area", "pref_chan"]:
    summarytab[colnm] = tab[colnm]
summarytab.groupby("area").mean()