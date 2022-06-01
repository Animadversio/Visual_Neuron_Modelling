"""Post hoc analysis and summary for the CorrFeatTsr analysis"""
from featvis_lib import load_featnet, rectify_tsr, tsr_factorize, tsr_posneg_factorize, vis_feattsr, vis_featvec, \
    vis_feattsr_factor, vis_featvec_point, vis_featvec_wmaps, \
    CorrFeatScore, preprocess, show_img, pad_factor_prod
from CorrFeatTsr_predict_lib import fitnl_predscore, loadimg_preprocess, score_images
import os
from os.path import join
from glob import glob
import re
from shutil import copyfile
import numpy as np
import numpy.ma as ma
import pickle as pkl
from easydict import EasyDict
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, pearsonr
from scipy.stats import f_oneway
from scipy.stats import pearsonr, spearmanr
import torch
import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt
from data_loader import mat_path, loadmat, load_score_mat
from GAN_utils import upconvGAN
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
#%%
figroot = "O:\corrFeatTsr_FactorVis"
sumdir = join(figroot, "summary")
exproot = join(figroot, "models")
matdir = r"O:\Mat_Statistics"
csv_list = glob(join(exproot,"*","*.csv",), recursive=False)
# os.listdir(sumdir)
#%% Analyze and summarize statistics

# tab = pd.read_csv(join(sumdir, 'Both_pred_stats_vgg16-conv3_3_bdr3_NF3.csv'))
# summarize_tab(tab)
#%% Handy statistical functions for plotting
areanummap = lambda A: {"V1": 1, "V4": 2, "IT": 3}[A]
def testProgression(tab, varnm, msk=None):
    validmsk = ~((tab.Animal == "Alfa") & (tab.Expi == 10)) & (~tab[varnm].isna())
    if msk is not None:
        validmsk = validmsk & msk
    cval, pval = spearmanr(tab[varnm][validmsk], tab["area"].apply(areanummap)[validmsk])
    return cval, pval

def area_cmp_plot(tab, varnm, targspace="all", tablab="", msk=None, inner="points", figdir=""):
    if msk is None:
        msk = ~((tab.Animal == "Alfa") & (tab.Expi == 10))
    plt.figure(figsize=[4, 5])
    sns.violinplot(x="area", y=varnm, data=tab[msk], order=["V1", "V4", "IT"],\
                   inner=inner, cut=0.1, bw=0.5)
    plt.hlines(0, *plt.xlim(), linestyles='-.', color="red")
    cval, pval = testProgression(tab, varnm, )
    statstr = "Spearman r=%.3f (%.1e)"%(cval, pval)
    plt.title("Model Prediction for %s Space\n"%targspace+statstr)
    plt.savefig(join(figdir,"%s_model_%s_area_cmp.png"%(varnm, tablab)))
    plt.savefig(join(figdir,"%s_model_%s_area_cmp.png"%(varnm, tablab)))
    plt.show()

def pred_cmp_scatter(tab1, tab2, explab1, explab2, varnm="cc_bef", colorvar="area", stylevar="Animal", masktab=None, mask=None):
    """Compare prediction score for 2 models
    Plotting the scatter of prediction accuracy, separated by area and animal.
    """
    if masktab is None:
        masktab = tab1
    if mask is None:
        mask = np.ones(tab1.shape[0], dtype=bool)
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
    plt.title("Linear model prediction %s comparison\ncc %.3f t test(Fisher z) %.2f (%.1e)"%(varnm, cc, tval, pval))
    plt.savefig(join(sumdir, "models_pred_cmp_%s_%s_%s.png"%(varnm, explab1, explab2)))
    plt.savefig(join(sumdir, "models_pred_cmp_%s_%s_%s.pdf"%(varnm, explab1, explab2)))
    plt.show()
    return figh


def pred_stripe(tab1, explab1, varnm="cc_bef", columnvar="area", colorvar="Animal", masktab=None, kind="strip",
                alpha=0.6, mask=None):
    """Compare prediction score for 2 models
    Plotting the scatter of prediction accuracy, separated by area and animal.
    """
    if masktab is None:
        masktab = tab1
    if mask is None:
        mask = np.ones(tab1.shape[0], dtype=bool)

    masktab["area"] = ""
    masktab["area"][masktab.pref_chan <= 32] = "IT"
    masktab["area"][(masktab.pref_chan <= 48) & (masktab.pref_chan >= 33)] = "V1"
    masktab["area"][masktab.pref_chan >= 49] = "V4"
    # figh, ax = plt.subplots(figsize=(4,6.5))
    # sns.catplot(x=masktab[columnvar], y=tab1[varnm], hue=masktab[colorvar], order=["V1","V4","IT"],
    #               ax=ax, kind=kind)#style=masktab[
    sns.catplot(x=columnvar, y=varnm, hue=colorvar, data=tab1, order=["V1","V4","IT"],
                  kind=kind, height=7.5, aspect=2/3, alpha=alpha)#style=masktab[
    # stylevar])
    plt.ylabel(varnm);plt.xlabel("area")
    plt.gca().set_aspect('equal', adjustable='box')  # datalim
    # cc = ma.corrcoef(ma.masked_invalid(tab1[varnm]), ma.masked_invalid(tab2[varnm]))[0, 1]
    # cc = np.corrcoef(tab1[varnm], tab2[varnm])
    # tval, pval = ttest_rel(np.arctanh(tab1[varnm]), np.arctanh(tab2[varnm]), nan_policy='omit') # ttest: exp1 - exp2
    # plt.title("%s\nModel prediction comparison\ncc %.3f t test(Fisher z) %.2f (%.1e)"%(explab1, cc, tval, pval))
    plt.title("%s\nModel prediction comparison"%(explab1))
    plt.savefig(join(sumdir, "models_pred_%s_%s.png"%(explab1, varnm)))
    plt.savefig(join(sumdir, "models_pred_%s_%s.pdf"%(explab1, varnm)))
    figh = plt.gcf()
    plt.show()
    return figh


def pred_perform_cmp(nf_csv_dict, label2num, statname, modelstr="net-layer", figdir=sumdir, expmsk=None):
    """Well formed plot function for model performance as a function of factor number
    Separated by area and animal
    Report the mean, sem of factor number per area
    Report the progression statistics
    """
    sumtab = pd.DataFrame()
    xticks = []
    for lab, tabfn in nf_csv_dict.items():
        tab = pd.read_csv(tabfn)
        sumtab[lab] = tab[statname]
        xticks.append(label2num[lab])
    if expmsk is None:
        expmsk = np.ones(tab.shape[0], dtype=bool)
    xticks = np.array(xticks).reshape([-1,1])
    summarymat = np.array(sumtab)
    summarymat_norm = (summarymat / np.abs(summarymat).max(axis=1, keepdims=True))
    # if expmsk is not None:
    sumtab = sumtab[expmsk]
    summarymat = summarymat[expmsk]
    summarymat_norm = summarymat_norm[expmsk, :]

    figh, ax = plt.subplots(1, 3, figsize=[9, 6])
    area_str = ["V1","V4","IT"]
    msks = [tab[expmsk].area=="V1", tab[expmsk].area=="V4", tab[expmsk].area=="IT"]
    minormsks = [tab[expmsk].Animal=="Alfa", tab[expmsk].Animal=="Beto"]
    clrs = ["red", "green", "blue"]
    styles = ["-", "-."]
    for i, (clr, msk) in enumerate(zip(clrs, msks)):
        for mi, mmsk in enumerate(minormsks):
            xjit = np.random.randn(1,sum(msk&mmsk))*0.1
            ax[i].plot(xticks+xjit, summarymat_norm[msk&mmsk,:].T,
                       alpha=0.5, color=clr, linestyle=styles[mi])
        ax[i].set_title(area_str[i])
        ax[i].set_xlabel("Factor Number")
        ax[i].set_xticks(xticks[:,-1])
    ax[0].set_ylabel(statname+" norm to max")

    best_NFnum = sumtab.idxmax(axis=1).apply(lambda lab: label2num[lab])
    # best_NFnum["area"] = tab.area
    sumtab["best_NFnum"] = best_NFnum
    for colnm in ["Animal", "Expi", "area", "pref_chan"]:
        sumtab[colnm] = tab[expmsk][colnm]
    # Quick way to get summary statistics for each group
    summary = sumtab.groupby("area", sort=False).mean()
    summary_sem = sumtab.groupby("area", sort=False).sem()
    summary_cnt = sumtab.groupby("area", sort=False).size()
    print("Summarize statistics %s vs Number of factors"%statname)
    print(summary)
    # summary2 = sumtab.groupby(["Animal", "area"], sort=False).mean()
    # print(summary2)
    valmsk = ~sumtab.best_NFnum.isna()
    Fval, F_pval = f_oneway(sumtab.best_NFnum[(sumtab.area=="V1")&valmsk], sumtab.best_NFnum[(sumtab.area=="V4")&valmsk], \
                    sumtab.best_NFnum[(sumtab.area=="IT")&valmsk])
    area_num = sumtab.area.apply(areanummap)
    rval, rpval = spearmanr(area_num[valmsk], best_NFnum[valmsk])
    print("Best NF number ~ area, ANOVA F %.2f (p=%.1e)"%(Fval, F_pval))
    print("Best NF number ~ area, Spearman R %.3f (p=%.1e)"%(rval, rpval))
    figh.suptitle("%s model prediction as function of NF\nbest factor N ~ area: ANOVA F %.2f (p=%.1e) Spearman R %.3f ("
                  "p=%.1e)" % (modelstr, Fval, F_pval, rval, rpval))
    for i, area in enumerate(area_str):
        ax[i].axvline(summary.best_NFnum[area], C='k', ls=":")
        ax[i].text(summary.best_NFnum[area], 1.0, '%.2f+-%.2f (N=%d)'%(summary.best_NFnum[area], summary_sem.best_NFnum[area], summary_cnt[area]), rotation=0)
        # ax[i].vlines(sumtab.best_NFnum[msks[i]].to_list(), 0, 1, color='k', ls="-.", alpha=0.1)
    plt.tight_layout()
    figh.savefig(join(figdir, "NF-modelAccur_area_sep_%s_%s.png"%(modelstr, statname)),
                bbox_inches='tight')
    figh.savefig(join(figdir, "NF-modelAccur_area_sep_%s_%s.pdf"%(modelstr, statname)),
                bbox_inches='tight')
    plt.show()
    return sumtab, summary, figh

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
pred_cmp_scatter(tab1, tab2, lab1, lab2, varnm="cc_bef_norm_manif", colorvar="area", stylevar="Animal")


#%% Find best model
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

#%% Prediction as a function of factor numbers
#%%
label2num = {"NF1":1, "NF2":2, "NF3":3, "NF5":5, "NF7":7, "NF9":9, "Full":1024, np.nan:np.nan}
nf_csv_list = \
{'NF1': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF1_bdr1_Tthresh_3__nobdr\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF1.csv',
 'NF2': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF2_bdr1_Tthresh_3__nobdr\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF2.csv',
 'NF3': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF3_bdr1_Tthresh_3__nobdr\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF3.csv',
 'NF5': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF5_bdr1_Tthresh_3__nobdr\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF5.csv',
 'NF7': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF7_bdr1_Tthresh_3__nobdr\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF7.csv',}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_all", modelstr="VGG16-conv4_3")
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_all", modelstr="VGG16-conv4_3")
#%
nf_csv_list = \
{"NF1":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF1_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF1.csv',
 "NF2":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF2_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF2.csv',
 "NF3":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF3_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF3.csv',
 "NF5":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF5_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF5.csv',
 "NF7":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF7_bdr1_Tthresh_3__nobdr_alex\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF7.csv', }
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_all", modelstr="alexnet-conv4")
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_all", modelstr="alexnet-conv4")
#%
nf_csv_list = {
 "NF1":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF1_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF1.csv',
 "NF2":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF2_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF2.csv',
 "NF3":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF3.csv',
 "NF5":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF5_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF5.csv',
 "NF7":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF7_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF7.csv',
 # "Full":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust'
 #        '\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr0_full.csv',
}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_all", modelstr="resnet50_robust-layer3")
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_all", modelstr="resnet50_robust-layer3")
#%
nf_csv_list = {
    "NF1": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF1_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF1.csv',
    "NF2": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF2_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF2.csv',
    "NF3": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF3.csv',
    "NF5": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF5_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF5.csv',
    "NF7": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF7_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF7.csv',
}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_all", modelstr="resnet50-layer3")
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_all", modelstr="resnet50-layer3")


#%% Cross Validated version of prediction
label2num = {"NF1":1, "NF2":2, "NF3":3, "NF5":5, "NF7":7,  "NF9":9, "Full":1024, np.nan:np.nan}
valmsk = ~((exptab1.Animal=="Alfa") * (exptab1.Expi==10))
nf_csv_list = {
 "NF1":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF1_bdr1_Tthresh_3__nobdr_res-robust_CV\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF1_CV.csv',
 "NF2":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF2_bdr1_Tthresh_3__nobdr_res-robust_CV\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF2_CV.csv',
 "NF3":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust_CV\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF3_CV.csv',
 "NF5":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF5_bdr1_Tthresh_3__nobdr_res-robust_CV\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF5_CV.csv',
 "NF7":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF7_bdr1_Tthresh_3__nobdr_res-robust_CV\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF7_CV.csv',
 "NF9":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF9_bdr1_Tthresh_3__nobdr_res-robust_CV\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF9_CV.csv',
 # "Full":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust'
 #        '\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr0_full.csv',
}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_manif", modelstr="resnet50_robust-layer3_CV", expmsk=valmsk)
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_manif", modelstr="resnet50_robust-layer3_CV", expmsk=valmsk)
#%% VGG16
# sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_manif", modelstr="resnet50_robust-layer3_CV")
# sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_manif", modelstr="resnet50_robust-layer3_CV")
#%
nf_csv_list = {
 'NF1': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF1_bdr1_Tthresh_3__nobdr_CV\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF1_CV.csv',
 'NF2': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF2_bdr1_Tthresh_3__nobdr_CV\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF2_CV.csv',
 'NF3': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF3_bdr1_Tthresh_3__nobdr_CV\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF3_CV.csv',
 'NF5': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF5_bdr1_Tthresh_3__nobdr_CV\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF5_CV.csv',
 'NF7': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF7_bdr1_Tthresh_3__nobdr_CV\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF7_CV.csv',
 'NF9': 'O:\\corrFeatTsr_FactorVis\\models\\vgg16-conv4_3_NF9_bdr1_Tthresh_3__nobdr_CV\\Both_pred_stats_vgg16-conv4_3_Tthresh_bdr1_NF9_CV.csv',
}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_manif", modelstr="VGG16-conv4_3_CV", expmsk=valmsk)
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_manif", modelstr="VGG16-conv4_3_CV", expmsk=valmsk)
#% Alexnet
nf_csv_list = {
"NF1":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF1_bdr1_Tthresh_3__nobdr_alex_CV\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF1_CV.csv',
"NF2":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF2_bdr1_Tthresh_3__nobdr_alex_CV\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF2_CV.csv',
"NF3":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF3_bdr1_Tthresh_3__nobdr_alex_CV\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF3_CV.csv',
"NF5":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF5_bdr1_Tthresh_3__nobdr_alex_CV\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF5_CV.csv',
"NF7":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF7_bdr1_Tthresh_3__nobdr_alex_CV\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF7_CV.csv',
"NF9":'O:\\corrFeatTsr_FactorVis\\models\\alexnet-conv4_NF9_bdr1_Tthresh_3__nobdr_alex_CV\\Both_pred_stats_alexnet-conv4_Tthresh_bdr1_NF9_CV.csv',
}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_manif", modelstr="alexnet-conv4_CV",
                                      expmsk=valmsk)
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_manif", modelstr="alexnet-conv4_CV",
                                      expmsk=valmsk)
#% Resnet50
nf_csv_list = {
"NF1": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF1_bdr1_Tthresh_3__nobdr_resnet_CV\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF1_CV.csv',
"NF2": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF2_bdr1_Tthresh_3__nobdr_resnet_CV\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF2_CV.csv',
"NF3": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet_CV\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF3_CV.csv',
"NF5": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF5_bdr1_Tthresh_3__nobdr_resnet_CV\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF5_CV.csv',
"NF7": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF7_bdr1_Tthresh_3__nobdr_resnet_CV\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF7_CV.csv',
"NF9": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF9_bdr1_Tthresh_3__nobdr_resnet_CV\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF9_CV.csv',
}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_manif", modelstr="resnet50-layer3_CV",
                                      expmsk=valmsk)
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_manif", modelstr="resnet50-layer3_CV",
                                      expmsk=valmsk)

#%%










#%%
nf_csv_list = {
 'NF1': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer2_NF1_bdr3_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer2_Tthresh_bdr3_NF1.csv',
 'NF2': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer2_NF2_bdr3_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer2_Tthresh_bdr3_NF2.csv',
 'NF3': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer2_NF3_bdr3_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer2_Tthresh_bdr3_NF3.csv',
 'NF5': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer2_NF5_bdr3_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer2_Tthresh_bdr3_NF5.csv',
 'NF7': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer2_NF7_bdr3_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer2_Tthresh_bdr3_NF7.csv',
 'NF9': 'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer2_NF9_bdr3_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer2_Tthresh_bdr3_NF9.csv',
}
label2num = {"NF1":1, "NF2":2, "NF3":3, "NF5":5, "NF7":7, "NF9":9, "Full":1024, np.nan:np.nan}
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_bef_norm_all", modelstr="resnet50_robust-layer2")
sumtab, summary, _ = pred_perform_cmp(nf_csv_list, label2num, "cc_aft_norm_all", modelstr="resnet50_robust-layer2")
#%% Compare performance of Full and NF3 model
tab1 = pd.read_csv(r'O:\corrFeatTsr_FactorVis\models\resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust' \
 '\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr0_full.csv')
explab1 = "resnet50-robust-layer3_full"
tab2 = pd.read_csv('O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust' \
 '\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF3.csv')
explab2 = "resnet50_linf8-layer3_NF3"
pred_cmp_scatter(tab1, tab2, explab1, explab2, varnm="cc_bef_norm_all", colorvar="area", stylevar="Animal")
pred_cmp_scatter(tab1, tab2, explab1, explab2, varnm="cc_bef_norm_manif", colorvar="area", stylevar="Animal")
pred_cmp_scatter(tab1, tab2, explab1, explab2, varnm="cc_aft_norm_all", colorvar="area", stylevar="Animal")
pred_cmp_scatter(tab1, tab2, explab1, explab2, varnm="cc_bef_all", colorvar="area", stylevar="Animal")
pred_cmp_scatter(tab1, tab2, explab1, explab2, varnm="cc_aft_all", colorvar="area", stylevar="Animal")
#%% Plot scatter for statistics of performance for Full model
tab = pd.read_csv(r'O:\corrFeatTsr_FactorVis\models\resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust' \
                '\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr0_full.csv')
explab = "resnet50_linf8-layer3_Tthresh_bdr0_full"
pred_stripe(tab, explab, varnm="cc_bef_all", kind="swarm")
pred_stripe(tab, explab, varnm="cc_bef_norm_all", kind="swarm")
pred_stripe(tab, explab, varnm="cc_aft_norm_all", kind="swarm")
pred_stripe(tab, explab, varnm="cc_bef_manif", kind="swarm")
pred_stripe(tab, explab, varnm="cc_bef_norm_manif", kind="swarm")
pred_stripe(tab, explab, varnm="cc_aft_norm_manif", kind="swarm")
#%% Plot scatter for statistics of performance for NF3 model
tab = pd.read_csv('O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust' \
 '\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF3.csv')
explab = "resnet50_linf8-layer3_Tthresh_bdr1_NF3"
pred_stripe(tab, explab, varnm="cc_bef_all", kind="swarm")
pred_stripe(tab, explab, varnm="cc_bef_norm_all", kind="swarm")
pred_stripe(tab, explab, varnm="cc_aft_norm_all", kind="swarm")
pred_stripe(tab, explab, varnm="cc_bef_manif", kind="swarm")
pred_stripe(tab, explab, varnm="cc_bef_norm_manif", kind="swarm")
pred_stripe(tab, explab, varnm="cc_aft_norm_manif", kind="swarm")



#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% Obsolete development zone
#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nf_csv_list = {
    "NF1": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF1_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF1.csv',
    "NF2": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF2_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF2.csv',
    "NF3": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF3_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF3.csv',
    "NF5": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF5_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF5.csv',
    "NF7": 'O:\\corrFeatTsr_FactorVis\\models\\resnet50-layer3_NF7_bdr1_Tthresh_3__nobdr_resnet\\Both_pred_stats_resnet50-layer3_Tthresh_bdr1_NF7.csv',
}
label2num = {"NF1":1, "NF2":2, "NF3":3, "NF5":5, "NF7":7, "Full":1024}
# plot prediction as a function of factor numbers
#%%
statname = "cc_bef_norm_all"
summarytab = pd.DataFrame()
for lab, tabfn in nf_csv_list.items():
    tab = pd.read_csv(tabfn)
    summarytab[lab] = tab[statname]
    area_cmp_plot(tab, statname, "all", tablab=lab, figdir=sumdir, inner="box")
    print(tab.groupby("area").mean()["cc_bef_all"])

summarymat = np.array(summarytab)
summarymat_norm = (summarymat / np.abs(summarymat).max(axis=1, keepdims=True))
figh, ax = plt.subplots(1, 3, figsize=[7, 6])
labels = ["V1", "V4", "IT"]
msks = [tab.area=="V1", tab.area=="V4", tab.area=="IT"]
clrs = ["red", "green", "blue"]
for i, (clr, msk) in enumerate(zip(clrs, msks)):
    xjit = np.random.randn(1,sum(msk))*0.1
    ax[i].plot(np.array([[1,2,3,5,7]]).T+xjit,summarymat_norm[msk,:].T,alpha=0.5,color=clr,)
    ax[i].set_title(labels[i])
plt.show()
best_NFnum = summarytab.idxmax(axis=1).apply(lambda lab: label2num[lab])
# best_NFnum["area"] = tab.area
summarytab["best_NFnum"] = best_NFnum
for colnm in ["Animal", "Expi", "area", "pref_chan"]:
    summarytab[colnm] = tab[colnm]
summarytab.groupby("area").mean()

#%%
sigma = 0.1
xjit = np.random.randn(summarytab.shape[0]) * sigma
model_names = list(nf_csv_list)
plt.figure(figsize=[10,8])
for i, lab in enumerate(model_names):
    sns.scatterplot(x=i+xjit, y=summarytab[lab], hue=summarytab.area, alpha=0.7, style=summarytab.Animal)
plt.show()
#%%
#%% ResNet50_linf robust factor number 
nf_csv_list = {
 "NF1":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF1_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF1.csv',
 "NF2":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF2_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF2.csv',
 "NF3":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF3_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF3.csv',
 "NF5":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF5_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF5.csv',
 "NF7":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_NF7_bdr1_Tthresh_3__nobdr_res-robust\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr1_NF7.csv',
    # "Full":'O:\\corrFeatTsr_FactorVis\\models\\resnet50_linf8-layer3_Full_bdr0_Tthresh_3__nobdr_res-robust'
 #        '\\Both_pred_stats_resnet50_linf8-layer3_Tthresh_bdr0_full.csv',
}
label2num = {"NF1":1, "NF2":2, "NF3":3, "NF5":5, "NF7":7, "Full":1024, np.nan: np.nan}
statname = "cc_bef_norm_all"
summarytab = pd.DataFrame()
for lab, tabfn in nf_csv_list.items():
    tab = pd.read_csv(tabfn)
    summarytab[lab] = tab[statname]
    # area_cmp_plot(tab, statname, "all", tablab=lab, figdir=sumdir, inner="box")
    print(tab.groupby("area").mean()[statname])
summarymat = np.array(summarytab)
summarymat_norm = (summarymat / np.abs(summarymat).max(axis=1, keepdims=True))
# %
figh, ax = plt.subplots(1, 3, figsize=[7, 6])
labels = ["V1", "V4", "IT"]
msks = [tab.area=="V1", tab.area=="V4", tab.area=="IT"]
clrs = ["red", "green", "blue"]
for i, (clr, msk) in enumerate(zip(clrs, msks)):
    xjit = np.random.randn(1,sum(msk))*0.1
    ax[i].plot(np.array([[1,2,3,5,7]]).T+xjit,summarymat_norm[msk,:].T,alpha=0.5,color=clr,)
    ax[i].set_title(labels[i])
plt.show()
best_NFnum = summarytab.idxmax(axis=1).apply(lambda lab: label2num[lab])
# best_NFnum["area"] = tab.area
summarytab["best_NFnum"] = best_NFnum
for colnm in ["Animal", "Expi", "area", "pref_chan"]:
    summarytab[colnm] = tab[colnm]
summarytab.groupby("area").mean()

#%%
#%%
plt.figure(figsize=[4,5])
sns.violinplot(x="area", y="cc_bef_norm_evoref", data=tab, inner="box", cut=0.1, bw=0.5)#saturation=0.7
plt.show()
#%%
validmsk = ~((tab.Animal == "Alfa") & (tab.Expi == 10))
targspace = "all"
varnm = "cc_bef_norm_all"
area_cmp_plot(tab, varnm, targspace, figdir=sumdir)
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
