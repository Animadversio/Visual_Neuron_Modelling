"""
Summarizing all penalized regression and factor regression models.
"""
import re
import os
from os.path import join
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
rootdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
sumdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress\summary"
figdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress\summary"
#%%
df_all = pd.read_csv(join(sumdir, "Combined_Penalize-FactorRegression_summary.csv"), index_col=0)
df_all_reset = df_all.reset_index(inplace=False)
df_all_reset["layer_s"] = df_all_reset.layer.apply(lambda s: s.split(".")[1])  # short name for layer
validmsk = ~((df_all_reset.Animal == "Alfa") & (df_all_reset.Expi == 10))
df_all_valid = df_all_reset[validmsk]
#%%
"""
Combine statistics for all exp all layers all methods (Penalized regression + FeatureCorrelation)
"""
exptype = "all"
df_all = pd.DataFrame()
for Animal in ["Alfa", "Beto"]:
    for Expi in tqdm(range(1, 47)):
        if Animal == "Beto" and Expi == 46: continue
        expdir = join(rootdir, f"{Animal}_{Expi:02d}")
        for exptype in ["Evol", "Manif", "Gabor", "Pasu", "EvolRef", "allnat", "all"]:
            for featlayer in [".layer2.Bottleneck3", ".layer3.Bottleneck5", ".layer4.Bottleneck2"]:
                # factor regression
                df = pd.read_csv(join(expdir, f"eval_predict_factreg-{featlayer}-{exptype}.csv"), index_col=(0,1))
                df["Animal"] = Animal
                df["Expi"] = Expi
                df["factorreg"] = True
                df_all = pd.concat([df_all, df], axis=0)
                # penalized regression
                df2 = pd.read_csv(join(expdir, f"eval_predict_{featlayer}-{exptype}.csv"), index_col=(0, 1))
                df2["Animal"] = Animal
                df2["Expi"] = Expi
                df2["factorreg"] = False
                df_all = pd.concat([df_all, df2], axis=0)
#%%
df_all = df_all.set_index("layer", append=True).swaplevel(0,2)
df_all = df_all.rename_axis(['layer', 'regressor', 'FeatRed'])
df_all.to_csv(join(sumdir, "Combined_Penalize-FactorRegression_summary.csv"))
df_all_reset = df_all.reset_index(inplace=False)
df_all_reset["layer_s"] = df_all_reset.layer.apply(lambda s: s.split(".")[1])
validmsk = ~((df_all_reset.Animal == "Alfa") & (df_all_reset.Expi == 10))
#%%
# Manif
df_all_valid[(df_all_valid.img_space == "all") & \
             (df_all_valid.FeatRed.str.contains("featvec3|spmask3|pca|srp"))].\
    groupby(["layer_s", "FeatRed", 'regressor'], sort=False).\
    rho_p.agg(["mean", "sem", "count"])
#%%
"""Plot the method comparison for prediction in different image spaces. """
imgspace = "allnat"
g = sns.FacetGrid(df_all_reset[validmsk & (df_all_reset.img_space == imgspace)],#Manif
                  col="layer_s", row="regressor", hue="factorreg",
                  height=5, aspect=0.75, ylim=(-0.05, 0.40),
                  row_order=["Ridge", "Lasso", ],)
g.map(sns.barplot, "FeatRed", "rho_p",
      order=["facttsr1", 'factor1', 'spmask1', 'featvec1',
             'facttsr3', 'factor3', 'spmask3', 'featvec3',
             'pca', 'srp', 'sp_avg'], )
g.set_xticklabels(rotation=30)
g.set_titles(size=13)
plt.suptitle(f"Compare All Regression methods (regressor, featred, layer) all Expi, image space {imgspace}", fontsize=15)
plt.tight_layout()
plt.legend()
g.savefig(join(figdir, f"overall_prediction_synopsis_{imgspace}_rho.png"))
plt.show()
#%%
imgspace = "Manif" # "all"
g = sns.FacetGrid(df_all_reset[validmsk & (df_all_reset.img_space == imgspace)],#Manif
                  col="layer_s", row="regressor", hue="factorreg",
                  height=5, aspect=0.6, ylim=(-0.05, 0.90),
                  row_order=["Ridge", "Lasso", ],)
g.map(sns.barplot, "FeatRed", "rho_p",
      order=["facttsr1", 'factor1', 'spmask1', 'featvec1',
             'facttsr3', 'factor3', 'spmask3', 'featvec3',
             'pca', 'srp', 'sp_avg'], )
g.set_xticklabels(rotation=30)
g.set_titles(size=13)
plt.suptitle(f"Compare All Regression methods (regressor, featred, layer) all Expi, image space: {imgspace}", fontsize=15)
plt.tight_layout()
plt.legend()
g.savefig(join(figdir, f"overall_prediction_synopsis_{imgspace}_rho.png"))
plt.show()
#%%
imgspace = "allnat"
g = sns.FacetGrid(df_all_reset[validmsk & (df_all_reset.img_space == imgspace)],#Manif
                  col="layer_s", row="regressor", hue="factorreg",
                  height=5, aspect=0.75, ylim=(-0.50, 0.30),
                  row_order=["Ridge", "Lasso", ],)
g.map(sns.barplot, "FeatRed", "D2",
      order=["facttsr1", 'factor1', 'spmask1', 'featvec1',
             'facttsr3', 'factor3', 'spmask3', 'featvec3',
             'pca', 'srp', 'sp_avg'], )
g.set_xticklabels(rotation=30)
g.set_titles(size=13)
plt.suptitle(f"Compare All Regression methods (regressor, featred, layer) all Expi, image space: {imgspace}", fontsize=15)
plt.tight_layout()
plt.legend()
g.savefig(join(figdir, f"overall_prediction_synopsis_{imgspace}_R2.png"))
plt.show()

#%%
imgspace = "Manif"
g = sns.FacetGrid(df_all_reset[validmsk & (df_all_reset.img_space == imgspace)],#Manif
                  col="layer_s", row="regressor", hue="factorreg",
                  height=5, aspect=0.75, ylim=(-0.20, 0.50),
                  row_order=["Ridge", "Lasso", ],)
g.map(sns.barplot, "FeatRed", "D2",
      order=["facttsr1", 'factor1', 'spmask1', 'featvec1',
             'facttsr3', 'factor3', 'spmask3', 'featvec3',
             'pca', 'srp', 'sp_avg'], )
g.set_xticklabels(rotation=30)
g.set_titles(size=13)
plt.suptitle(f"Compare All Regression methods (regressor, featred, layer) all Expi, image space: {imgspace}", fontsize=15)
plt.tight_layout()
plt.legend()
g.savefig(join(figdir, f"overall_prediction_synopsis_{imgspace}_R2.png"))
plt.show()
#%%
"""
Per experiment model comparison 
"""
import matplotlib
massproduce = True
matplotlib.use("Agg" if massproduce else 'module://backend_interagg') #
#%%
df_all = pd.read_csv(join(sumdir, "Combined_Penalize-FactorRegression_summary.csv"))
df_all_reset = df_all.reset_index(inplace=False)
df_all_reset["layer_s"] = df_all_reset.layer.apply(lambda s: s.split(".")[1])  # short name for layer
validmsk = ~((df_all_reset.Animal == "Alfa") & (df_all_reset.Expi == 10))
#%% [markdown]
"""
Visualize as barplot the prediction accuracy
for each layer and regressor individually for each experiment.
"""
rootdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
sumdir  = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress\summary"
figdir = join(sumdir, "per_experiment")  # per experiment summary figure
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        df_exp = df_all_reset[(df_all_reset.Animal == Animal) & (df_all_reset.Expi == Expi)]
        sns.set(font_scale=1.2)
        for featlayer in [".layer2.Bottleneck3", ".layer3.Bottleneck5", ".layer4.Bottleneck2"]:
            g = sns.FacetGrid(df_exp[df_exp.layer == featlayer], row="FeatRed", col="regressor",
                              height=5, aspect=1.05, ylim=(-0.2, 1.0),
                              row_order=["pca", "srp", "sp_avg", 'spmask3', 'featvec3'],)
            g.map(sns.barplot, "img_space", "rho_p",
                  order=["Evol", "Manif", "Gabor", "Pasu", "EvolRef", "all", "allnat"], )
            plt.suptitle(f"Compare Regression methods {Animal} {Expi:02d} {featlayer}")
            g.savefig(join(figdir, f"prediction_comparison_factor_{Animal}_{Expi:02d}_{featlayer}.png"))
            plt.show()
#%%
figdir = join(sumdir, "per_experiment")  # per experiment summary figure
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        df_exp = df_all_reset[(df_all_reset.Animal == Animal) & (df_all_reset.Expi == Expi)]
        sns.set(font_scale=1.0)
        for featlayer in [".layer2.Bottleneck3", ".layer3.Bottleneck5", ".layer4.Bottleneck2"]:
            g = sns.FacetGrid(df_exp[df_exp.layer == featlayer], col="img_space", row="regressor", hue="factorreg",
                              height=5, aspect=0.7, ylim=(-0.2, 1.0),
                              col_order=["Evol", "Manif", "all", "allnat", "Gabor", "Pasu", "EvolRef"],
                              row_order=["Ridge","Lasso"],)
            g.map(sns.barplot, "FeatRed", "rho_p",
                  order=['pca', 'srp', 'sp_avg', "spmask3", "featvec3", "factor3", "facttsr3", \
                         "spmask1", "featvec1", "factor1", "facttsr1"], )
            # g = sns.FacetGrid(df_exp[df_exp.layer == featlayer], col="FeatRed", row="regressor",
            #                   height=5, aspect=0.7, ylim=(-0.2, 1.0),
            #                   col_order=["spmask3", "featvec3", "factor3", "facttsr3", \
            #                              "spmask1", "featvec1", "factor1", "facttsr1"],
            #                   row_order=["Ridge","Lasso"],)
            # g.map(sns.barplot, "img_space", "rho_p",
            #       order=["Evol", "Manif", "all", "allnat", "Gabor", "Pasu", "EvolRef"], )
            # g = sns.FacetGrid(df_exp[df_exp.layer == featlayer], col="img_space", row="regressor",
            #                   height=5, aspect=0.7, ylim=(-0.2, 1.0),
            #                   col_order=["Evol", "Manif", "all", "allnat", "Gabor", "Pasu", "EvolRef", ],
            #                   row_order=["Ridge", "Lasso"], )
            # g.map(sns.barplot, "FeatRed", "rho_p",
            #       order=["pca", "srp", "sp_avg", 'spmask3', 'featvec3'], )
            plt.suptitle(f"Compare Regression methods {Animal} Exp {Expi:02d} {featlayer}")
            g.set_xticklabels(rotation=30)
            plt.tight_layout()
            g.savefig(join(figdir, f"prediction_comparison_factor_Trsp_{Animal}_{Expi:02d}_{featlayer}.png"))
            plt.show()

#%% Compare across the area / pref chan
from scipy.io import loadmat
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
"""
Record the pref chan and area for each experiment
"""
def chan2area(chan):
    if chan <= 32 and chan >= 1:
        return "IT"
    elif chan >= 49 and chan <=64:
        return "V4"
    elif chan > 32 and chan < 49:
        return "V1"


df_all_reset["prefchan"] = 0
for Animal in ["Alfa", "Beto"]:
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    for Expi in range(1, len(EStats) + 1):
        if Animal == "Beto" and Expi == 46: continue
        expmask = (df_all_reset.Animal == Animal) & (df_all_reset.Expi == Expi)
        df_all_reset["prefchan"][expmask] = EStats[Expi - 1].evol.pref_chan

df_all_reset["area"] = df_all_reset.prefchan.apply(chan2area)
#%%
df_sum = df_all_reset[(df_all_reset.img_space == "all") &
                     (df_all_reset.FeatRed == "pca") &
                     (df_all_reset.regressor == "Ridge")].\
                     groupby(["Animal", "Expi", "layer_s"]).mean()
df_sum.area = df_sum.prefchan.apply(chan2area)
#%%
"""Prediction accuracy as function of layer; across area`"""
plt.figure(figsize=(8.5, 5.5))
ax1 = plt.subplot(1, 3, 1)
df_sum[df_sum.area == "V1"].unstack(level=(0, 1)).\
    plot(y="rho_p", alpha=0.6, legend=False, ax=ax1)
ax1.set_title("V1")
ax2 = plt.subplot(1, 3, 2)
df_sum[df_sum.area == "V4"].unstack(level=(0, 1)).\
    plot(y="rho_p", alpha=0.6, legend=False, ax=ax2)
ax2.set_title("V4")
ax3 = plt.subplot(1, 3, 3)
df_sum[df_sum.area == "IT"].unstack(level=(0, 1)).\
    plot(y="rho_p", alpha=0.6, legend=False, ax=ax3)
ax3.set_title("IT")
plt.show()

#%%
""" Compute The feature layer of best predicting method per experiment """
df_sum = df_all_reset[(df_all_reset.img_space == "allnat") &
             (df_all_reset.FeatRed == "spmask3") &
             (df_all_reset.regressor == "Ridge")].\
             groupby(["Animal", "Expi", "layer_s"]).mean()
df_sum.area = df_sum.prefchan.apply(chan2area)
performtab = df_sum["rho_p"].unstack(level=(0,1)).T
maxlayer = performtab.idxmax(axis=1)
maxidx = maxlayer.apply(lambda s: int(s[-1]))  # numerical index of the layer
area_col = df_sum.area.unstack((0, 1)).T["layer2"]
maxidxtab = pd.concat((maxidx, area_col), axis=1)
maxidxtab.groupby("layer2").agg(["mean", 'sem', 'count'])

#%%
"""
Plot the prototype of the best performing regression model in montage
"""
from build_montages import make_grid_np, crop_from_montage
def row2filename(row):
    return f"{row.Animal}-Exp{row.Expi:02d}-{row.layer}-{row.FeatRed}-{row.regressor}_vis.png"

outdir = join(rootdir, "summary\per_exp_best")
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        featvis_dir = join(rootdir, f"{Animal}_{Expi:02d}", "featvis")
        df_exp = df_all_reset[(df_all_reset.Animal == Animal) & (df_all_reset.Expi == Expi)]
        df_exp_rank = df_exp[df_exp.img_space == "all"].sort_values(by=["rho_p"], ascending=False)\
            .head(10)  # [["layer_s", "factorreg", "FeatRed", "regressor", "rho_p", "D2"]]
        proto_col = []
        for idx, row in df_exp_rank.iterrows():
            # assert os.path.exists(join(featvis_dir, row2filename(row)))
            mtg = plt.imread(join(featvis_dir, row2filename(row)))
            proto_first = crop_from_montage(mtg, (0, 0))
            proto_col.append(proto_first)
        method_mtg = make_grid_np(proto_col, nrow=5, rowfirst=True)
        df_exp_excerpt = df_exp_rank[["layer_s", "factorreg", "FeatRed", "regressor", "rho_p", "D2"]]
        df_exp_excerpt.to_csv(join(outdir, f"{Animal}_{Expi:02d}_best_methods.csv"))
        plt.imsave(join(outdir, f"{Animal}_{Expi:02d}_best_methods_proto.png"), method_mtg)
        # raise Exception("")





#%% Scratch zone
#%% Collect results of each exp into a dataframe
exptype = "all"
df_all = pd.DataFrame()
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        expdir = join(rootdir, f"{Animal}_{Expi:02d}")
        for exptype in ["Evol", "Manif", "Gabor", "Pasu", "EvolRef", "allnat", "all"]:
            for featlayer in [".layer2.Bottleneck3", ".layer3.Bottleneck5", ".layer4.Bottleneck2"]:
                df = pd.read_csv(join(expdir, f"eval_predict_factreg-{featlayer}-{exptype}.csv"), index_col=(0,1))
                df["Animal"] = Animal
                df["Expi"] = Expi
                df_all = pd.concat([df_all, df], axis=0)
#%%
df_all = df_all.set_index("layer", append=True).swaplevel(0,2)
df_all = df_all.rename_axis(['layer', 'regressor', 'FeatRed'])
df_all["layer_s"] = df_all.layer.apply(lambda s:s.split(".")[1])
df_all.to_csv(join(sumdir, "Combined_FactorRegression_summary.csv"))

df_all_reset = df_all.reset_index(inplace=False)
validmsk = ~((df_all_reset.Animal == "Alfa") & (df_all_reset.Expi == 10))
#%%
validmsk = ~((df_all.Animal == "Alfa") & (df_all.Expi == 10))
df_all[validmsk].groupby(level=(0, 1, 2)).agg(("mean","sem","count"))\
    [["rho_p", "D2"]]
#%% Factor grid plot
g = sns.FacetGrid(df_all_reset[validmsk], col="img_space", row="regressor", hue="layer",
                  height=5, aspect=0.75, ylim=(-0.2, 1.0),
                  col_order=["all", "allnat", "Manif", "Gabor", "Pasu", "EvolRef", ])
g.map(sns.barplot, "FeatRed", "rho_p",
      order=["facttsr1", 'factor1', 'spmask1', 'featvec1',
             'facttsr3', 'factor3', 'spmask3', 'featvec3'], )
plt.suptitle(f"Compare Regression methods {Animal} {Expi:02d} {featlayer}")
# g.savefig(join(figdir, f"prediction_comparison_{Animal}_{Expi:02d}_{featlayer}.png"))
plt.show()
#%%
g = sns.FacetGrid(df_all_reset[validmsk & (df_all_reset.img_space == "allnat")],#Manif
                  col="layer_s", row="regressor",
                  height=5, aspect=0.75, ylim=(-0.2, 1.0),)
g.map(sns.barplot, "FeatRed", "rho_p",
      order=["facttsr1", 'factor1', 'spmask1', 'featvec1',
             'facttsr3', 'factor3', 'spmask3', 'featvec3'], )
plt.suptitle(f"Compare Regression methods all Expi, all layers, allnat image space", fontsize=14)
g.set_xticklabels(rotation=30)
g.set_titles(size=11)
plt.tight_layout()
# g.savefig(join(figdir, f"prediction_comparison_{Animal}_{Expi:02d}_{featlayer}.png"))
plt.show()