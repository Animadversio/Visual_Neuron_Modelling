""" Collect the results from Penalized regression
Plot the overall results across experiments, or individual experiments comparing methods.

"""
import re
import os
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
rootdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
sumdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress\summary"
exptype = "Manif"
df_all = pd.DataFrame()
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        expdir = join(rootdir, f"{Animal}_{Expi:02d}")
        for featlayer in [".layer2.Bottleneck3", ".layer3.Bottleneck5", ".layer4.Bottleneck2"]:
            df = pd.read_csv(join(expdir, f"eval_predict_{featlayer}-{exptype}.csv"), index_col=(0,1))
            df["Animal"] = Animal
            df["Expi"] = Expi
            df_all = pd.concat([df_all, df], axis=0)
#%%
df_all.to_csv(join(sumdir, "Manif_Regression_summary.csv"))
df_all = df_all.set_index("layer", append=True).swaplevel(0,2)
df_all = df_all.rename_axis(['layer', 'regressor', 'FeatRed'])
#
df_all_reset = df_all.reset_index(inplace=False)
validmsk = ~((df_all_reset.Animal == "Alfa") & (df_all_reset.Expi == 10))
#%%
df_all.groupby(level=(0, 1, 2)).agg(("mean","sem"))\
    [["rho_s", "rho_p", "D2"]].\
    to_csv(join(sumdir, "Manif_Regression_method_summary.csv"))
#%%
validmsk = ~((df_all.Animal == "Alfa") & (df_all.Expi == 10))
df_all[validmsk].groupby(level=(0, 1, 2)).agg(("mean","sem","count"))\
    [["rho_s", "rho_p", "D2"]].\
    to_csv(join(sumdir, "Manif_Regression_method_valid_summary.csv"))
#%%
validmsk = ~((df_all.Animal == "Alfa") & (df_all.Expi == 10))
df_all[validmsk].groupby(level=(0, 1, 2)).agg()[["rho_s", "rho_p", "D2"]]
#%%
df_all[validmsk].groupby(level=(0, 1, 2)).mean().plot(y="rho_p", kind="bar", stacked=False)
plt.show()
#%%
sns.barplot(x="layer", y="rho_p", hue="regressor",
            data=df_all_reset[validmsk])
plt.title("Compare methods with Ridge Regression")
plt.savefig(join(sumdir, "Manif_Regression_method_valid_summary_cmp_regr.png"))
plt.show()
#%%
sns.barplot(x="regressor", y="rho_p", hue="layer",
            data=df_all_reset[validmsk])
plt.title("Compare methods with Ridge Regression")
plt.savefig(join(sumdir, "Manif_Regression_method_valid_summary_cmp_layer_regr.png"))
plt.show()
#%%
sns.barplot(x="layer", y="rho_p", hue="FeatRed",
            data=df_all_reset[validmsk & df_all_reset.regressor.isin(["Ridge"])])
plt.title("Compare layer and feature reduction with Ridge Regression")
plt.savefig(join(sumdir, "Manif_Regression_method_valid_summary_cmp_Ridge.png"))
plt.show()
#%%
sns.barplot(x="regressor", y="rho_p", hue="FeatRed", data=df_all_reset[validmsk])
plt.title("Compare regressor and feature reduction")
plt.savefig(join(sumdir, "Manif_Regression_method_valid_summary_cmp_regr_Xtfm.png"))
plt.show()
#%%
"""
Visualize as barplot the prediction accuracy
for each layer and regressor individually for each experiment.
"""
rootdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
sumdir = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress\summary"
figdir = join(sumdir, "per_experiment")  # per experiment summary figure
exptype = "Manif"
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        df_exp = pd.DataFrame()
        expdir = join(rootdir, f"{Animal}_{Expi:02d}")
        for featlayer in [".layer2.Bottleneck3", ".layer3.Bottleneck5", ".layer4.Bottleneck2"]:
            for exptype in ["Evol", "Manif", "Gabor", "Pasu", "EvolRef", "all", 'allnat']:
                df = pd.read_csv(join(expdir, f"eval_predict_{featlayer}-{exptype}.csv"), index_col=(0,1))
                df_exp = pd.concat([df_exp, df], axis=0)
                df_fac = pd.read_csv(join(expdir, f"eval_predict_factreg-{featlayer}-{exptype}.csv"), index_col=(0, 1))
                df_exp = pd.concat([df_exp, df_fac], axis=0)

        df_exp = df_exp.rename_axis(["FeatRed", 'regressor', ])
        df_exp.reset_index(inplace=True)
        # raise NotImplementedError
        sns.set(font_scale=1.2)
        for featlayer in [".layer2.Bottleneck3", ".layer3.Bottleneck5", ".layer4.Bottleneck2"]:
            g = sns.FacetGrid(df_exp[df_exp.layer == featlayer], row="FeatRed", col="regressor",
                              height=5, aspect=0.75, ylim=(-0.2, 1.0))
            g.map(sns.barplot, "img_space", "rho_p",
                  order=["Evol", "Manif", "Gabor", "Pasu", "EvolRef"], )
            plt.suptitle(f"Compare Regression methods {Animal} {Expi:02d} {featlayer}")
            g.savefig(join(figdir, f"prediction_comparison_{Animal}_{Expi:02d}_{featlayer}.png"))
            plt.show()

#%%
df_exp = df_exp.rename_axis(["FeatRed", 'regressor',])
df_exp.reset_index(inplace=True)
#%%
df_exp[df_exp.layer == ".layer3.Bottleneck5"].plot(y="rho_p", x="img_space", kind="bar", stacked=False)
plt.show()
#%%
g = sns.FacetGrid(df_exp[df_exp.layer == ".layer3.Bottleneck5"], row="FeatRed", col="regressor", height=4, aspect=0.8)
g.map(sns.barplot, "img_space", "rho_p",
      order=["Evol", "Manif", "Gabor", "Pasu", "EvolRef"])#
plt.show()
#%%
from build_montages import crop_from_montage, make_grid_np
regr_cfgs = [
             ('pca', 'Ridge'),
             ('pca', 'Lasso'),
             ('pca', 'PLS'),
             ('srp', 'Ridge'),
             ('srp', 'Lasso'),
             ('srp', 'PLS'),
             ('sp_avg', 'Ridge'),
             ('sp_avg', 'Lasso'),
             ('sp_avg', 'PLS'),
]
for Animal in ["Alfa", "Beto"]:
    for Expi in range(1, 47):
        if Animal == "Beto" and Expi == 46: continue
        expdir = join(rootdir, f"{Animal}_{Expi:02d}")
        featvis_dir = join(expdir, "featvis")
        for featlayer in [".layer2.Bottleneck3", ".layer3.Bottleneck5", ".layer4.Bottleneck2"]:
            proto_col = []
            for regrlabel in regr_cfgs:
                Xtype, regressor = regrlabel
                mtg = plt.imread(join(featvis_dir, f"{Animal}-Exp{Expi:02d}-{featlayer}-{Xtype}-{regressor}_vis.png"))
                proto_first = crop_from_montage(mtg, (0, 0))
                proto_col.append(proto_first)
            method_mtg = make_grid_np(proto_col, nrow=3)
            plt.imsave(join(figdir, f"{Animal}-Exp{Expi:02d}-{featlayer}-regr_merge_vis.png"), method_mtg, )
#%%
