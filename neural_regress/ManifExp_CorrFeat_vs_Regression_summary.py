"""Collect the results from Penalized regression """
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
    # MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
    # EStats = \
    # loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)[
    #     'EStats']
    # ReprStats = \
    # loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)[
    #     'ReprStats']
    for Expi in range(1, 46):
        expdir = join(rootdir, f"{Animal}_{Expi:02d}")
        for featlayer in [".layer2.Bottleneck3", ".layer3.Bottleneck5", ".layer4.Bottleneck2"]:
            df = pd.read_csv(join(expdir, f"eval_predict_{featlayer}-{exptype}.csv"), index_col=(0,1))
            df["Animal"] = Animal
            df["Expi"] = Expi
            df_all = pd.concat([df_all, df], axis=0)
            # df_all = pd.concat([df_all, df], axis=0)
#%%
df_all.to_csv(join(sumdir, "Manif_Regression_summary.csv"))
df_all = df_all.set_index("layer", append=True).swaplevel(0,2)
df_all = df_all.rename_axis(['layer', 'regressor', 'FeatRed'])
#%%
df_all.groupby(level=(0, 1, 2)).agg(("mean","sem"))\
    [["rho_s", "rho_p", "D2"]].\
    to_csv(join(sumdir, "Manif_Regression_method_summary.csv"))
#%%
validmsk = ~((df_all.Animal == "Alfa") & (df_all.Expi == 10))
df_all[validmsk].groupby(level=(0, 1, 2)).agg(("mean","sem"))\
    [["rho_s", "rho_p", "D2"]].\
    to_csv(join(sumdir, "Manif_Regression_method_valid_summary.csv"))
#%%
validmsk = ~((df_all.Animal == "Alfa") & (df_all.Expi == 10))
df_all[validmsk].groupby(level=(0, 1, 2)).agg()[["rho_s", "rho_p", "D2"]]
#%%
df_all[validmsk].groupby(level=(0, 1, 2)).mean().plot(y="rho_p", kind="bar", stacked=False)
plt.show()
#%%
df_all_reset = df_all.reset_index(inplace=False)
validmsk = ~((df_all_reset.Animal == "Alfa") & (df_all_reset.Expi == 10))
sns.barplot(x="layer", y="rho_p", hue="regressor",
            data=df_all_reset[validmsk])
plt.title("Compare methods with Ridge Regression")
plt.savefig(join(sumdir, "Manif_Regression_method_valid_summary_cmp_regr.png"))
plt.show()
#%%
df_all_reset = df_all.reset_index(inplace=False)
validmsk = ~((df_all_reset.Animal == "Alfa") & (df_all_reset.Expi == 10))
sns.barplot(x="regressor", y="rho_p", hue="layer",
            data=df_all_reset[validmsk])
plt.title("Compare methods with Ridge Regression")
plt.savefig(join(sumdir, "Manif_Regression_method_valid_summary_cmp_layer.png"))
plt.show()
#%%
df_all_reset = df_all.reset_index(inplace=False)
validmsk = ~((df_all_reset.Animal == "Alfa") & (df_all_reset.Expi == 10))
sns.barplot(x="layer", y="rho_p", hue="FeatRed",
            data=df_all_reset[validmsk & df_all_reset.regressor.isin(["Ridge"])])
plt.title("Compare methods with Ridge Regression")
plt.savefig(join(sumdir, "Manif_Regression_method_valid_summary_cmp_Ridge.png"))
plt.show()
#%%
df_all_reset = df_all.reset_index(inplace=False)
validmsk = ~((df_all_reset.Animal == "Alfa") & (df_all_reset.Expi == 10))
sns.barplot(x="FeatRed", y="rho_p", hue="regressor", data=df_all_reset[validmsk])
plt.show()
