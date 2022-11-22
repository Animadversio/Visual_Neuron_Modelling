""" Visualize and summarize the results comparing different neural regression fit_models."""
import sys
import os
from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
dataroot = r"E:\OneDrive - Harvard University\CNN_neural_regression"
#%%
def get_result_summary(dataroot):
    """
    Get the result summary from the result folder
    """
    result_path = join(dataroot, "resnet50_linf8")
    result_files = [f for f in os.listdir(result_path) if f.endswith(".csv")]
    result_files.sort()
    result_df = pd.DataFrame()
    for f in result_files:
        patt = re.findall(r"(.*)_Exp(\d*)_resnet50_linf8_(.*)_regression_results", f) # Alfa_Exp01_resnet50_linf8_.layer2.Bottleneck3_regression_results
        if len(patt) == 0:
            print("Error: file name pattern not matching: {}".format(f))
            continue
        else:
            Animal, Expi, layer = patt[0]
            df = pd.read_csv(join(result_path, f))
            df["Animal"] = Animal
            df["Exp"] = Expi
            df["Layer"] = layer
            result_df = result_df.append(df)

    return result_df


def rename_col(result_df, ):
    """Rename the column names"""
    result_df.rename(columns={'Unnamed: 0': "Xtype",
                              'Unnamed: 1': "ytype",
                              'Unnamed: 2': "Regressor",}, inplace=True)
    return result_df


def shorten_layer_name(result_df, ):
    """Shorten the layer name"""
    result_df["ShortLayer"] = result_df["Layer"].apply(lambda x:
       x.replace(".layer", "layer").replace("Bottleneck", "B"))
    return result_df

result_df = get_result_summary(dataroot)
result_df = shorten_layer_name(result_df)
result_df = rename_col(result_df)
#%%
def plot_result_summary(result_df, ):
    """"""
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.barplot(x="Epoch", y="test_score", hue="Layer", data=result_df)
    plt.legend(loc="upper right")
    plt.show()

plot_result_summary(result_df)
#%%
result_df.groupby(["Xtype", "ytype", "Regressor", "ShortLayer", ]).mean()["test_score"]\
    .unstack().plot(kind="barh", stacked=False, figsize=(8, 6), )
plt.xlim([-0.1, 0.35])
plt.tight_layout()
plt.show()
#%%
result_df.groupby(["Xtype", "Regressor", "ShortLayer", ]).median()["test_score"]\
    .unstack().plot(kind="barh", stacked=False, figsize=(8, 6), )
plt.xlim([-0.1, 0.25])
plt.xlabel("Test score (median)")
plt.tight_layout()
plt.show()
#%%
"""Bar plot for the test score distribution """
result_df.groupby(["Regressor", "Layer", ]).mean()["test_score"]\
    .unstack().plot(kind="barh", stacked=False, figsize=(10, 6), )
plt.xlim([-0.2, 0.35])
plt.tight_layout()
plt.show()
#%%
result_df[(result_df.ShortLayer == "layer3.B5")].groupby(["Regressor", "Xtype", ])\
    .mean()["test_score"]\
    .unstack(-1).plot(kind="barh", stacked=False, figsize=(8, 6), )
plt.xlim([-0.05, 0.35])
plt.tight_layout()
plt.show()

result_df[(result_df.ShortLayer == "layer3.B5")].groupby(["Regressor", "Xtype", ])\
    .mean()["test_score"]\
    .unstack(-2).plot(kind="barh", stacked=False, figsize=(8, 6), )
plt.xlim([-0.05, 0.35])
plt.tight_layout()
plt.show()
#%%
"""Box plot for the test score distribution """
result_df\
    .boxplot(by=["Xtype", "Regressor", "ShortLayer", ],
             column="test_score", figsize=(8, 10),
             vert=False, grid=True, )
plt.xlim([-0.2, 0.8])
plt.tight_layout()
plt.show()
#%%
result_df.groupby(["Animal", "Exp", ]).max()\
    .plot(kind="barh", y="test_score", stacked=False, figsize=(6, 10), )
plt.xlim([0, 0.70])
plt.xlabel("Max test D2 score")
plt.tight_layout()
plt.show()
#%%
result_df.groupby(["Animal", "Exp", ]).max()["test_score"].transform(np.sqrt)\
    .plot(kind="barh", y="test_score", stacked=False, figsize=(6, 10), )
plt.xlim([0, 0.85])
plt.xlabel("Max test correlation")
plt.tight_layout()
plt.show()
#%%
result_df.groupby(["Animal", "Exp", ]).max()["test_score"]\
    .plot(kind="hist", y="test_score", figsize=(6, 5), bins=15, )
# plt.xlim([0, 0.85])
plt.xlabel("Max test D2 score")
plt.tight_layout()
plt.show()
#%%
result_df.groupby(["Animal", "Exp", ]).max()["test_score"].transform(np.sqrt)\
    .plot(kind="hist", y="test_score", figsize=(6, 5), bins=15, )
# plt.xlim([0, 0.85])
plt.xlabel("Max test correlation")
plt.tight_layout()
plt.show()
#%%
result_df.groupby(["Animal", "Exp", ])["test_score"]\
    .plot(kind="scatter", y="test_score", figsize=(6, 5), bins=15, )
plt.xlabel("Max test correlation")
plt.tight_layout()
plt.show()
#%%
result_df.groupby(["Animal", "Exp", "Xtype", "Regressor", "ShortLayer"])["test_score"]\
    .plot(kind="scatter", y="test_score", figsize=(6, 5), bins=15, )
plt.xlabel("Max test correlation")
plt.tight_layout()
plt.show()
#%%
result_df.to_csv(join(dataroot, "result_summary.csv"))

align_result = result_df.groupby(["Animal", "Exp", "Xtype", "Regressor", "ShortLayer"])["test_score"].mean().unstack([-3,-2,-1])
align_result.to_csv(join(dataroot, "align_result_summary.csv"))
#%%
msk = (result_df.ShortLayer == "layer3.B5") #& (result_df.Regressor!="Kernel_poly")
align_result_select = result_df[msk].groupby(["Animal", "Exp", "Xtype", "Regressor", "ShortLayer"])["test_score"].mean().unstack([-3,-2,-1])
align_result_select.to_csv(join(dataroot, "align_result_select_summary.csv"))

#%% Testing the relative strength of the different regressors x Xtypes
from scipy.stats import spearmanr, pearsonr, ttest_ind, ttest_rel, ranksums, wilcoxon
wilcoxon(align_result_select[("srp", "Ridge", "layer3.B5")], \
          align_result_select[("srp", "Poisson", "layer3.B5")])

#%% Comparing different  (Xtype, Regressor) with same Layer (layer3.B5)
stat_mat = np.ones((len(align_result_select.columns), len(align_result_select.columns)))
for c1, col1 in enumerate(align_result_select.columns):
    for c2, col2 in enumerate(align_result_select.columns):
        tval, pval = ttest_rel(align_result_select[col1],
                               align_result_select[col2], nan_policy="omit")
        if c1 < c2 and pval < 0.05:
            print(col1[:2], col2[:2], "P=%.1e t=%.3f"%(pval, tval))
        stat_mat[c1, c2] = pval
#%% Comparing different layers with same (Xtype, Regressor)
stat_mat = np.ones((len(align_result.columns), len(align_result.columns)))
for c1, col1 in enumerate(align_result_select.columns):
    for layer in ['layer2.B3', "layer4.B2"]:
        col2 = (*col1[:2], layer)
        # tval, pval = wilcoxon(align_result[col1], align_result[col2])
        tval, pval = ttest_rel(align_result[col1],
                               align_result[col2], nan_policy="omit")
        if pval < 0.05:
            print(col1[:2], col1[2], "vs", layer, "P=%.1e t=%.3f"%(pval, tval))
