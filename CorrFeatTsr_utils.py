"""
Many small utils functions useful for Correlated Feature Visualization Analysis etc.
"""
from os.path import join
from easydict import EasyDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False

def saveallforms(figdirs, fignm, figh=None, fmts=["png","pdf"]):
    if type(figdirs) is str:
        figdirs = [figdirs]
    if figh is None:
        figh = plt.gcf()
    for figdir in figdirs:
        for sfx in fmts:
            figh.savefig(join(figdir, fignm+"."+sfx), bbox_inches='tight')


def area_mapping(num):
    if num <= 32: return "IT"
    elif num <= 48 and num >= 33: return "V1"
    elif num >= 49: return "V4"


def add_suffix(dict: dict, sfx: str=""):
    newdict = EasyDict()
    for k, v in dict.items():
        newdict[k + sfx] = v
    return newdict


def merge_dicts(dicts: list):
    newdict = EasyDict()
    for D in dicts:
        newdict.update(D)
    return newdict


def multichan2rgb(Hmaps):
    """Util function to summarize multi channel array to show as rgb"""
    if Hmaps.ndim == 2:
        Hmaps_plot = np.repeat(Hmaps[:,:,np.newaxis], 3, axis=2)
    elif Hmaps.shape[2] < 3:
        Hmaps_plot = np.concatenate((Hmaps, np.zeros((*Hmaps.shape[:2], 3 - Hmaps.shape[2]))), axis=2)
    else:
        Hmaps_plot = Hmaps[:, :, :3]
    Hmaps_plot = Hmaps_plot/Hmaps_plot.max()
    return Hmaps_plot


def showimg(ax, imgarr, cbar=False, ylabel=None):
    pcm = ax.imshow(imgarr)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if cbar:
        plt.colorbar(pcm, ax=ax)
    return pcm


def off_axes(axs):
    for ax in axs:
        ax.axis("off")