
import sys
from os.path import join
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
pixobj_dir = r"pixelobjectness"
save_path = join(pixobj_dir, r"pixel_objectness.pt")
#%% Only need to run once at first to translate the project.
def translate_caffe2torch():
    sys.path.append(r"D:\Github\pytorch-caffe")
    from caffenet import CaffeNet  # Pytorch-caffe converter
    # protofile = r"D:\Generator_DB_Windows\nets\upconv\fc6\generator.prototxt"
    weightfile = join(pixobj_dir, r'pixel_objectness.caffemodel')  # 'resnet50/resnet50.caffemodel'
    protofile = join(pixobj_dir, r"pixel_objectness.prototxt")
    net = CaffeNet(protofile, width=512, height=512, channels=3)
    print(net)
    net.load_weights(weightfile)
    torch.save(net.state_dict(), save_path)
    return net
#%% Define the model
class PixObjectiveNet(nn.Module):
    def __init__(self, pretrained=True):
        super(PixObjectiveNet, self).__init__()
        self.M = nn.Sequential(OrderedDict([
            ("conv1_1", nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ("relu1_1", nn.ReLU(inplace=True)),
            ("conv1_2", nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ("relu1_2", nn.ReLU(inplace=True)),
            ("pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=True)),
            ("conv2_1", nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ("relu2_1", nn.ReLU(inplace=True)),
            ("conv2_2", nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ("relu2_2", nn.ReLU(inplace=True)),
            ("pool2", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=True)),
            ("conv3_1", nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ("relu3_1", nn.ReLU(inplace=True)),
            ("conv3_2", nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ("relu3_2", nn.ReLU(inplace=True)),
            ("conv3_3", nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ("relu3_3", nn.ReLU(inplace=True)),
            ("pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=True)),
            ("conv4_1", nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ("relu4_1", nn.ReLU(inplace=True)),
            ("conv4_2", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ("relu4_2", nn.ReLU(inplace=True)),
            ("conv4_3", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
            ("relu4_3", nn.ReLU(inplace=True)),
            ("pool4", nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)),
            ("conv5_1", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))),
            ("relu5_1", nn.ReLU(inplace=True)),
            ("conv5_2", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))),
            ("relu5_2", nn.ReLU(inplace=True)),
            ("conv5_3", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))),
            ("relu5_3", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)),
            ("pool5a", nn.AvgPool2d(kernel_size=3, stride=1, padding=1)),
            ("fc6", nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12))),
            ("relu6", nn.ReLU(inplace=True)),
            ("drop6", nn.Dropout(p=0.5, inplace=True)),
            ("fc7", nn.Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))),
            ("relu7", nn.ReLU(inplace=True)),
            ("drop7", nn.Dropout(p=0.5, inplace=True)),
            ("fc8_voc12", nn.Conv2d(1024, 2, kernel_size=(1, 1), stride=(1, 1)))
            ]))
        if pretrained:
            self.M.load_state_dict(torch.load(save_path))

    def forward(self, image: torch.Tensor, fullmap: bool=False) -> torch.Tensor:
        image = image[:, [2,1,0], :, :] - BGRmean.to(image.device) # RGB swap and normalize using BGRmean.
        Objmap = self.M(image)
        if fullmap:
            fullObjmap = F.interpolate(Objmap, size=[imgtsr.shape[2], imgtsr.shape[3]], mode='bilinear')
            return fullObjmap
        else:
            return Objmap

BGRmean = torch.Tensor([104.008, 116.669, 122.675]).reshape(1,3,1,1) # constant
PNet = PixObjectiveNet()
PNet.eval().cuda()
PNet.requires_grad_(False)
#%%
from skimage.io import imread
from kornia.filters import gaussian_blur2d
import matplotlib.pylab as plt
imgnm = join(pixobj_dir, r"images\block088_thread000_gen_gen087_003518.JPG")
#   r"images\block079_thread000_gen_gen078_003152.JPG"
img = imread(imgnm)
imgtsr = torch.from_numpy(img).float().permute([2,0,1]).unsqueeze(0)
imgtsr_pp = gaussian_blur2d(imgtsr, (5,5), (3,3))
objmap = PNet(imgtsr_pp.cuda(), fullmap=True).cpu()
objmsk = (objmap[:, 0, :, :] < objmap[:, 1, :, :]).numpy()
probmap = objmap[:, 1, :, :].numpy()
#%%
import numpy as np
def alpha_blend(img, mask, color_val=(0, 1, 0), alpha=0.5):
    if (img.dtype == np.uint8) or img.max() > 1.1:
        scale = 255
    else:
        scale = 1.0
    color_val = np.array(color_val).reshape([1, 1, 3]) * scale
    alphaimg = (mask[:, :, np.newaxis]) * alpha
    out_img = alphaimg * color_val + (1 - alphaimg) * img
    if scale == 255:
        return np.clip(out_img, 0, 255).astype("uint8")
    else:
        return out_img

# #%%
# from build_montages import make_grid_np
# grid = make_grid_np([img, imgmskblend, probmap], nrow=1)
#%
def visualize_result(objmap, img, titstr="", savenm="", figdir=""):
    objmsk = (objmap[:, 0, :, :] < objmap[:, 1, :, :]).numpy()
    probmap = objmap[:, 1, :, :].numpy()
    bkgrmap = objmap[:, 0, :, :].numpy()
    imgmskblend = alpha_blend(img, objmsk[0], alpha=0.4)
    figh = plt.figure(figsize=[13, 3])
    plt.subplot(1, 4, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1, 4, 2)
    plt.imshow(imgmskblend)
    plt.axis("off")
    plt.subplot(1, 4, 3)
    plt.imshow(probmap[0])
    plt.colorbar()
    plt.axis("off")
    plt.subplot(1, 4, 4)
    plt.imshow(bkgrmap[0])
    plt.colorbar()
    plt.axis("off")
    plt.suptitle(titstr)
    plt.savefig(join(figdir, "%s_pixobjmap.png"%(savenm,)))
    plt.savefig(join(figdir, "%s_pixobjmap.pdf"%(savenm,)))
    plt.show()
    return imgmskblend, figh

# create objectiveness map for each prototype in the evolution
figdir = "O:\ProtoObjectivenss"
from scipy.io import loadmat
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
Animal = "Alfa"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
for Expi in range(1, len(EStats)+1):
    # img = imread(imgnm)
    img = ReprStats[Expi-1].Evol.BestBlockAvgImg
    imgtsr = torch.from_numpy(img).float().permute([2,0,1]).unsqueeze(0)
    imgtsr_pp = gaussian_blur2d(imgtsr, (5, 5), (3,3))
    objmap = PNet(imgtsr_pp.cuda(), fullmap=True).cpu()
    visualize_result(objmap, img, titstr="%s Exp%02d EvolBlock Best image"%(Animal, Expi), savenm="%s_Exp%02d_EvolBlock"%(Animal, Expi), figdir=figdir)
#%% Evaluate correlation with correlation mask.
#%%
from easydict import EasyDict
from skimage.transform import resize
def norm_value(tsr):
    normtsr = (tsr - np.nanmin(tsr)) / (np.nanmax(tsr) - np.nanmin(tsr))
    return normtsr

def calc_map_from_tensors(D, res=256):
    cctsr = D["cctsr"].item()
    Ttsr = D["Ttsr"].item()
    layers = list(cctsr.keys())
    maps = EasyDict()
    maps.mean = EasyDict()
    maps.max = EasyDict()
    maps.tsig_mean = EasyDict()
    maps.mean_rsz = EasyDict()
    maps.max_rsz = EasyDict()
    maps.tsig_mean_rsz = EasyDict()
    for layer in layers:
        maps.max[layer] = np.nanmax(np.abs(cctsr[layer]), axis=0)
        maps.mean[layer] = np.nansum(np.abs(cctsr[layer]), axis=0) / cctsr[layer].shape[0]
        cctsr_layer = np.copy(cctsr[layer])
        cctsr_layer[np.abs(Ttsr[layer]) < 8] = 0
        maps.tsig_mean[layer] = np.nansum(np.abs(cctsr_layer), axis=0) / cctsr[layer].shape[0]
        maps.mean_rsz[layer] = norm_value(resize(maps.mean[layer], [res, res]))
        maps.max_rsz[layer] = norm_value(resize(maps.max[layer], [res, res]))
        maps.tsig_mean_rsz[layer] = norm_value(resize(maps.tsig_mean[layer], [res, res]))
    return maps


def visualize_msks(msk_list, titstr=None):
    nmsk = len(msk_list)
    figh, axs = plt.subplots(1, nmsk, figsize=[3*nmsk+1.5, 3])
    if nmsk == 1: axs = np.array([axs])
    else: axs = axs.reshape(-1)
    for i, msk in enumerate(msk_list):
        plt.sca(axs[i])
        plt.imshow(msk)
        plt.axis("off")
        plt.colorbar()
    if titstr is not None:
        plt.suptitle(titstr)
    plt.show()
    return figh
#%%
def merge_msks(msk_list, weights=None):
    nmsk = len(msk_list)
    if weights is None:
        weights = np.ones(nmsk)
    mmsk = sum(w * msk for w, msk in zip(weights, msk_list))
    mmsk = mmsk / np.sum(weights)
    return mmsk

mmsk = merge_msks(mean_map_rsz.values(), [map.mean() for map in max_map.values()])
visualize_msks([mmsk])
#%%
import pandas as pd
from scipy.stats import ttest_rel, ttest_ind, ranksums, pearsonr
def compare_maps(corrmap, probmap, thresh=0.5, namestr="", suffix=""):
    """Calculate stats from 2 maps. Tstats, """
    rval, ccpval = pearsonr(corrmap.flatten(), probmap.flatten())
    threshval =  thresh * (np.max(corrmap) - np.min(corrmap)) + np.min(corrmap)
    corrmsk = corrmap > threshval
    tval, pval = ttest_ind(probmap[corrmsk], probmap[~corrmsk])
    objin_m, objin_s, objin_MX = np.mean(probmap[corrmsk]), np.std(probmap[corrmsk]), np.max(probmap[corrmsk])
    objout_m, objout_s, objout_MX = np.mean(probmap[~corrmsk]), np.std(probmap[~corrmsk]), np.max(probmap[~corrmsk])
    print("%s Objness Ttest T=%.3f (P=%.1e) corr=%.3f(P=%.1e)\nin msk %.3f(%.3f) out msk %.3f(%.3f)"%\
        (namestr, tval, pval, rval, ccpval, objin_m, objin_s, objout_m, objout_s, ))
    S = EasyDict()
    for varnm in ["rval", "ccpval", "tval", "pval", "rval", "ccpval", "objin_m", "objin_s", "objin_MX", "objout_m", "objout_s", "objout_MX"]:
        S[varnm+suffix] = eval(varnm)
    return S
#%%
def dict_union(*args):
    S = EasyDict()
    for vdict in args:
        S.update(vdict)
    return S

#%%
ccdir = "S:\corrFeatTsr"
figdir = r"O:\ProtoObjectivenss\summary"
Scol = []
for Animal in ["Alfa", "Beto"]:
    EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
    ReprStats = loadmat(join(mat_path, Animal + "_ImageRepr.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['ReprStats']
    for Expi in range(1, len(EStats)+1):
        imgsize = EStats[Expi - 1].evol.imgsize
        imgpos = EStats[Expi - 1].evol.imgpos
        pref_chan = EStats[Expi - 1].evol.pref_chan
        metadict = {"Animal":Animal, "Expi":Expi, "imgsize":imgsize, "imgpos":imgpos, "pref_chan":pref_chan}
        imgpix = int(imgsize * 40)
        titstr = "%s Exp %d Driver Chan %d, %.1f deg [%s]" % (Animal, Expi, pref_chan, imgsize, tuple(imgpos))
        print(titstr)
        img = ReprStats[Expi-1].Evol.BestBlockAvgImg
        imgtsr = torch.from_numpy(img).float().permute([2,0,1]).unsqueeze(0)
        imgtsr_pp = gaussian_blur2d(imgtsr, (5, 5), (3,3))
        objmap = PNet(imgtsr_pp.cuda(), fullmap=True).cpu()
        objmsk = (objmap[:, 0, :, :] < objmap[:, 1, :, :]).numpy()[0]
        probmap = objmap[:, 1, :, :].numpy()[0]
        probmap_fg = np.copy(probmap)
        probmap_fg[~objmsk] = 0.0
        D = np.load(join(ccdir, "%s_Exp%d_EM_corrTsr.npz" % (Animal, Expi)),
                    allow_pickle=True)
        maps = calc_map_from_tensors(D)
        layernm = "conv4_3"
        corrmap = maps.max_rsz[layernm]
        S = compare_maps(corrmap, probmap, namestr="%s vs prob"%layernm)
        S_fg = compare_maps(corrmap, probmap_fg, namestr="%s vs fg"%layernm, suffix="_fg")#probmap)
        figh = visualize_msks([img, corrmap, probmap], titstr=titstr+" corr%.3f"%S.rval)
        figh.savefig(join(figdir,"%s_Exp%02d_objcorrmask_cmp.png"%(Animal, Expi)))
        Stot = dict_union(metadict, S, S_fg)
        Scol.append(Stot)
#%%
Both_df = pd.DataFrame(Scol)
Both_df.to_csv(join(figdir, "obj_corrmsk_cmp.csv"))

#%%
ITmsk = Both_df.pref_chan < 33
V1msk = (Both_df.pref_chan < 49) & (Both_df.pref_chan > 32)
V4msk = Both_df.pref_chan > 48
Both_df[]
#%%
visualize_msks(maps.max_rsz.values())
#%%
#%% More refined mask using RF mapping of CNN units
sys.path.append("E:\Github_Projects\pytorch-receptive-field")
from torch_receptive_field import receptive_field, receptive_field_for_unit
from torchvision.models import alexnet, vgg16
net = vgg16()
RF_dict = receptive_field(net.features, (3, 224, 224), device="cpu")
RF_for_unit = receptive_field_for_unit(RF_dict, (3, 224, 224), "8", (6,6))

#%%
resize()
