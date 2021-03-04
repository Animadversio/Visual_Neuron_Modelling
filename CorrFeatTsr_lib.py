"""A library of functions to do correlation feature analysis
Core machinery:
    Corr_Feat_Machine, to compute correlation online
Visualizing masks from `Corr_Feat_Machine`
    visual_cctsr
    visualize_cctsr_embed
Full pipeline
    ???, for given network, list of layers, list of pairs of input and output, do correlation feature analysis
"""
from os.path import join
import matplotlib.pylab as plt
import torch
import numpy as np
from skimage.transform import resize
from skimage.filters import gaussian
from collections import defaultdict
from scipy.stats import t
from easydict import EasyDict

#  New name array to match that of matlab
layername_dict = {"vgg16": ['conv1_1', 'conv1_1_relu',
                            'conv1_2', 'conv1_2_relu', 'pool1',
                            'conv2_1', 'conv2_1_relu',
                            'conv2_2', 'conv2_2_relu', 'pool2',
                            'conv3_1', 'conv3_1_relu',
                            'conv3_2', 'conv3_2_relu',
                            'conv3_3', 'conv3_3_relu', 'pool3',
                            'conv4_1', 'conv4_1_relu',
                            'conv4_2', 'conv4_2_relu',
                            'conv4_3', 'conv4_3_relu', 'pool4',
                            'conv5_1', 'conv5_1_relu',
                            'conv5_2', 'conv5_2_relu',
                            'conv5_3', 'conv5_3_relu', 'pool5',
                            'fc1', 'fc1_relu', 'dropout1',
                            'fc2', 'fc2_relu', 'dropout2',
                            'fc3'],
                  "densenet121": ['conv1',
                                  'bn1',
                                  'bn1_relu',
                                  'pool1',
                                  'denseblock1', 'transition1',
                                  'denseblock2', 'transition2',
                                  'denseblock3', 'transition3',
                                  'denseblock4',
                                  'bn2',
                                  'fc1'],
                  "resnet50": ["layer1", "layer2", "layer3", "layer4"]}


class Corr_Feat_Machine:
    """The machinery to hook up a layer in CNN and to compute correlation online with certain output
    """

    def __init__(self):
        self.feat_tsr = {}
        self.hooks = []
        self.layers = []
        self.innerProd = defaultdict(lambda: None)
        self.scoreS = None
        self.scoreSSq = None
        self.scoreM = None
        self.scoreStd = None
        self.featS = defaultdict(lambda: None)
        self.featSSq = defaultdict(lambda: None)
        self.featM = defaultdict(lambda: None)
        self.featStd = defaultdict(lambda: None)
        self.cctsr = defaultdict(lambda: None)
        self.Ttsr = defaultdict(lambda: None)

    def hook_forger(self, layer):
        # this function is important, or layer will be redefined in the same scope!
        def activ_hook(module, fea_in, fea_out):
            print("Extract from hooker on %s" % module.__class__)
            ref_feat = fea_out.detach().clone().cpu()
            ref_feat.requires_grad_(False)
            self.feat_tsr[layer] = ref_feat
            return None

        return activ_hook

    def register_hooks(self, net, layers, netname="vgg16"):
        if isinstance(layers, str):
            layers = [layers]

        for layer in layers:
            if netname == "vgg16":
                layer_idx = layername_dict[netname].index(layer)
                targmodule = net.features[layer_idx]
            elif netname == "resnet50":
                targmodule = net.__getattr__(layer)
            actH = targmodule.register_forward_hook(self.hook_forger(layer))
            self.hooks.append(actH)
            self.layers.append(layer)

    def init_corr(self):
        self.feat_tsr = {}
        self.innerProd = defaultdict(lambda: None)
        self.innerProd_M = defaultdict(lambda: None)
        self.scoreS = None
        self.scoreSSq = None
        self.scoreM = None
        self.scoreStd = None
        self.featS = defaultdict(lambda: None)
        self.featSSq = defaultdict(lambda: None)
        self.featMSq = defaultdict(lambda: None)
        self.featM = defaultdict(lambda: None)
        self.featStd = defaultdict(lambda: None)
        self.cctsr = defaultdict(lambda: None)
        self.Ttsr = defaultdict(lambda: None)
        self.imgN = 0

    def update_corr(self, score_tsr):
        self.imgN += score_tsr.shape[0]
        self.scoreS = score_tsr.sum(0) if self.scoreS is None else self.scoreS + score_tsr.sum(0)
        self.scoreSSq = (score_tsr ** 2).sum(0) if self.scoreSSq is None else self.scoreSSq + (score_tsr ** 2).sum(0)
        for layer, part_tsr in self.feat_tsr.items():
            innerProd_tmp = torch.einsum("i,ijkl->jkl", score_tsr, part_tsr, )  # [time by features]
            featS_tmp = part_tsr.sum(dim=0)
            featSSq_tmp = (part_tsr ** 2).sum(dim=0)
            self.innerProd[layer] = innerProd_tmp if self.innerProd[layer] is None else innerProd_tmp \
                                                                                        + self.innerProd[layer]
            self.featS[layer] = featS_tmp if self.featS[layer] is None else featS_tmp + self.featS[layer]
            self.featSSq[layer] = featSSq_tmp if self.featSSq[layer] is None else featSSq_tmp + self.featSSq[layer]

    def update_corr_rep(self, scorecol):
        """Multiple measurement of response for the same image.
        Responses are collected in a list in `scorecol`, the order match the order in the batch

        """
        repN = torch.tensor([len(scores) for scores in scorecol])
        actsum = torch.tensor([(scores).sum() for scores in scorecol]).float()
        actSqsum = torch.tensor([(scores ** 2).sum() for scores in scorecol]).float()
        self.imgN += repN.sum().item()  # score_tsr.shape[0]
        self.scoreS = actsum.sum(0) if self.scoreS is None else self.scoreS + actsum.sum(0)
        self.scoreSSq = actSqsum.sum(0) if self.scoreSSq is None else self.scoreSSq + actSqsum.sum(0)
        for layer, part_tsr in self.feat_tsr.items():
            featS_tmp = torch.einsum("i,ijkl->jkl", repN.float(), part_tsr, )  # weighted sum of activation tensor
            featSSq_tmp = torch.einsum("i,ijkl->jkl", repN.float(), part_tsr ** 2, )  # sum of activation square
            innerProd_tmp = torch.einsum("i,ijkl->jkl", actsum, part_tsr, )  # [time by features]
            self.innerProd[layer] = innerProd_tmp if self.innerProd[layer] is None else innerProd_tmp \
                                                                                        + self.innerProd[layer]
            self.featS[layer] = featS_tmp if self.featS[layer] is None else featS_tmp + self.featS[layer]
            self.featSSq[layer] = featSSq_tmp if self.featSSq[layer] is None else featSSq_tmp + self.featSSq[layer]

    def calc_corr(self, ):
        self.scoreM = self.scoreS / self.imgN
        scorMSq = self.scoreSSq / self.imgN
        self.scoreStd = (scorMSq - self.scoreM ** 2).sqrt()

        for layer in self.layers:
            self.featM[layer] = self.featS[layer] / self.imgN
            self.featMSq[layer] = self.featSSq[layer] / self.imgN
            self.innerProd_M[layer] = self.innerProd[layer] / self.imgN
            self.featStd[layer] = (self.featMSq[layer] - self.featM[layer] ** 2).sqrt()
            self.cctsr[layer] = (self.innerProd_M[layer] - self.scoreM * self.featM[layer]) / self.featStd[
                layer] / self.scoreStd
            self.Ttsr[layer] = np.sqrt(self.imgN - 2) * self.cctsr[layer] / (1 - self.cctsr[layer] ** 2).sqrt()

    def make_savedict(self, numpy=True):
        """make dict for saving, contains dictionaries of dictionary
        Contains the mean feature activation(featM), std feature activation (featStd),
        correlation tensor to each unit in the tensor (cctsr), T value for each units' correlation (Ttsr),
        number of image / sample that goes into correlation (imgN)
        """
        savedict = EasyDict()
        savedict.imgN = self.imgN
        savedict.cctsr = {layer: tsr.cpu().data.numpy() for layer, tsr in self.cctsr.items()}
        savedict.Ttsr = {layer: tsr.cpu().data.numpy() for layer, tsr in self.Ttsr.items()}
        savedict.featM = {layer: tsr.cpu().data.numpy() for layer, tsr in self.featM.items()}
        savedict.featStd = {layer: tsr.cpu().data.numpy() for layer, tsr in self.featStd.items()}
        return savedict

    def clear_hook(self):
        for h in self.hooks:
            h.remove()

    def __del__(self):
        self.clear_hook()
        print('Feature Correlator Destructed, Hooks deleted.')


#%%
def visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, ExpType, Titstr, figdir=""):
    """
    Demo
    ExpType = "EM_cmb"
    layers2plot = ['conv3_3', 'conv4_3', 'conv5_3']
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, ExpType, )
    figh.savefig(join("S:\corrFeatTsr","VGGsummary","%s_Exp%d_%s_corrTsr_vis.png"%(Animal,Expi,ExpType)))
    :param featFetcher:
    :param layers2plot:
    :param ReprStats:
    :param Expi:
    :param Animal:
    :param Titstr:
    :return:
    """
    nlayer = len(layers2plot)
    figh, axs = plt.subplots(3,nlayer,figsize=[10/3*nlayer,10])
    if ReprStats is not None:
        axs[0,0].imshow(ReprStats[Expi-1].Evol.BestImg)
        axs[0,0].set_title("Best Evol Img")
        axs[0,0].axis("off")
        axs[0,1].imshow(ReprStats[Expi-1].Evol.BestBlockAvgImg)
        axs[0,1].set_title("Best BlockAvg Img")
        axs[0,1].axis("off")
        axs[0,2].imshow(ReprStats[Expi-1].Manif.BestImg)
        axs[0,2].set_title("Best Manif Img")
        axs[0,2].axis("off")
    for li, layer in enumerate(layers2plot):
        chanN = featFetcher.cctsr[layer].shape[0]
        tmp=axs[1,li].matshow(np.nansum(featFetcher.cctsr[layer].abs().numpy(),axis=0) / chanN)
        plt.colorbar(tmp, ax=axs[1,li])
        axs[1,li].set_title(layer+" mean abs cc")
        tmp=axs[2,li].matshow(np.nanmax(featFetcher.cctsr[layer].abs().numpy(),axis=0))
        plt.colorbar(tmp, ax=axs[2,li])
        axs[2,li].set_title(layer+" max abs cc")
    figh.suptitle("%s Exp%d Corr Tensor %s %s"%(Animal, Expi, ExpType, Titstr))
    plt.show()
    figh.savefig(join(figdir, "%s_Exp%d_%s_corrTsr_vis.png" % (Animal, Expi, ExpType)))
    figh.savefig(join(figdir, "%s_Exp%d_%s_corrTsr_vis.pdf" % (Animal, Expi, ExpType)))
    return figh

def visualize_cctsr_embed(featFetcher, layers2plot, ReprStats, Expi, Animal, ExpType, Titstr, figdir="", imgpix=120, fullimgsz=224, borderblur=True):
    """
    Demo
    ExpType = "EM_cmb"
    layers2plot = ['conv3_3', 'conv4_3', 'conv5_3']
    figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, ExpType, )
    figh.savefig(join("S:\corrFeatTsr","VGGsummary","%s_Exp%d_%s_corrTsr_vis.png"%(Animal,Expi,ExpType)))
    :param featFetcher:
    :param layers2plot:
    :param ReprStats:
    :param Expi:
    :param Animal:
    :param Titstr:
    :return:
    """
    nlayer = len(layers2plot)
    figh, axs = plt.subplots(3,nlayer,figsize=[10/3*nlayer, 10])
    if ReprStats is not None:
        protos = [ReprStats[Expi-1].Evol.BestImg,\
                  ReprStats[Expi-1].Evol.BestBlockAvgImg,\
                  ReprStats[Expi-1].Manif.BestImg]
        labels = ["Best Evol Img",
                "Best BlockAvg Img",
                "Best Manif Img"]
        imgpix = min(imgpix, fullimgsz)
        padbef = (fullimgsz - imgpix) // 2
        padaft = (fullimgsz - imgpix) - padbef
        for i in range(3):
            proto_rsz = resize(protos[i], [imgpix, imgpix])
            background = 0.5*np.ones([fullimgsz,fullimgsz,3])
            proto_rsz_pad = np.pad(proto_rsz, [(padbef, padaft), (padbef, padaft), (0, 0)], 'constant', constant_values=0.5)
            if borderblur:
                border = round(imgpix * 0.05)
                msk = np.ones([imgpix - 2 * border, imgpix - 2 * border])
                msk = np.pad(msk, [(padbef + border, padaft + border), (padbef + border, padaft + border)], 'constant', constant_values=0)
                blurmsk = (msk < 0.5).reshape([fullimgsz, fullimgsz,1]);
                blurmsk_trans = gaussian(blurmsk, 3, mode='reflect')
                # bkgrd_img = gaussian(proto_rsz_pad * blurmsk, (3, 3), mode='reflect')
                brdblur_proto = background * blurmsk_trans + proto_rsz_pad * (1-blurmsk_trans)
                axs[0, i].imshow(brdblur_proto)
            else:
                axs[0,i].imshow(proto_rsz_pad)
            axs[0,i].set_title(labels[i])
            axs[0,i].axis("off")
    for li, layer in enumerate(layers2plot):
        tmp=axs[1,li].matshow(featFetcher.cctsr[layer].abs().numpy().mean(axis=0))
        plt.colorbar(tmp, ax=axs[1,li])
        axs[1,li].set_title(layer+" mean abs cc")
        tmp=axs[2,li].matshow(featFetcher.cctsr[layer].abs().numpy().max(axis=0))
        plt.colorbar(tmp, ax=axs[2,li])
        axs[2,li].set_title(layer+" max abs cc")
    figh.suptitle("%s Exp%d Corr Tensor %s %s"%(Animal, Expi, ExpType, Titstr))
    plt.show()
    figh.savefig(join(figdir, "%s_Exp%d_%s_corrTsr_vis.png" % (Animal, Expi, ExpType)))
    figh.savefig(join(figdir, "%s_Exp%d_%s_corrTsr_vis.pdf" % (Animal, Expi, ExpType)))
    return figh

#%%
from tqdm import tqdm
def Corr_Feat_pipeline(net, featFetcher, score_vect, imgfullpath_vect, imgload_func, online_compute=True, batchsize=121, savedir="S:\corrFeatTsr", savenm="Evol"):
    imgN = len(imgfullpath_vect)
    if type(score_vect) is not list:
        score_tsr = torch.tensor(score_vect).float()  # torchify the score vector
        rep_score = False
    else:
        rep_score = True
    csr = 0
    pbar = tqdm(total=imgN)
    while csr < imgN:
        cend = min(csr + batchsize, imgN)
        input_tsr = imgload_func(imgfullpath_vect[csr:cend])  #
        # input_tsr = loadimg_embed_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
        # Pool through VGG
        with torch.no_grad():
            part_tsr = net(input_tsr.cuda()).cpu()
        if rep_score:
            featFetcher.update_corr_rep(score_vect[csr:cend])
        else:
            featFetcher.update_corr(score_tsr[csr:cend])
        # update bar!
        pbar.update(cend - csr)
        csr = cend
    pbar.close()
    featFetcher.calc_corr()
    np.savez(join(savedir, "%s_corrTsr.npz" % (savenm)), **featFetcher.make_savedict())

#%%
#
# figdir = join("S:\corrFeatTsr", "VGGsummary")
# layers2plot = ["conv5_3", "conv4_3", "conv3_3", "conv2_2", ]
# for Expi in range(1, len(EStats) + 1):
#     imgsize = EStats[Expi - 1].evol.imgsize
#     imgpos = EStats[Expi - 1].evol.imgpos
#     pref_chan = EStats[Expi - 1].evol.pref_chan
#     imgpix = int(imgsize * 40)
#     titstr = "Driver Chan %d, %.1f deg [%s]" % (pref_chan, imgsize, tuple(imgpos))
#
#     featFetcher = Corr_Feat_Machine()
#     featFetcher.register_hooks(VGG, ["conv2_2", "conv3_3", "conv4_3", "conv5_3"])
#     featFetcher.init_corr()
#
#     score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)])
#     imgN = len(imgfullpath_vect)
#     score_tsr = torch.tensor(score_vect).float()  # torchify the score vector
#     csr = 0
#     pbar = tqdm(total=imgN)
#     while csr < imgN:
#         cend = min(csr + batchsize, imgN)
#         input_tsr = loadimg_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix)
#         # input_tsr = loadimg_embed_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
#         # Pool through VGG
#         with torch.no_grad():
#             part_tsr = VGG.features(input_tsr.cuda()).cpu()
#         featFetcher.update_corr(score_tsr[csr:cend])
#         # update bar!
#         pbar.update(cend - csr)
#         csr = cend
#     pbar.close()
#     featFetcher.calc_corr()
#     np.savez(join("S:\corrFeatTsr", "%s_Exp%d_Evol_corrTsr.npz" % (Animal, Expi)), **featFetcher.make_savedict())
#     figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "Evol", titstr, figdir=figdir)
#
#     # Load manifold experiment, single trial response
#     scorecol_M, imgfullpath_vect_M = load_score_mat(EStats, MStats, Expi, "Manif_sgtr", wdws=[(50, 200)])
#     imgN_M = len(imgfullpath_vect_M)
#     csr = 0
#     pbar = tqdm(total=imgN_M)
#     while csr < imgN_M:
#         cend = min(csr + batchsize, imgN_M)
#         input_tsr = loadimg_preprocess(imgfullpath_vect[csr:cend], imgpix=imgpix)
#         # input_tsr = loadimg_embed_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
#         # Pool through VGG
#         with torch.no_grad():
#             part_tsr = VGG.features(input_tsr.cuda()).cpu()
#         featFetcher.update_corr_rep(scorecol_M[csr:cend])
#         # update bar!
#         pbar.update(cend - csr)
#         csr = cend
#     pbar.close()
#
#     featFetcher.calc_corr()
#     np.savez(join("S:\corrFeatTsr", "%s_Exp%d_EM_corrTsr.npz" % (Animal, Expi)), **featFetcher.make_savedict())
#     figh = visualize_cctsr(featFetcher, layers2plot, ReprStats, Expi, Animal, "EM_sgtr", titstr, figdir=figdir)
#     featFetcher.clear_hook()