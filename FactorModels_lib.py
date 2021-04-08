from skimage.io import imread, imread_collection
from torchvision import models
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from tqdm import tqdm
from CorrFeatTsr_lib import layername_dict
from os.path import join
from glob import glob
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
from scipy.io import loadmat
from easydict import EasyDict
mat_path = r"E:\OneDrive - Washington University in St. Louis\Mat_Statistics"
Pasupath = r"N:\Stimuli\2019-Manifold\pasupathy-wg-f-4-ori"
Gaborpath = r"N:\Stimuli\2019-Manifold\gabor"
#%%
class FactModel(nn.Module):
    def __init__(self, basenet="vgg16", layername="conv9", img_dim=256):
        super(FactModel, self).__init__()
        if basenet == "vgg16":
            VGG = models.vgg16(pretrained=True)
            layer_idx = layername_dict[basenet].index(layername)
            self.FeatNet = VGG.features[:layer_idx + 1].cuda()
        dummy = torch.zeros([1, 3, img_dim, img_dim]).cuda()
        feat_tsr = self.FeatNet(dummy)
        self.ch, self.maskH, self.maskW = tuple(feat_tsr.shape[-3:])
        print("Using pretrained mode %s upto layer %s, %d channels, (%d, %d) spatial"%(basenet, layername, self.ch, self.maskH, self.maskW))
        self.feat_trans = nn.Conv2d(self.ch, 1, (1, 1)).cuda()  # feature transform
        self.sp_mask = nn.Linear(self.maskH * self.maskW, 1).cuda()  # assume scaler rate output
        self.optimizer = Adam(list(self.feat_trans.parameters())+list(self.sp_mask.parameters()), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)

    def forward(self, img):
        x = self.FeatNet(img)
        x = self.feat_trans(x).reshape([-1, self.maskH * self.maskW])
        r = F.relu(self.sp_mask(x))
        return r

lpls_ftr = torch.tensor([[ 0, -1,  0],
                         [-1, -4, -1],
                         [ 0, -1,  0]]).reshape(1,1,3,3).float().cuda()

def spatial_regularizer(tsr):
    """Compute"""
    if tsr.ndim == 3:
        tsr = tsr.unsqueeze(1)
    elif tsr.ndim == 2:
        tsr = tsr.unsqueeze(0).unsqueeze(0)
    # C = tsr.shape[1]
    # the groups enforce the same filter applied to each channel independently. 
    lpls_energy = torch.sqrt(F.conv2d(tsr, lpls_ftr, ).pow(2).sum())
    return lpls_energy

class FactorRegr_Machine:
    """The machinery to hook up a layer in CNN and to perform online regression of scores onto the activations of that layer. 
    """
    def __init__(self, Nfactor=2, spatial_nonneg=True, feat_nonneg=False, device="cuda"):
        self.feat_tsr = {}
        self.feat_tsr_shape = {}
        self.prediction = {}
        self.hooks = []
        self.layers = []
        # model configs 
        self.Nfactor = Nfactor # number of factors / rank in the factorized regression model
        self.sp_nonneg = spatial_nonneg # Spatial component is positive
        self.ft_nonneg = feat_nonneg
        self.device = device
        self.flatten = nn.Flatten(start_dim=1) # just put it here.
        # training configs
        self.smooth_lambda = 1E-3
        self.spL2_lambda = 1E-3
        self.ftL2_lambda = 1E-3
        self.lr = 1E-3

    def hook_forger(self, layer):
        # this function is important, or layer will be redefined in the same scope!
        def activ_hook(module, fea_in, fea_out):
            # print("Extract from hooker on %s" % module.__class__)
            ref_feat = fea_out.detach()#.clone().cpu()
            #ref_feat.requires_grad_(False)
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

    def clear_hook(self):
        for h in self.hooks:
            h.remove()

    def __del__(self):
        self.clear_hook()
        print('Feature Extractor Destructed, Hooks deleted.')

    def init_model(self, net, img_dim=256):
        """init model after hook up the activations
        The factors / weights will be set up
        """
        self.feat_tsr = {}
        self.prediction = {}
        self.sp_mask = {}
        self.feat_trans = {}
        self.full_predtsr = {}
        self.full_scoretsr = None
        self.optimizer = {}
        self.imgN = 0
        dummy = torch.zeros([1, 3, img_dim, img_dim]).to(self.device)
        with torch.no_grad():
            net(dummy)
        for layer, part_tsr in self.feat_tsr.items():
            _, C, H, W = part_tsr.shape
            self.feat_tsr_shape[layer] = (C, H, W)
            self.sp_mask[layer] = nn.Linear(in_features=self.Nfactor * H * W, out_features=1).to(self.device)
            self.feat_trans[layer] = nn.Conv2d(in_channels=C, out_channels=self.Nfactor, kernel_size=1, bias=False).to(self.device)
            self.optimizer[layer] = Adam(list(self.feat_trans[layer].parameters())+list(self.sp_mask[layer].parameters()), lr=self.lr, weight_decay=0)


    def predict(self):
        """Predict score after passing image through net you hooked"""
        for layer, part_tsr in self.feat_tsr.items():
            compress_tsr = self.feat_trans[layer](part_tsr.to(self.device))
            score_vec = self.sp_mask[layer](self.flatten(compress_tsr))
            self.prediction[layer] = score_vec
            if layer not in self.full_predtsr:
                self.full_predtsr[layer] = score_vec.detach().cpu().squeeze()
            else:
                self.full_predtsr[layer] = torch.cat((self.full_predtsr[layer], score_vec.detach().cpu().squeeze()), dim=0)

    def record_score(self, score_tsr):
        if self.full_scoretsr is None:
            self.full_scoretsr = score_tsr.detach().cpu()
        else:
            self.full_scoretsr = torch.cat((self.full_scoretsr, score_tsr.detach().cpu()), dim=0)

    def update_model(self, score_tsr):
        """
        score_tsr: a n torch tensor as target
        """
        self.predict() # get predictions by passing activations through the
        if self.full_scoretsr is None:
            self.full_scoretsr = score_tsr.detach().cpu()
        else:
            self.full_scoretsr = torch.cat((self.full_scoretsr, score_tsr.detach().cpu()), dim=0)
        for layer, pred_score in self.prediction.items():
            _, H, W = self.feat_tsr_shape[layer]
            self.optimizer[layer].zero_grad()
            # compute loss and regularization term
            loss = F.mse_loss(score_tsr.to(self.device), pred_score)
            lpls_reg = spatial_regularizer(self.sp_mask[layer].weight.reshape(self.Nfactor, H, W))
            sp_L2reg = self.sp_mask[layer].weight.pow(2).sum()
            ft_L2reg = self.feat_trans[layer].weight.pow(2).sum()
            print("Loss: %.3f smoothReg: %.3f spatialL2: %.3f featL2: %.3f"%(loss.item(), lpls_reg.item(), sp_L2reg.item(), ft_L2reg.item()))
            loss += self.smooth_lambda * lpls_reg + self.spL2_lambda * sp_L2reg + self.ftL2_lambda * ft_L2reg
            loss.backward()
            # maybe record loss 
            self.optimizer[layer].step()
            if self.sp_nonneg:
                self.sp_mask[layer].weight.data.clamp_(min=0)  # spatial mask set to be non negative.
                # debug the non negative
            if self.ft_nonneg:
                self.feat_trans[layer].weight.data.clamp_(min=0)  # spatial mask set to be non negative.

    def clear_record(self):
        self.full_scoretsr = None
        self.full_predtsr = {}

    def record_summary(self):
        for layer, predvec in self.full_predtsr.items():
            cc = np.corrcoef(predvec.numpy(), self.full_scoretsr.numpy())[0,1]
            mse = F.mse_loss(predvec, self.full_scoretsr) 
            print("Layer %s: Corr %.3f MSE %.3f"%(layer, cc, mse))

    def visualize_weights(self, savestr="", figdir=""):
        for layer in self.feat_trans:
            NF = self.Nfactor
            C,H,W = self.feat_tsr_shape[layer]
            feat_vecs = self.feat_trans[layer].weight.detach().clone().cpu()
            sp_msks = self.sp_mask[layer].weight.detach().clone().cpu()
            feat_vecs = feat_vecs.reshape((NF, C))
            sp_msks = sp_msks.reshape((NF, H, W))

            plt.figure(figsize=[11,6])
            for i in range(NF):
                plt.subplot(3,NF,i+1)
                plt.imshow(sp_msks[i,:,:].squeeze().numpy())
                plt.subplot(3,NF,i+1+NF)
                plt.plot(feat_vecs[i,:].numpy(),alpha=0.5)
                plt.subplot(3,NF,i+1+2*NF)
                plt.plot(sorted(feat_vecs[i,:].numpy()),alpha=0.5)
            plt.suptitle(layer+" lr%.1e smoothLambda %.1e spatialL2 %.1e featL2 %.1e"%(self.lr, self.smooth_lambda, self.spL2_lambda, self.ftL2_lambda, ))
            plt.savefig(join(figdir, "%s_%s_weightVis.png"%(savestr, layer)))
            plt.show()

    def make_savedict(self, numpy=True):
        """make dict for saving, contains dictionaries of dictionary
        Contains the mean feature activation(featM), std feature activation (featStd),
        correlation tensor to each unit in the tensor (cctsr), T value for each units' correlation (Ttsr),
        number of image / sample that goes into correlation (imgN)
        """
        savedict = EasyDict()
        NF = self.Nfactor
        for layer in self.feat_trans:
            savedict[layer] = EasyDict()
            C,H,W = self.feat_tsr_shape[layer]
            feat_vecs = self.feat_trans[layer].weight.detach().clone().cpu().numpy()
            sp_msks = self.sp_mask[layer].weight.detach().clone().cpu().numpy()
            bias = self.sp_mask[layer].bias.detach().clone().cpu().numpy()
            savedict[layer].feat_vecs = feat_vecs.reshape((NF, C))
            savedict[layer].sp_msks = sp_msks.reshape((NF, H, W))
            savedict[layer].bias = bias.reshape((-1))
            savedict[layer].shape = (C,H,W)
            savedict[layer].NF = NF
        return savedict
    # def record_prediction(self, score_tsr):
    #     for layer, pred_score in self.prediction.items():
    #         if layer not in self.full_predvec


    # def update_model_rep(self, scorecol):
    #     """Multiple measurement of response for the same image.
    #     Responses are collected in a list in `scorecol`, the order match the order in the batch
    #     """
    #     repN = torch.tensor([len(scores) for scores in scorecol])
    #     actsum = torch.tensor([(scores).sum() for scores in scorecol]).float()
    #     actSqsum = torch.tensor([(scores ** 2).sum() for scores in scorecol]).float()
    #     self.imgN += repN.sum().item()  # score_tsr.shape[0]
    #     self.scoreS = actsum.sum(0) if self.scoreS is None else self.scoreS + actsum.sum(0)
    #     self.scoreSSq = actSqsum.sum(0) if self.scoreSSq is None else self.scoreSSq + actSqsum.sum(0)
    #     for layer, part_tsr in self.feat_tsr.items():
    #         featS_tmp = torch.einsum("i,ijkl->jkl", repN.float(), part_tsr, )  # weighted sum of activation tensor
    #         featSSq_tmp = torch.einsum("i,ijkl->jkl", repN.float(), part_tsr ** 2, )  # sum of activation square
    #         innerProd_tmp = torch.einsum("i,ijkl->jkl", actsum, part_tsr, )  # [time by features]
    #         self.innerProd[layer] = innerProd_tmp if self.innerProd[layer] is None else innerProd_tmp \
    #                                                                                     + self.innerProd[layer]
    #         self.featS[layer] = featS_tmp if self.featS[layer] is None else featS_tmp + self.featS[layer]
    #         self.featSSq[layer] = featSSq_tmp if self.featSSq[layer] is None else featSSq_tmp + self.featSSq[layer]

    # def calc_corr(self, ):
    #     self.scoreM = self.scoreS / self.imgN
    #     scorMSq = self.scoreSSq / self.imgN
    #     self.scoreStd = (scorMSq - self.scoreM ** 2).sqrt()

    #     for layer in self.layers:
    #         self.featM[layer] = self.featS[layer] / self.imgN
    #         self.featMSq[layer] = self.featSSq[layer] / self.imgN
    #         self.innerProd_M[layer] = self.innerProd[layer] / self.imgN
    #         self.featStd[layer] = (self.featMSq[layer] - self.featM[layer] ** 2).sqrt()
    #         self.cctsr[layer] = (self.innerProd_M[layer] - self.scoreM * self.featM[layer]) / self.featStd[
    #             layer] / self.scoreStd
    #         self.Ttsr[layer] = np.sqrt(self.imgN - 2) * self.cctsr[layer] / (1 - self.cctsr[layer] ** 2).sqrt()

    

    
from CorrFeatTsr_lib import visualize_cctsr, visualize_cctsr_embed, Corr_Feat_Machine, Corr_Feat_pipeline, loadimg_preprocess, loadimg_embed_preprocess
from data_loader import load_score_mat, mat_path
#%%
from torchvision.models import vgg16, alexnet
VGG = vgg16(pretrained=True).cuda()
VGG.requires_grad_(False)
VGG.eval()
#%%
datadir = r"S:\FeatTsr"
Animal = "Alfa"
MStats = loadmat(join(mat_path, Animal + "_Manif_stats.mat"), struct_as_record=False, squeeze_me=True)['Stats']
EStats = loadmat(join(mat_path, Animal + "_Evol_stats.mat"), struct_as_record=False, squeeze_me=True, chars_as_strings=True)['EStats']
Expi = 3
#%%
savedir = join(datadir, "%s_Exp%02d_E"%(Animal, Expi))
score_vect, imgfullpath_vect = load_score_mat(EStats, MStats, Expi, "Evol", wdws=[(50, 200)])
# normalize score and turn into tensor
score_M = score_vect.mean()
score_S = score_vect.std()
score_norm = (score_vect - score_M) / score_S
#%%
from sklearn.model_selection import train_test_split
score_tsr = torch.tensor(score_norm).float()
imgN_M = len(imgfullpath_vect)
train_idx, valid_idx = train_test_split(range(imgN_M), test_size=0.2, random_state=35, shuffle=True, )

imgfp_train, imgfp_valid = [imgfullpath_vect[idx] for idx in train_idx], [imgfullpath_vect[idx] for idx in valid_idx]
score_train, score_valid = score_tsr[train_idx], score_tsr[valid_idx]
trainN, validN = len(train_idx), len(valid_idx)
#%%
epocsN = 15
batchsize = 60
imgpix = 224
# Set up models and config
featNet = VGG.features
featFetcher = FactorRegr_Machine(Nfactor=10, spatial_nonneg=False, )
featFetcher.register_hooks(VGG, ["conv3_3", "conv4_3", "conv5_3"])  # "conv2_2",
featFetcher.ftL2_lambda = 2E-2
featFetcher.spL2_lambda = 2E-3
featFetcher.smooth_lambda = 2E-2
featFetcher.lr = 1E-3
featFetcher.init_model(VGG, img_dim=imgpix)
logstr = "fact10"#"Sp_nneg_mod"
# training pipeline
for epi in tqdm(range(epocsN)):
    print("Current epocs %02d"%epi)
    csr = 0
    pbar = tqdm(total=trainN)
    while csr < trainN:
        cend = min(csr + batchsize, trainN)
        input_tsr = loadimg_preprocess(imgfp_train[csr:cend], borderblur=True)  # imgpix=120, fullimgsz=224
        # input_tsr = loadimg_embed_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
        # Pool through VGG
        with torch.no_grad():
            part_tsr = featNet(input_tsr.cuda()).cpu()
        # featFetcher.update_corr_rep(scorecol_M[csr:cend])
        featFetcher.update_model(score_train[csr:cend])
        # update bar!
        pbar.update(cend-csr)
        csr = cend
    pbar.close()
    print("Epoc %02d Train summary"%epi)
    featFetcher.record_summary()
    featFetcher.clear_record()
    csr = 0
    pbar = tqdm(total=validN)
    while csr < validN:
        cend = min(csr + batchsize, validN)
        input_tsr = loadimg_preprocess(imgfp_valid[csr:cend], borderblur=True)  # imgpix=120, fullimgsz=224
        # input_tsr = loadimg_embed_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
        with torch.no_grad():
            part_tsr = featNet(input_tsr.cuda()).cpu()
        featFetcher.predict()
        featFetcher.record_score(score_valid[csr:cend])
        # update bar!
        pbar.update(cend - csr)
        csr = cend
    pbar.close()
    print("Epoc %02d Validation summary" % epi)
    featFetcher.record_summary()
    featFetcher.clear_record()
featFetcher.clear_hook()
featFetcher.visualize_weights(savestr=logstr, figdir="")


#%%

layer = "conv4_3"
for layer in featFetcher.feat_trans:
    NF = featFetcher.Nfactor
    C,H,W = featFetcher.feat_tsr_shape[layer]
    feat_vecs = featFetcher.feat_trans[layer].weight.detach().clone().cpu()
    sp_msks = featFetcher.sp_mask[layer].weight.detach().clone().cpu()
    feat_vecs = feat_vecs.reshape((NF, C))
    sp_msks = sp_msks.reshape((NF, H, W))

    plt.figure(figsize=[21,6])
    for i in range(NF):
        plt.subplot(3,NF,i+1)
        plt.imshow(sp_msks[i,:,:].squeeze().numpy())
        plt.subplot(3,NF,i+1+NF)
        plt.plot(feat_vecs[i,:].numpy(),alpha=0.5)
        plt.subplot(3,NF,i+1+2*NF)
        plt.plot(sorted(feat_vecs[i,:].numpy()),alpha=0.5)
    plt.suptitle(layer)
    plt.show()
#%%
# #%%
# featTsrs = {}
# imgN_M = len(imgfullpath_vect_M)
# csr = 0
# pbar = tqdm(total=imgN_M)
# while csr < imgN_M:
#     cend = min(csr + batchsize, imgN_M)
#     input_tsr = loadimg_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix)
#     # input_tsr = loadimg_embed_preprocess(imgfullpath_vect_M[csr:cend], imgpix=imgpix, fullimgsz=(256, 256))
#     # Pool through VGG
#     with torch.no_grad():
#         part_tsr = featNet(input_tsr.cuda()).cpu()
#     # featFetcher.update_corr_rep(scorecol_M[csr:cend])
#     for layer, tsr in featFetcher.feat_tsr.items():
#         if not layer in featTsrs:
#             featTsrs[layer] = tsr.clone().half()
#         else:
#             featTsrs[layer] = torch.cat((featTsrs[layer], tsr.clone().half()), dim=0)
#     # update bar!
#     pbar.update(cend-csr)
#     csr = cend
# pbar.close()
# featFetcher.clear_hook()