"""
Make a PyTorch VGG with the same weights as matlab to match the interpretations there.
"""
#%%
from torchvision import models
import torch
from scipy.io import loadmat
from skimage.io import imread, imread_collection
from os.path import join
import numpy as np
from collections import OrderedDict
# system("subst S: E:\Network_Data_Sync")
def getMatVGG():
    VGG = models.vgg16(pretrained=True)
    vggmatW = loadmat(r"N:\vgg16W.mat", struct_as_record=False, squeeze_me=True)["vggmatW"]
    state_dict = VGG.state_dict()
    for paramname, matpname in zip(state_dict.keys(), vggmatW._fieldnames):
        matweight = vggmatW.__getattribute__(matpname)
        trcweight = state_dict[paramname]
        trcmatW = torch.from_numpy(matweight)
        if len(trcweight.shape) == 4: # conv2 layer
            trcmatW = trcmatW.permute(3, 2, 0, 1)
            # matlab weight shape [FilterSize(1),FilterSize(2),NumChannels,NumFilters]
            # torch weight shape `[out_channels(NumFilters), in_channels(NumChannels), kernel_size ]`
        elif len(trcweight.shape) == 2: # fc layer matmul weight is 2d
            pass
        assert trcmatW.shape == trcweight.shape
        state_dict[paramname] = trcmatW
        print(paramname, matpname, trcweight.shape, matweight.shape, )
    VGG.load_state_dict(state_dict)
    VGG.eval()
    VGG.requires_grad_(False)
    return VGG

def Matforward(VGG, tsr):
    RGBmean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape([1, 3, 1, 1])
    feattsr = VGG.avgpool(VGG.features(tsr - RGBmean))
    outcode = VGG.classifier(torch.flatten(feattsr, 1))
    return outcode

def lucent_layernames(net, prefix=[]):
    """ Return the layername and str representation of the layer """
    layernames = OrderedDict()
    def hook_layernames(net, prefix=[]):
        """Recursive function to return the layer name"""
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                layernames["_".join(prefix+[name])] = layer.__repr__()
                hook_layernames(layer, prefix=prefix+[name])
    hook_layernames(net, prefix)
    return layernames