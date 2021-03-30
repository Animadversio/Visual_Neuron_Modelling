"""PyTorch Definition of Pixel-objectness model"""
import sys
from os.path import join
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
pixobj_dir = r"pixelobjectness"
save_path = join(pixobj_dir, r"pixel_objectness.pt")
#%% Only need to run once at first to translate the project.
def translate_caffe2torch():
    """Function to translate the Caffe Model weight into Pytorch and save in `save_path`"""
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
            fullObjmap = F.interpolate(Objmap, size=[image.shape[2], image.shape[3]], mode='bilinear')
            return fullObjmap
        else:
            return Objmap

BGRmean = torch.Tensor([104.008, 116.669, 122.675]).reshape(1,3,1,1) # constant


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
    plt.savefig(join(figdir, "%s_pixobjmap.png" % (savenm,)))
    plt.savefig(join(figdir, "%s_pixobjmap.pdf" % (savenm,)))
    plt.show()
    return imgmskblend, figh

if __name__=="__main__":
    PNet = PixObjectiveNet()
    PNet.eval().cuda()
    PNet.requires_grad_(False)

    from skimage.io import imread
    from kornia.filters import gaussian_blur2d

    imgnm = join(pixobj_dir, r"images\block088_thread000_gen_gen087_003518.JPG")
    #   r"images\block079_thread000_gen_gen078_003152.JPG"
    img = imread(imgnm)
    imgtsr = torch.from_numpy(img).float().permute([2,0,1]).unsqueeze(0)
    imgtsr_pp = gaussian_blur2d(imgtsr, (5,5), (3,3))
    objmap = PNet(imgtsr_pp.cuda(), fullmap=True).cpu()
    objmsk = (objmap[:, 0, :, :] < objmap[:, 1, :, :]).numpy()
    probmap = objmap[:, 1, :, :].numpy()
    #%%
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
