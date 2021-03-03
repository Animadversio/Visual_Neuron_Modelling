
import sys
from os.path import join
import torch
sys.path.append(r"D:\Github\pytorch-caffe")
from caffenet import *  # Pytorch-caffe converter
#%%
pixobj_dir = r"pixelobjectness"
# protofile = r"D:\Generator_DB_Windows\nets\upconv\fc6\generator.prototxt"
weightfile = join(pixobj_dir, r'pixel_objectness.caffemodel')  # 'resnet50/resnet50.caffemodel'
save_path = join(pixobj_dir, r"pixel_objectness.pt")
protofile = join(pixobj_dir, r"pixel_objectness.prototxt")
net = CaffeNet(protofile, width=512, height=512, channels=3)
print(net)
#%%
net.load_weights(weightfile)
#%%
torch.save(net.state_dict(), save_path)
#%%
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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
            ("conv5_1", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))),
            ("relu5_1", nn.ReLU(inplace=True)),
            ("conv5_2", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))),
            ("relu5_2", nn.ReLU(inplace=True)),
            ("conv5_3", nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))),
            ("relu5_3", nn.ReLU(inplace=True)),
            ("pool5", nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=True)),
            ("pool5a", nn.AvgPool2d(kernel_size=3, stride=1, padding=1)),
            ("fc6", nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12))),
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
            return fullObjmap, Objmap
        else:
            return Objmap

BGRmean = torch.Tensor([104.008, 116.669, 122.675]).reshape(1,3,1,1)
PNet = PixObjectiveNet()
PNet.eval().cuda()
PNet.requires_grad_(False)
#%%
from skimage.io import imread
import matplotlib.pylab as plt
imgnm = r"images\block088_thread000_gen_gen087_003518.JPG"
 #   r"images\block079_thread000_gen_gen078_003152.JPG"
img = imread(join(pixobj_dir, imgnm))
imgtsr = torch.from_numpy(img).float().permute([2,0,1]).unsqueeze(0)
objmap = PNet(imgtsr.cuda()).cpu()
#%%
plt.figure(figsize=[4,8])
plt.subplot(3,1,1)
plt.imshow(img)
plt.axis("off")
plt.subplot(3,1,2)
plt.imshow(objmap[0,0,:,:].squeeze().detach().numpy())
# plt.colorbar()
plt.axis("off")
plt.subplot(3,1,3)
plt.imshow(objmap[0,1,:,:].squeeze().detach().numpy())
# plt.colorbar()
plt.axis("off")
plt.show()