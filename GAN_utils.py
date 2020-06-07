#%%
# import torch
# torch.save(G, r"E:\Monkey_Data\Generator_DB_Windows\nets\upconv\fc6\fc6GAN.pt")
# G = torch.load(r"E:\Monkey_Data\Generator_DB_Windows\nets\upconv\fc6\fc6GAN.pt")
# This is no use...since you still need network definition to use this
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import os
from os.path import join
from sys import platform
load_urls = False
if platform == "linux":  # CHPC cluster
    homedir = os.path.expanduser('~')
    netsdir = os.path.join(homedir, 'Generate_DB/nets')
    load_urls = True
    # ckpt_path = {"vgg16": "/scratch/binxu/torch/vgg16-397923af.pth"}
else:
    if os.environ['COMPUTERNAME'] == 'DESKTOP-9DDE2RH':  # PonceLab-Desktop 3
        homedir = "D:/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'PONCELAB-ML2C':  # PonceLab-Desktop Victoria
        homedir = r"C:\Users\ponce\Documents\Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-MENSD6S':  # Home_WorkStation
        homedir = "E:/Monkey_Data/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    elif os.environ['COMPUTERNAME'] == 'DESKTOP-9LH02U9':  # Home_WorkStation Victoria
        homedir = "C:/Users/zhanq/OneDrive - Washington University in St. Louis/Generator_DB_Windows"
        netsdir = os.path.join(homedir, 'nets')
    else:
        load_urls = True
        homedir = os.path.expanduser('~')
        netsdir = os.path.join(homedir, 'Documents/nets')

model_urls = {"pool5" : "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145337&authkey=AFaUAgeoIg0WtmA",
    "fc6": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145339&authkey=AC2rQMt7Obr0Ba4",
    "fc7": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145338&authkey=AJ0R-daUAVYjQIw",
    "fc8": "https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145340&authkey=AKIfNk7s5MGrRkU"}

def load_statedict_from_online(name="fc6"):
    torchhome = torch.hub._get_torch_home()
    ckpthome = join(torchhome, "checkpoints")
    os.makedirs(ckpthome, exist_ok=True)
    filepath = join(ckpthome, "upconvGAN_%s.pt"%name)
    if os.path.exists(filepath):
        torch.hub.download_url_to_file(model_urls[name], filepath, hash_prefix=None,
                                   progress=True)
    SD = torch.load(filepath)
    return SD

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)

RGB_mean = torch.tensor([123.0, 117.0, 104.0])
RGB_mean = torch.reshape(RGB_mean, (1, 3, 1, 1))

class upconvGAN(nn.Module):
    def __init__(self, name="fc6", pretrained=True):
        super(upconvGAN, self).__init__()
        savepath = {"fc6": join(netsdir, r"upconv/fc6/generator_state_dict.pt"),
                    "fc7": join(netsdir, r"upconv/fc7/generator_state_dict.pt"),
                    "fc8": join(netsdir, r"upconv/fc8/generator_state_dict.pt"),
                    "pool5": join(netsdir, r"upconv/pool5/generator_state_dict.pt")}
        self.name = name
        if name == "fc6" or name == "fc7":
            self.G = nn.Sequential(OrderedDict([
        ('defc7', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('defc6', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('defc5', nn.Linear(in_features=4096, out_features=4096, bias=True)),
        ('relu_defc5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('reshape', View((-1, 256, 4, 4))),
        ('deconv5', nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
            ]))
            self.codelen = self.G[0].in_features
        elif name == "fc8":
            self.G = nn.Sequential(OrderedDict([
  ("defc7", nn.Linear(in_features=1000, out_features=4096, bias=True)),
  ("relu_defc7", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("defc6", nn.Linear(in_features=4096, out_features=4096, bias=True)),
  ("relu_defc6", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("defc5", nn.Linear(in_features=4096, out_features=4096, bias=True)),
  ("relu_defc5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("reshape", View((-1, 256, 4, 4))),
  ("deconv5", nn.ConvTranspose2d(256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv5", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("conv5_1", nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
  ("relu_conv5_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv4", nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv4", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("conv4_1", nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
  ("relu_conv4_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv3", nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("conv3_1", nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
  ("relu_conv3_1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv2", nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv1", nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ("relu_deconv1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
  ("deconv0", nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
  ]))
            self.codelen = self.G[0].in_features
        elif name == "pool5":
            self.G = nn.Sequential(OrderedDict([
        ('Rconv6', nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu6', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('Rconv7', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu7', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('Rconv8', nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1))),
        ('Rrelu8', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv5', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))), ]))
            self.codelen = self.G[0].in_channels
        # load pre-trained weight
        if pretrained:
            if load_urls:
                SDnew = load_statedict_from_online(name)
            else:
                SD = torch.load(savepath[name])
                SDnew = OrderedDict()
                for name, W in SD.items():
                    name = name.replace(".1.", ".")
                    SDnew[name] = W
            self.G.load_state_dict(SDnew)

    def forward(self, x):
        return self.G(x)[:, [2, 1, 0], :, :]

    def visualize(self, x, scale=1.0):
        raw = self.G(x)[:, [2, 1, 0], :, :]
        return torch.clamp(raw + RGB_mean.to(raw.device), 0, 255.0) / 255.0 * scale

# # layer name translation
# # "defc7.weight", "defc7.bias", "defc6.weight", "defc6.bias", "defc5.weight", "defc5.bias".
# # "defc7.1.weight", "defc7.1.bias", "defc6.1.weight", "defc6.1.bias", "defc5.1.weight", "defc5.1.bias".
# SD = G.state_dict()
# SDnew = OrderedDict()
# for name, W in SD.items():
#     name = name.replace(".1.", ".")
#     SDnew[name] = W
# UCG.G.load_state_dict(SDnew)
#%% The first time to run this you need these modules
if __name__ == "__main__":
    import sys
    import matplotlib.pylab as plt
    sys.path.append(r"E:\Github_Projects\Visual_Neuro_InSilico_Exp")
    from torch_net_utils import load_generator, visualize
    G = load_generator(GAN="fc6")
    UCG = upconvGAN("fc6")
    #%%
    def test_consisitency(G, UCG):#_
        code = torch.randn((1, UCG.codelen))
        # network outputs are the same.
        assert torch.allclose(UCG(code), G(code)['deconv0'][:, [2, 1, 0], :, :])
        # visualization function is the same
        imgnew = UCG.visualize(code).permute([2, 3, 1, 0]).squeeze()
        imgorig = visualize(G, code.numpy(), mode="cpu")
        assert torch.allclose(imgnew, imgorig)
        plt.figure(figsize=[6,3])
        plt.subplot(121)
        plt.imshow(imgnew.detach())
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(imgorig.detach())
        plt.axis('off')
        plt.show()
    test_consisitency(G, UCG)
    #%%
    G = load_generator(GAN="fc7")
    UCG = upconvGAN("fc7")
    test_consisitency(G, UCG)
    #%%
    # pool5 GAN
    def test_FCconsisitency(G, UCG):#_
        code = torch.randn((1, UCG.codelen, 6, 6))
        # network outputs are the same.
        assert torch.allclose(UCG(code), G(code)['generated'][:, [2, 1, 0], :, :])
        # visualization function is the same
        imgnew = UCG.visualize(code).permute([2, 3, 1, 0]).squeeze()
        imgorig = G(code)['generated'][:, [2, 1, 0], :, :]
        imgorig = torch.clamp(imgorig + RGB_mean, 0, 255.0) / 255.0
        imgorig = imgorig.permute([2, 3, 1, 0]).squeeze()
        # imgorig = visualize(G, code.numpy(), mode="cpu")
        # assert torch.allclose(imgnew, imgorig)
        plt.figure(figsize=[6,3])
        plt.subplot(121)
        plt.imshow(imgnew.detach())
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(imgorig.detach())
        plt.axis('off')
        plt.show()
    G = load_generator(GAN="pool5")
    UCG = upconvGAN("pool5")
    test_FCconsisitency(G, UCG)