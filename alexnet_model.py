"""Credit to Will Xiao for tranlating this AlexNet model"""
from typing import Optional, Any, Tuple, Union
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn

__all__ = ["MatLabAlexNet", "matlab_alexnet", "load_alexnet_matlab"]


class AlexNetFlat5(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.flatten(torch.transpose(x, 2, 3), start_dim=1)


class MatLabAlexNet(nn.Module):
    def __init__(
            self,
            output_layers: Optional[Union[str, Tuple[str]]] = None,
            inplace: bool = False,  # use False if tapping model acts, True for training to save memory
    ) -> None:
        super().__init__()
        cross_chan_norm_params = dict(size=5, alpha=0.0001, beta=0.75, k=1.0)
        if isinstance(output_layers, str):
            output_layers = (output_layers,)
        self.output_layers = output_layers
        self.inplace = inplace

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0)
        self.relu1 = nn.ReLU(inplace=inplace)
        self.norm1 = nn.LocalResponseNorm(**cross_chan_norm_params)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2)
        self.relu2 = nn.ReLU(inplace=inplace)
        self.norm2 = nn.LocalResponseNorm(**cross_chan_norm_params)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=inplace)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2)
        self.relu4 = nn.ReLU(inplace=inplace)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2)
        self.relu5 = nn.ReLU(inplace=inplace)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flat5 = AlexNetFlat5()
        self.fc6 = nn.Linear(9216, 4096)
        self.relu6 = nn.ReLU(inplace=inplace)
        self.drop6 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=inplace)
        self.drop7 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, 1000)
        self.prob = nn.Softmax(dim=1)
        self.layer_names = (
            'conv1', 'relu1', 'norm1', 'pool1',
            'conv2', 'relu2', 'norm2', 'pool2',
            'conv3', 'relu3', 'conv4', 'relu4',
            'conv5', 'relu5', 'pool5', 'flat5',
            'fc6', 'relu6', 'drop6',
            'fc7', 'relu7', 'drop7',
            'fc8', 'prob'
        )
        self.eval()

    def forward(
            self,
            x: torch.Tensor,
            output_layers: Optional[Union[str, Tuple[str]]] = None
    ) -> Union[torch.Tensor, dict]:
        if output_layers is None:
            output_layers = self.output_layers
        if isinstance(output_layers, str):
            output_layers = (output_layers,)
        if output_layers is not None:
            output_layers = list(output_layers)
        outputs = {}
        for name in self.layer_names:
            x = getattr(self, name)(x)
            if output_layers is not None and name in output_layers:
                outputs[name] = x.copy() if self.inplace else x
                output_layers.remove(name)
                if not len(output_layers):
                    return outputs
        return x


def matlab_alexnet(
        weights_path: Optional[Path] = None,
        **kwargs: Any
) -> MatLabAlexNet:
    r"""AlexNet model architecture as implemented by MATLAB.
    Please refer to the documentation at
    <https://www.mathworks.com/help/deeplearning/ref/alexnet.html>.

    The pretrained model in MATLAB is ported to PyTorch
    for niche projects that require access to this exact model.
    The numerical agreement between the ported and the source model
    is verified to be accurate to 1e-9 +/- 2e-8 (mean +/- std)
    mean absolute deviation, or a correlation of 0.9999999999995774,
    across 100 images (ILSVRC 12 validation set) x 1000 probability
    values (i.e., the final network output).

    Note that this model (and the pretrained weights) expect different
    input sizes, preprocessing, etc. from the torchvision model.
    Here are some differences w.r.t. the torchvision model
    - the mean values for normalization is per-pixel (i.e., shape 227 x 227 x 3)
    - there is no adaptive pooling between pool5/fc6, so input size must be 227 x 227 x 3
    - there is no dropout between pool5/fc6 but an added one between relu7/fc8
    - num chans in conv1 ia 96 instead of 64; padding 0 instead of 2
    - num i/o chans in conv2 is 96/256 instead of 64/192; has 2 groups instead of 1
    - num in chans in conv3 is 256 instead of 192
    - num out chans in conv4 is 384 instead of 256; has 2 groups instead of 1
    - num in chans in conv5 is 384 instead of 256; has 2 groups instead of 1
    - num classes is fixed at 1000 (can relax at cost of inability to use pretrained weights)
    - final softmax layer is built-in

    To preprocess an image for input to the network, please
    use the `'preprocessing_mean'` values saved together with the weights.
    A minimal working example is:
    ```from PIL import Image
    from torchvision import transforms

    preproc_ = transforms.Compose([
        transforms.Resize(227),
        transforms.CenterCrop(227)])
    im = Image.open('TEST_IMAGE.JPEG')
    tim = np.array(preproc_(im))
    tim = tim.astype(np.float32).transpose(2,0,1)    # dim order: (C x H x W)
    tim -= preproc_mean
    tim = torch.Tensor(tim)
    prob = net(tim[None,...])
    ```

    Args:
        weights_path (Path): path to a file that can be loaded with
            `state_dict = torch.load(weights_path)['model']`,
    """
    model = MatLabAlexNet(**kwargs)
    if weights_path is not None:
        state_dict = torch.load(weights_path)['model']
        model.load_state_dict(state_dict)
    return model


def load_alexnet_matlab():
    """Load the pretrained weights from the MATLAB implementation."""
    model = matlab_alexnet(r"E:\Cluster_Backup\torch\matlab_alexnet.pt")
    model.eval().cuda()
    model.requires_grad_(False)
    return model