import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, ToPILImage, \
    Normalize, Compose, Resize
from imageio import imread, imsave

class ImagePathDataset(Dataset):
    """
    Modeling Matlab ImageDatastore, using a set of image paths to create the dataset.
    TODO: Support image data without labels.
    """
    def __init__(self, imgfp_vect, scores, img_dim=(227, 227), transform=None):
        self.imgfps = imgfp_vect
        if scores is None:
            self.scores = torch.tensor([0.0] * len(imgfp_vect))
        else:
            self.scores = torch.tensor(scores)
        # self.img_dim = img_dim
        if transform is None:
            self.transform = Compose([ToTensor(),
                                      Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                                      Resize(img_dim)])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.imgfps)

    def __getitem__(self, idx):
        img_path = self.imgfps[idx]
        img = imread(img_path)
        imgtsr = self.transform(img)
        score = self.scores[idx]
        return imgtsr, score

# ImageNet Validation Dataset
