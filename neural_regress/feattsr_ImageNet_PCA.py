"""
Use ImageNet validation set to obtain features to train PCA and sparse random projection.
Then use PCA and sparse random projection to do neuron prediction.
"""

from NN_PC_visualize.NN_PC_lib import \
    create_imagenet_valid_dataset, Dataset, DataLoader
from neural_regress.regress_lib import calc_features, calc_reduce_features, featureFetcher, tqdm, torch, np
from featvis_lib import load_featnet
from torch.utils.data import Subset, SubsetRandomSampler
def calc_features_in_dataset(dataset, net, featlayer, idx_range=None,
                  batch_size=40, workers=6, ):
    """
    Calculate features for a set of images.
    :param score_vect: numpy vector of scores,
            if None, then it's default to be zeros. ImagePathDataset will handle None scores.
    :param imgfullpath_vect: a list full path to images
    :param net: net to extract features from
    :param featlayer: layer to extract features from
    :param batch_size: batch size for DataLoader
    :param workers: number of workers for DataLoader
    :param img_dim: image dimensions
    :return:
    """
    if idx_range is None:
        imgloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    else:
        imgloader = DataLoader(Subset(ImageNet_dataset, idx_range), batch_size=batch_size,
                               shuffle=False, num_workers=workers,)

    featFetcher = featureFetcher(net, print_module=False)
    featFetcher.record(featlayer, )
    feattsr_col = []
    for i, (imgtsr, _) in tqdm(enumerate(imgloader)):
        with torch.no_grad():
            net(imgtsr.cuda())
        feattsr = featFetcher[featlayer]
        feattsr_col.append(feattsr.cpu().numpy())

    feattsr_all = np.concatenate(feattsr_col, axis=0)
    print("feature tensor shape", feattsr_all.shape)
    del feattsr_col, featFetcher
    return feattsr_all
#%%
ImageNet_dataset = create_imagenet_valid_dataset(imgpix=227, )
featnet, net = load_featnet("resnet50_linf8")
# featlayer = ".layer3.Bottleneck5"
# featlayer = ".layer4.Bottleneck2"
featlayer = ".layer2.Bottleneck3"
feattsr_all = calc_features_in_dataset(ImageNet_dataset, net, featlayer, idx_range=range(0, 2000),
                  batch_size=40, workers=6, )
#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=1000)
pca.fit(feattsr_all.reshape(feattsr_all.shape[0], -1))
#%%
from sklearn.random_projection import johnson_lindenstrauss_min_dim, \
            SparseRandomProjection, GaussianRandomProjection
n_components_ = johnson_lindenstrauss_min_dim(n_samples=3000, eps=0.1) # len(score_vect)
srp = SparseRandomProjection(n_components=n_components_, random_state=0)
srp.fit(np.zeros((1, np.prod(feattsr_all.shape[1:]))))
#%%
from os.path import join
import pickle as pkl
saveroot = r"E:\OneDrive - Harvard University\Manifold_NeuralRegress"
pkl.dump({"srp": srp, "pca": pca},
         open(join(saveroot, f"{featlayer}_regress_Xtransforms.pkl"), "wb"))
