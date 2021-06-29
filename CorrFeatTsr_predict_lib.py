""" Functions of Using CorrFeatTsr Models (`CorrFeatScore`) to predict neural response """
import torch
import numpy as np
from easydict import EasyDict
from os.path import join
import matplotlib.pylab as plt
#% This Section contains functions that do predictions for the images.
from tqdm import tqdm
from scipy.optimize import curve_fit
from CorrFeatTsr_lib import loadimg_preprocess
from CorrFeatTsr_visualize_lib import CorrFeatScore
def score_images(featNet, scorer, layername, imgfps, imgloader=loadimg_preprocess, batchsize=70,):
    """ Basic function to use scorers to load and score a bunch of imgfps. 
    :param featNet: a feature processing network nn.Module.
    :param scorer: CorrFeatScore
    :param layername: str, the layer you are generating the score from
    :param imgfps: a list of full paths to the images.
    :param imgloader: image loader, a function taking a list to full path as input and returns a preprocessed image
        tensor.
    :param batchsize: batch size in processing images. Usually 120 is fine with a 6gb gpu.
    :return:
        score_all: tensor of returned scores.

    :Example:
        scorer = CorrFeatScore()
        scorer.register_hooks(net, layer, netname=netname)
        scorer.register_weights({layer: DR_Wtsr})
        pred_score = score_images(featnet, scorer, layer, imgfullpath_vect, imgloader=loadimg_preprocess, batchsize=80,)
        scorer.clear_hook()
        nlfunc, popt, pcov, scaling, nlpred_score = fitnl_predscore(pred_score.numpy(), score_vect)

    """
    imgN = len(imgfps)
    csr = 0
    pbar = tqdm(total=imgN)
    score_all = []
    while csr < imgN:
        cend = min(csr + batchsize, imgN)
        input_tsr = imgloader(imgfps[csr:cend])  # imgpix=120, fullimgsz=224, borderblur=True
        with torch.no_grad():
            part_tsr = featNet(input_tsr.cuda()).cpu()
            score = scorer.corrfeat_score(layername)
        score_all.append(score.detach().clone().cpu())
        pbar.update(cend - csr)
        csr = cend
    pbar.close()
    score_all = torch.cat(tuple(score_all), dim=0)
    return score_all


def softplus(x, a, b, thr):
    """ A soft smooth version of ReLU"""
    return a * np.logaddexp(0, x - thr) + b


def fitnl_predscore(pred_score_np: np.ndarray, score_vect: np.ndarray, show=True, savedir="", savenm="", suptit=""):
    """Given a linearly predicted score and target score, fit a nonlinearity to minimize error.
    TODO: Maybe need cross fit and prediction.
    :param pred_score_np: predicted scores to be transformed. np.array
    :param score_vect: target scores. np.array
    Return: 
        nlfunc: lambda function that maps linear prediction to nonlinear prediction.
        popt, pcov: optimized parameter and their covariance. 
        scaling: scaler to multiply linear prediction before going into Softplus
        nlpred_score: Nonlinear prediction score 
        Stat: EasyDict of these entries, "cc_bef", "cc_aft", "Nimg"
    :Example
        nlfunc, popt, pcov, scaling, nlpred_score, Stat = fitnl_predscore(pred_score.numpy(), score_vect)
    """
    # first normalize scale of pred score
    scaling = 1/pred_score_np.std()*score_vect.std()
    pred_score_np_norm = pred_score_np * scaling
    popt, pcov = curve_fit(softplus, pred_score_np_norm, score_vect, \
          p0=[1, min(score_vect), np.median(pred_score_np_norm)], \
          bounds=([0, min(score_vect) - 10, min(pred_score_np_norm)-10], 
                  [np.inf, max(score_vect), max(pred_score_np_norm)]))
    print("NL parameters: amp %.1e baseline %.1e thresh %.1e" % tuple(popt))
    nlpred_score = softplus(pred_score_np_norm, *popt)
    nlfunc = lambda predicted: softplus(predicted * scaling, *popt)
    cc_bef = np.corrcoef(score_vect, pred_score_np)[0, 1]
    cc_aft = np.corrcoef(score_vect, nlpred_score)[0, 1]
    print("Correlation before nonlinearity fitting %.3f; after nonlinearity fitting %.3f"%(cc_bef, cc_aft))
    np.savez(join(savedir, "nlfit_result%s.npz"%savenm), cc_bef=cc_bef, cc_aft=cc_aft, scaling=scaling, popt=popt,
             pcov=pcov, nlpred_score=nlpred_score, obs_score=score_vect)
    Stat = EasyDict({"cc_bef": cc_bef, "cc_aft": cc_aft, "Nimg": len(score_vect)})
    if show:
        figh = plt.figure(figsize=[8, 4.5])
        plt.subplot(121)
        plt.scatter(pred_score_np, nlpred_score, alpha=0.5, label="fitting")
        plt.scatter(pred_score_np, score_vect, alpha=0.5, label="data")
        plt.xlabel("Factor Prediction")
        plt.ylabel("Original Scores")
        plt.title("Before Fitting corr %.3f"%(cc_bef))
        plt.legend()
        plt.subplot(122)
        plt.scatter(nlpred_score, score_vect, alpha=0.5)
        plt.axis("image")
        plt.xlabel("Factor Prediction + nl")
        plt.ylabel("Original Scores")
        plt.title("After Fitting corr %.3f"%(cc_aft))
        plt.suptitle(suptit+" score cmp")
        figh.savefig(join(savedir, "nlfit_vis_%s.png"%savenm))
        figh.savefig(join(savedir, "nlfit_vis_%s.pdf"%savenm))
        plt.show()
    return nlfunc, popt, pcov, scaling, nlpred_score, Stat


def visualize_prediction(pred_score, score_vect, nlpred_score=None, 
                     savedir="", savenm="dataset_pred", suptit="dataset", show=True):
    lin_only = nlpred_score is None 
    cc_bef = np.corrcoef(score_vect, pred_score)[0, 1]
    if not lin_only: 
        cc_aft = np.corrcoef(score_vect, nlpred_score)[0, 1]
    figh = plt.figure(figsize=[8, 4.5])
    plt.subplot(121)
    if not lin_only: plt.scatter(pred_score, nlpred_score, alpha=0.5, label="fitting")
    plt.scatter(pred_score, score_vect, alpha=0.5, label="data")
    plt.xlabel("Factor Prediction")
    plt.ylabel("Original Scores")
    plt.title("Before Fitting corr %.3f"%(cc_bef))
    plt.legend()
    if not lin_only: 
        plt.subplot(122)
        plt.scatter(nlpred_score, score_vect, alpha=0.5)
        plt.axis("image")
        plt.xlabel("Factor Prediction + nl")
        plt.ylabel("Original Scores")
    plt.title("After Fitting corr %.3f"%(cc_aft))
    plt.suptitle(suptit+" score cmp")
    figh.savefig(join(savedir, "nlfit_vis_%s.png"%savenm))
    figh.savefig(join(savedir, "nlfit_vis_%s.pdf"%savenm))
    if show:
        plt.show()
    

def resample_correlation(scorecol, trial=100):
    """ Compute noise ceiling for correlating with a collection of noisy data"""
    resamp_scores_col = []
    for tri in range(trial):
        resamp_scores = np.array([np.random.choice(A, len(A), replace=True).mean() for A in scorecol])
        resamp_scores_col.append(resamp_scores)
    resamp_scores_arr = np.array(resamp_scores_col)
    resamp_ccmat = np.corrcoef(resamp_scores_arr)
    resamp_ccmat += np.diag(np.nan*np.ones(trial))
    split_cc_mean = np.nanmean(resamp_ccmat)
    split_cc_std = np.nanstd(resamp_ccmat)
    return split_cc_mean, split_cc_std


def predict_fit_dataset(DR_Wtsr, imgfullpath_vect, score_vect, scorecol, net, layer, netname="", featnet=None, nlfunc=None,\
                imgloader=loadimg_preprocess, batchsize=62, show=True, figdir="", savenm="pred", suptit=""):
    """ Use the weight tensor DR_Wtsr to do a linear model over 
        DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)
    Note when the exgeneous `nlfunc` is provided, it will not do fitting, but use the exogeneous one.  

    :param imgfullpath_vect:
    :param score_vect:
    :param scorecol:
    :param net:
    :param layer:
    :param featnet:
    :param netname:
    :return:
    """
    if len(imgfullpath_vect)==0:
        return np.array([]), np.array([]), lambda x:x, EasyDict({"Nimg":0, "cc_bef":np.nan, "cc_aft":np.nan})
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    scorer.register_weights({layer: DR_Wtsr})
    pred_score = score_images(featnet, scorer, layer, imgfullpath_vect, imgloader=imgloader,
                                  batchsize=batchsize, ).numpy()
    scorer.clear_hook()
    if nlfunc is None:
        nlfunc, popt, pcov, scaling, nlpred_score, PredStat = fitnl_predscore(pred_score, score_vect,
                      savedir=figdir, savenm=savenm, suptit=suptit, show=show)
    else:
        nlpred_score = nlfunc(pred_score)
        cc_bef = np.corrcoef(score_vect, pred_score)[0, 1]
        cc_aft = np.corrcoef(score_vect, nlpred_score)[0, 1]
        PredStat = EasyDict({"cc_bef": cc_bef, "cc_aft": cc_aft, "Nimg": len(score_vect)})
        print("%s Corr before nonlinearity fitting %.3f; after nonlinearity fitting %.3f"%(savenm, cc_bef, cc_aft))
        visualize_prediction(pred_score, score_vect, nlpred_score, 
                         savedir=figdir, savenm=savenm, suptit=suptit, show=show)
    # Record stats and form population statistics
    if scorecol is not None: # compute the noise ceiling by resample from the trials. 
        corr_ceil_mean, corr_ceil_std = resample_correlation(scorecol, trial=100)
        for varnm in ["corr_ceil_mean", "corr_ceil_std"]:
            PredStat[varnm] = eval(varnm)
        PredStat.cc_aft_norm = PredStat.cc_aft / corr_ceil_mean  # prediction normalized by noise ceiling.
        PredStat.cc_bef_norm = PredStat.cc_bef / corr_ceil_mean
    return pred_score, nlpred_score, nlfunc, PredStat


def predict_dataset(DR_Wtsr, imgfullpath_vect, score_vect, scorecol, net, layer, netname="", featnet=None,nlfunc=lambda x:x, \
                imgloader=loadimg_preprocess, batchsize=62, show=True, figdir="", savenm="pred", suptit=""):
    """ Use the weight tensor DR_Wtsr to do a linear model over features in CNN net, layer, with netname.
        DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)

    :param imgfullpath_vect:
    :param score_vect:
    :param scorecol:
    :param net:
    :param layer:
    :param featnet:
    :param netname:
    :return:
    """
    if len(imgfullpath_vect)==0:
        return np.array([]), np.array([]), lambda x:x, EasyDict({"Nimg":0, "cc_bef":np.nan, "cc_aft":np.nan})
    scorer = CorrFeatScore()
    scorer.register_hooks(net, layer, netname=netname)
    scorer.register_weights({layer: DR_Wtsr})
    pred_score = score_images(featnet, scorer, layer, imgfullpath_vect, imgloader=imgloader,
                                  batchsize=batchsize, ).numpy()
    scorer.clear_hook()
    nlpred_score = nlfunc(pred_score)
    cc_bef = np.corrcoef(score_vect, pred_score)[0, 1]
    cc_aft = np.corrcoef(score_vect, nlpred_score)[0, 1]
    PredStat = EasyDict({"cc_bef": cc_bef, "cc_aft": cc_aft, "Nimg": len(score_vect)})
    print("%s Corr before nonlinearity fitting %.3f; after nonlinearity fitting %.3f"%(savenm, cc_bef, cc_aft))
    visualize_prediction(pred_score, score_vect, nlpred_score, 
                     savedir=figdir, savenm=savenm, suptit=suptit, show=show)
    # Record stats and form population statistics
    if scorecol is not None: # compute the noise ceiling by resample from the trials. 
        corr_ceil_mean, corr_ceil_std = resample_correlation(scorecol, trial=100)
        for varnm in ["corr_ceil_mean", "corr_ceil_std"]:
            PredStat[varnm] = eval(varnm)
        PredStat.cc_aft_norm = PredStat.cc_aft / corr_ceil_mean  # prediction normalized by noise ceiling.
        PredStat.cc_bef_norm = PredStat.cc_bef / corr_ceil_mean
    return pred_score, nlpred_score, nlfunc, PredStat


def nlfit_merged_dataset(pred_score_col:list, score_vect_col:list, scorecol_col:list, figdir="", savenm="pred_all", suptit="",
                         show=True, nlfunc=None):
    """ Use the weight tensor DR_Wtsr to do a linear model over
        DR_Wtsr = pad_factor_prod(Hmaps, ccfactor, bdr=bdr)

    :param imgfullpath_vect:
    :param score_vect:
    :param scorecol:
    :param net:
    :param layer:
    :param featnet:
    :param netname:
    :return:
    """
    pred_score_all = np.concatenate(tuple(pred_score_col), axis=0)
    score_vect_all = np.concatenate(tuple(score_vect_col), axis=0)
    if nlfunc is None:
        nlfunc, popt, pcov, scaling, nlpred_score_all, PredStat = fitnl_predscore(pred_score_all, score_vect_all,
                      savedir=figdir, savenm=savenm, suptit=suptit, show=show)
    else:
        nlpred_score_all = nlfunc(pred_score_all)
        cc_bef = np.corrcoef(score_vect_all, pred_score_all)[0, 1]
        cc_aft = np.corrcoef(score_vect_all, nlpred_score_all)[0, 1]
        PredStat = EasyDict({"cc_bef": cc_bef, "cc_aft": cc_aft, "Nimg": len(score_vect_all)})
        print("%s Corr before nonlinearity fitting %.3f; after nonlinearity fitting %.3f"%(savenm, cc_bef, cc_aft))
        visualize_prediction(pred_score_all, score_vect_all, nlpred_score_all, 
                         savedir=figdir, savenm=savenm, suptit=suptit, show=show)
    # Record stats and form population statistics
    if scorecol_col is not None:
        scorecol_all = [scores  for scorecol in scorecol_col for scores in scorecol]
        corr_ceil_mean, corr_ceil_std = resample_correlation(scorecol_all, trial=100)
        for varnm in ["corr_ceil_mean", "corr_ceil_std"]:
            PredStat[varnm] = eval(varnm)
        PredStat.cc_aft_norm = PredStat.cc_aft / corr_ceil_mean  # prediction normalized by noise ceiling.
        PredStat.cc_bef_norm = PredStat.cc_bef / corr_ceil_mean
    return pred_score_all, nlpred_score_all, score_vect_all, nlfunc, PredStat