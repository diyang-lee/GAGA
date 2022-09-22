# the aim of this text is to calculate the weighted sample based on given
import numpy as np

def linear_loss(loss_vector,lam):
    """
    calculate corresponding v
    :param l: the np.array for the loss of each function
    :param lam: current learning pace
    :return: corresponding v
    """
    lenth = np.size(loss_vector)
    v = np.ones(lenth) - loss_vector / lam
    v[v < 0] = 0
    return v


    
def mixture_loss(loss_vector,lam,gamma):
    length = np.size(loss_vector)
    v = gamma * (np.ones(length) / np.sqrt(loss_vector) - np.ones(length) / lam)
    v[v < 0] = 0
    v[v > 1] = 1
    return v



def weighted_sample_lasso(X, y, v):
    X_ = np.diag(np.sqrt(v)) @ X
    y_ = y * np.sqrt(v)
    return X_, y_


def lam_sequence(lam_start, lam_end, mode):
    """
    this function is for generating lam_sequence for ACS
    :param lam_start: the starting point of lambda
    :param lam_end: the end
    :param mode: 1: sparce ACS , 2,3:ground truth ACS
    :return: the lam_sequence
    """
    if mode == 0:
        return np.linspace(lam_start, lam_end, 35)
    if mode == 1:
        return np.arange(lam_start, lam_end, 0.60)
    if mode == 2:
        return np.arange(lam_start, lam_end, 0.0001)
    if mode == 3:
        return np.linspace(lam_start, lam_end, 600)

