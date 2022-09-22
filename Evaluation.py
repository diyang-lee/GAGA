import numpy as np
from sklearn import linear_model
import Weighted_Samples
import math


def decision_MOSPL_LASSO(v_MOSPL,X,y,alpha):
    model = linear_model.Lasso(fit_intercept=False, normalize=False, alpha=alpha, warm_start=True, max_iter=5000,
                               tol=1e-6)
    m = np.shape(v_MOSPL)[0]
    w = []
    b = []
    for i in range(0,m):
        X_, y_ = Weighted_Samples.weighted_sample_lasso(X, y, v_MOSPL[i,:])
        model.fit(X_,y_)
        w.append(model.coef_.tolist())
        b.append(model.intercept_)
    return w, b


def decision_Lasso(X, w, b):
    """

    :param X: the sample matrix
    :param w: model parameter
    :param b:
    :return:
    """
    y = w @ X.T + b
    return y


def decision_SVM(X, w, b):
    y = w @ X.T + b
    y[y > 0] = 1
    y[y <= 0] = -1
    return y


def accuracy(y_test, y_star):
    if np.size(y_test) != np.size(y_star):
        print('error! the number of each array is not aligned!\n')
        return -1
    i = 0
    correct = 0
    while i < np.size(y_test):
        if y_test[i] == y_star[i]:
            correct += 1
        i += 1
    ac = correct / i
    return ac


def RMSE(y_test, y_star):
    if np.size(y_test) != np.size(y_star):
        print('error! the number of each array is not aligned!\n')
        return -1
    return math.sqrt(np.sum(np.square(y_test - y_star)) / np.size(y_star))


def MAE(y_test, y_star):
    if np.size(y_test) != np.size(y_star):
        print('error! the number of each array is not aligned\n')
        return -1
    return np.sum(np.abs(y_test - y_star)) / np.size(y_star)


def Lasso_score(w_sequence, X, y):
    """
    rate the algorithm based on a sequence of w
    :param y:
    :param X: sample matrix
    :param w_sequence: the w sequence w.r.t. b is presetted to be 0
    :return: matrix of the score w.r.t. w with first line of RMSE and the second line of MAE
    """
    RMSE_score = []
    MAE_score = []
    for w in w_sequence:
        y_ = decision_Lasso(X, w, 0)
        RMSE_score.append(RMSE(y,y_))
        MAE_score.append(MAE(y,y_))
    return RMSE_score, MAE_score

def SVM_score(w_sequence,X,y):
    """
    rate the algorithm based on a sequence of w
    :param w_sequence: as usual
    :param X:
    :param y:
    :return: the score sequence of accuracy
    """
    score = []
    for w in w_sequence:
        y_ = decision_SVM(X,w[1:],w[0])
        score.append(accuracy(y,y_))
    return score


def Outlier_traning(w_sequence,lam_sequence,X,y,alpha,confidence_tol):
    """
    this function removes samples with low confidence, and train the corresponding model sample
    :param confidence_tol: samples with v lower than confidence_tol will be seen as an outlier thus been removed
    :param w_sequence: the w sequence w.r.t. lambda w[0] = intercept
    :param X: sample matrix
    :param y:
    :return: the corresponding 'w_sequence' w.r.t. w_sequence
    """
    model = linear_model.Lasso(fit_intercept=False, normalize=False, alpha=alpha, warm_start=True, max_iter=5000, tol=1e-6)
    w_ = []
    for w, lam in w_sequence, lam_sequence:
        b = w[0]
        w = w[1:]
        loss_vector = np.square(y - w @ X.T - b)
        v = Weighted_Samples.linear_loss(loss_vector,lam)
        pos = np.where(v <= confidence_tol)
        X_ = np.delete(X,pos,axis=0)
        y_ = np.delete(y,pos)
        model.fit(X_,y_)
        w = model.coef_
        b = model.intercept_
        w_.append(np.vstack([b,w]))

    return w_



