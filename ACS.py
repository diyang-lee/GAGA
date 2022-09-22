import numpy as np
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import Weighted_Samples
import operator
from sklearn.linear_model import LogisticRegression

def ACS_Lasso(X, y, v, lam_sequence, alpha=1.0, mode=1, gamma=1.0):
    mod = Lasso(fit_intercept=False, normalize=False, alpha=alpha, warm_start=False, max_iter=8000, tol=1e-10)
    w_path = []
    b_path = []
    m = np.shape(X)[0]
    turing_point = []
    break_point = []
    pos_ = np.ones(m)
    pos =  np.zeros(m)
    m, n = np.shape(X)
    nums = -1
    nums_ = 0
    w_old = 0
    pos_ = np.zeros(m)
    pos = np.ones(m)
    for lam in lam_sequence:
        X_, y_ = Weighted_Samples.weighted_sample_lasso(X, y, v)
        mod.fit(X_, y_)
        w = mod.coef_
        w_path.append(w.tolist())
        b = mod.intercept_
        b_path.append(b)
        loss_vector = np.square(y - w @ X.T - b)
        if mode == 1:
            v = Weighted_Samples.linear_loss(loss_vector, lam)
        elif mode == 2:
            v = Weighted_Samples.mixture_loss(loss_vector,lam,gamma)
        # nums = np.size(v[v!=0])
        pos = np.where(abs(v-0)>1e-3)
        if np.size(pos[0]) != np.size(pos_[0]):
            turing_point.append(lam)
        elif operator.eq(np.sort(pos_[0]), np.sort(pos[0])).any() == False:
            turing_point.append(lam)
        elif np.sum(np.abs(w-w_old)) > 0.1:
            break_point.append(lam)
        pos_ = pos
        w_old = w.copy()
    t = [turing_point,break_point]
    return w_path,b_path,t


def ACS_svm(X, y, v:np.matrix, lam_sequence, C=1.0, mode=1, gamma=1.0):
    """

    :param gamma: the para in mixture loss
    :param mod:
    :param X:
    :param y:
    :param v: please note that this v has to be a np.matrix!
    :param lam_sequence:
    :param C:
    :return:
    """
    mod = SVC(kernel='linear',tol = 1e-5,cache_size=2048,)
    X = np.matrix(X)
    m,n = np.shape(X)
    w_path = []
    b_path = []
    for lam in lam_sequence:
        mod.fit(X, y, sample_weight=v[0]) # it seems that linar SVC only take matrix X as an input
        w = mod.coef_
        b = mod.intercept_
        w_path.append(w[0].tolist())
        b_path.append(b)
        loss_vector = 1 - (np.multiply(y , (w @ X.T + b)))[0]
        loss_vector = loss_vector.A
        loss_vector[loss_vector < 0] = 0
        if mode == 1:
            v = Weighted_Samples.linear_loss(loss_vector, lam)
        elif mode == 2:
            v = Weighted_Samples.mixture_loss(loss_vector, lam, gamma)
    return w_path, b_path


def ACS_logistic(X, y, v:np.matrix, lam_sequence, C=1.0, mode=1, gamma=1.0):
    """

        :param gamma: the para in mixture loss
        :param mod:
        :param X:
        :param y:
        :param v: please note that this v has to be a np.matrix!
        :param lam_sequence:
        :param C:
        :return:
        """
    mod = LogisticRegression(C=C, max_iter=2000, tol=1e-8, n_jobs=-1, warm_start=True)
    X = np.matrix(X)
    m, n = np.shape(X)
    w_path = []
    b_path = []
    turing_point = []
    break_point = []
    m, n = np.shape(X)
    w_old = 0
    pos_ = np.zeros(m)

    for lam in lam_sequence:
        v = np.matrix(v)
        mod.fit(X, y, sample_weight=v[0].tolist()[0])  # it seems that linar SVC only take matrix X as an input
        w = mod.coef_
        b = mod.intercept_

        loss_vector = C * np.log(1 + np.e ** (-y * mod.decision_function(X)))
        loss_vector[loss_vector < 0] = 0
        if mode == 1:
            v = Weighted_Samples.linear_loss(loss_vector, lam)
        elif mode == 2:
            v = Weighted_Samples.mixture_loss(loss_vector, lam, gamma)
        w_path.append(w[0].tolist())
        b_path.append(b)
        pos = np.where(abs(v - 0) > 1e-3)
        if np.size(pos[0]) != np.size(pos_[0]):
            turing_point.append(lam)
        elif operator.eq(np.sort(pos_[0]), np.sort(pos[0])).any() == False:
            turing_point.append(lam)
        elif np.sum(np.abs(w - w_old)) > 0.1:
            break_point.append(lam)
        pos_ = pos
        w_old = w.copy()
    t = [turing_point, break_point]
    return w_path, b_path, t

