
import numpy as np
import time
import matplotlib.pyplot as plt
from numpy.linalg import norm, pinv, det, inv
from numpy.random import randn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from scipy.integrate import solve_ivp
from sklearn import linear_model
import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from scipy.integrate import odeint
from sklearn.linear_model import LogisticRegression

gamma=2

def v_star_mix(l,lam):
    v=np.ones(l.shape)
    M=(l>(lam*gamma/(lam+gamma))**2) * (l<lam**2)
    v[M]=gamma*(1./np.sqrt(l[M])-1./lam)
    v[l >= lam**2]=0
    return v


def v_star_linear(l,lam):
    v=-l/lam+1
    v[np.where(v<0)]=0
    return v


def GAGA_Lasso_linear(X, y, w, up, low):  # w is initial value
    lam = low
    loss = np.square(X @ w - y)
    sp = []  # solution paths
    lams = []  # coresp lam
    nPoint = []  # num of points
    v = np.sqrt(v_star_linear(loss, lam))  # square loss l_i
    if np.sum(v > 0) < 5:  # make start stable
        print("====init wrong====")
    clf = linear_model.Lasso(
        alpha=0.006,
        fit_intercept=False, warm_start=1,
        max_iter=2000,
        tol=1e-6,
        random_state=666)
    w_pre = w.copy() + 1  # init for while loop
    while np.linalg.norm(w - w_pre) > 1e-6:
        w_pre = w.copy()
        loss = np.square(X @ w - y)
        v = np.sqrt(v_star_linear(loss, lam))
        clf.fit(np.diag(v) @ X, v * y)
        w = clf.coef_
    # w0 must be right before training
    ds = .1
    while lam < up:
        w_pre = w.copy() + 1  # init for while loop
        while np.linalg.norm(w - w_pre) > 1e-6:  # converge for acs
            w_pre = w.copy()
            loss = np.square(X @ w - y)
            v = np.sqrt(v_star_linear(loss, lam))
            clf.fit(np.diag(v) @ X, v * y)
            w = clf.coef_
        lams.append(lam)
        sp.append(w.flatten())
        nPoint.append(np.sum(v > 0))

        lam += ds

    return w, sp, lams, nPoint


def GAGA_Lasso_mixture(X, y, w, up, low):  # w is initial value
    lam = low
    loss = np.square(X @ w - y)
    sp = []  # solution paths
    lams = []  # coresp lam
    nPoint = []  # num of points
    v = np.sqrt(v_star_mix(loss, lam))  # square loss l_i
    if np.sum(v > 0) < 5:  # make start stable
        print("====init wrong====")
    clf = linear_model.Lasso(
        alpha=0.003,
        fit_intercept=False, warm_start=1,
        max_iter=2000,
        tol=1e-6,
        random_state=666)
    w_pre = w.copy() + 1  # init for while loop
    while np.linalg.norm(w - w_pre) > 1e-6:
        w_pre = w.copy()
        loss = np.square(X @ w - y)
        v = np.sqrt(v_star_mix(loss, lam))
        clf.fit(np.diag(v) @ X, v * y)
        w = clf.coef_
    # w0 must be right before training
    ds = .1
    while lam < up:
        w_pre = w.copy() + 1  # init for while loop
        while np.linalg.norm(w - w_pre) > 1e-6:  # converge for acs
            w_pre = w.copy()
            loss = np.square(X @ w - y)
            v = np.sqrt(v_star_mix(loss, lam))
            clf.fit(np.diag(v) @ X, v * y)
            w = clf.coef_
        lams.append(lam)
        sp.append(w.flatten())
        nPoint.append(np.sum(v > 0))

        lam += ds

    return w, sp, lams, nPoint

def GAGA_svm_linear(X, y, v, lam_sequence, C=1, tol=1e-10, max_ite=600, tol_svm=1e-10):
    m = X.shape[0]
    n = X.shape[1]
    gram = rbf_kernel(X, gamma=1 / (n * X.var())) * linear_kernel(y.reshape(m, 1))
    solution_path = []
    i = 0
    length = len(lam_sequence)

    svm = SVC(C=C, gamma=1 / (n * X.var()), tol=tol_svm)
    d_index = np.where(v == 0)[0]
    index = np.where(v != 0)[0]
    X_train = np.delete(X, d_index, 0)
    y_train = np.delete(y, d_index, 0)
    svm.fit(X_train, y_train, v[index])
    lam = lam_sequence[i]
    loss = C * np.maximum(np.zeros(m), 1 - y * svm.decision_function(X))
    v = np.maximum(np.zeros(m), 1 - loss / lam)
    while i < length:
        # acs找partial optimum
        ite = 1
        while True:
            d_index = np.where(v == 0)[0]
            index = np.where(v != 0)[0]
            X_train = np.delete(X, d_index, 0)
            y_train = np.delete(y, d_index, 0)
            svm.fit(X_train, y_train, v[index])

            support = svm.support_
            alpha_support = abs(svm.dual_coef_[0])
            alpha = np.zeros(m)
            b = svm.intercept_[0]
            alpha[index[support]] = alpha_support
            v_support = v[index[support]]
            support_E = np.where((alpha_support > 0) & (alpha_support == C * v_support))[0]
            support_S = np.where((alpha_support > 0) & (alpha_support != C * v_support))[0]
            index_R = np.delete(index, support)
            index_E = index[support[support_E]]
            index_S = index[support[support_S]]
            loss = C * np.maximum(np.zeros(m), 1 - y * svm.decision_function(X))
            v_new = np.maximum(np.zeros(m), 1 - loss / lam)
            if max(abs(v_new - v)) <= tol or ite > max_ite:
                v = v_new
                break
            v = v_new
            ite += 1
        solution_path.append(np.append(alpha, b))
        i += 1
        if i == length:
            break
        lam = lam_sequence[i]

        gram_EE = gram[np.ix_(index_E, index_E)]
        gram_SS = gram[np.ix_(index_S, index_S)]
        gram_ES = gram[np.ix_(index_E, index_S)]
        gram_SE = gram[np.ix_(index_S, index_E)]
        gram_S = gram[:, index_S]
        gram_E = gram[:, index_E]
        num_E = len(index_E)
        num_S = len(index_S)

        y_E = y[index_E].reshape(num_E, 1)
        y_S = y[index_S].reshape(num_S, 1)

        while i < length:
            jacobian = np.block([[gram_SS, gram_SE, y_S],
                                 [-C ** 2 / lam * gram_ES, np.identity(num_E) - C ** 2 / lam * gram_EE,
                                  -C ** 2 / lam * y_E],
                                 [y_S.T, y_E.T, 0]])
            constant = np.block([[np.ones((num_S, 1))],
                                 [(C - C ** 2 / lam) * np.ones((num_E, 1))],
                                 [0]])
            a = np.linalg.pinv(jacobian) @ constant
            alpha = np.zeros(m)
            alpha[index_S] = a[:num_S].reshape(num_S, )
            alpha[index_E] = a[num_S:-1].reshape(num_E, )
            b = a[-1]
            alpha_S = alpha[index_S]
            alpha_E = alpha[index_E]
            loss = C * (1 - gram_S @ alpha_S.reshape(num_S, 1) - gram_E @ alpha_E.reshape(num_E, 1) - y.reshape(m,
                                                                                                                1) * b)
            SR_vio = np.where(alpha[index_S] < 0)[0]
            SE_vio = np.where(alpha[index_S] > C)[0]
            ES_vio = np.where(loss[index_E] < 10e-7)[0]
            Ed_vio = np.where(loss[index_E] > lam)[0]
            RS_vio = np.where(loss[index_R] > -10e-7)[0]
            dE_vio = np.where(loss[d_index] < lam)[0]

            S_vio = len(SR_vio) + len(SE_vio)
            E_vio = len(ES_vio) + len(Ed_vio)
            R_vio = len(RS_vio)
            d_vio = len(dE_vio)
            sum_vio = S_vio + E_vio + R_vio + d_vio
            if sum_vio == 0:
                v = np.maximum(0, 1 - np.maximum(0, loss) / lam).reshape(-1, )
                solution_path.append(np.append(alpha, b))
                i += 1
                if i == length:
                    break
                lam = lam_sequence[i]
            else:
                if sum_vio == 1:
                    if S_vio > 0:
                        if len(SR_vio) > 0:
                            index_R = np.append(index_R, index_S[SR_vio])
                            index_S = np.delete(index_S, SR_vio)
                        else:
                            index_E = np.append(index_E, index_S[SE_vio])
                            index_S = np.delete(index_S, SE_vio)
                    if R_vio > 0:
                        index_S = np.append(index_S, index_R[RS_vio])
                        index_R = np.delete(index_R, RS_vio)
                    if d_vio > 0:
                        index = np.append(index, d_index[dE_vio])
                        index_E = np.append(index_E, d_index[dE_vio])
                        d_index = np.delete(d_index, dE_vio)
                    if E_vio > 0:
                        if len(ES_vio) > 0:
                            index_S = np.append(index_S, index_E[ES_vio])
                            index_E = np.delete(index_E, ES_vio)
                        else:
                            index_S = np.append(index_S, index_E[Ed_vio])
                            index_E = np.delete(index_E, Ed_vio)
                            index = np.delete(index, Ed_vio)

                    gram_EE = gram[np.ix_(index_E, index_E)]
                    gram_SS = gram[np.ix_(index_S, index_S)]
                    gram_ES = gram[np.ix_(index_E, index_S)]
                    gram_SE = gram[np.ix_(index_S, index_E)]
                    gram_S = gram[:, index_S]
                    gram_E = gram[:, index_E]
                    num_E = len(index_E)
                    num_S = len(index_S)

                    y_E = y[index_E].reshape(num_E, 1)
                    y_S = y[index_S].reshape(num_S, 1)
                    jacobian = np.block([[gram_SS, gram_SE, y_S],
                                         [-C ** 2 / lam * gram_ES, np.identity(num_E) - C ** 2 / lam * gram_EE,
                                          -C ** 2 / lam * y_E],
                                         [y_S.T, y_E.T, 0]])
                    constant = np.block([[np.ones((num_S, 1))],
                                         [(C - C ** 2 / lam) * np.ones((num_E, 1))],
                                         [0]])
                    a = np.linalg.pinv(jacobian) @ constant
                    alpha = np.zeros(m)
                    alpha[index_S] = a[:num_S].reshape(num_S, )
                    alpha[index_E] = a[num_S:-1].reshape(num_E, )
                    b = a[-1]
                    alpha_S = alpha[index_S]
                    alpha_E = alpha[index_E]
                    loss = C * (1 - gram_S @ alpha_S.reshape(num_S, 1) - gram_E @ alpha_E.reshape(num_E, 1) - y.reshape(
                        m, 1) * b)
                    v_new = np.maximum(0, np.maximum(0, loss))
                    S_vio = len(np.where(alpha[index_S] < 0)[0]) + len(np.where(alpha[index_S] > C)[0]) + len(
                        np.where(abs(loss[index_S]) > 10e-7)[0])
                    E_vio = len(np.where(loss[index_E] < 10e-7)[0]) + len(np.where(loss[index_E] > lam)[0]) + len(
                        np.where(abs(C * v_new[index_E].reshape(-1, ) - alpha[index_E]) > 10e-7)[0])
                    R_vio = len(np.where(loss[index_R] > -10e-7)[0])
                    d_vio = len(np.where(loss[d_index] < lam)[0])
                    sum_vio = S_vio + E_vio + R_vio + d_vio
                    if sum_vio == 0:
                        v = v_new.reshape(-1, )
                        solution_path.append(np.append(alpha, b))
                        i += 1
                        if i == length:
                            break
                        lam = lam_sequence[i]
                    else:
                        break
                else:
                    break
    return solution_path


def GAGA_svm_mixture(X, y, v, lam_sequence, C=1, gamma=1, tol=1e-10, max_ite=600, tol_svm=1e-10):
    m = X.shape[0]
    n = X.shape[1]
    gram = rbf_kernel(X, gamma=1 / (n * X.var())) * linear_kernel(y.reshape(m, 1))
    solution_path = []
    i = 0
    length = len(lam_sequence)

    svm = SVC(C=C, gamma=1 / (n * X.var()), tol=tol_svm)
    d_index = np.where(v == 0)[0]
    index = np.where(v != 0)[0]
    X_train = np.delete(X, d_index, 0)
    y_train = np.delete(y, d_index, 0)
    svm.fit(X_train, y_train, v[index])
    lam = lam_sequence[i]
    loss = C * np.maximum(np.zeros(m), 1 - y * svm.decision_function(X))
    l_0 = loss == 0
    v[l_0] = 1
    v[~l_0] = np.minimum(np.ones(m - sum(l_0)),
                         np.maximum(np.zeros(m - sum(l_0)), gamma * (1 / np.sqrt(loss[~l_0]) - 1 / lam)))
    while i < length:
        # acs找partial optimum
        ite = 1
        v_new = np.zeros(m)

        while True:
            d_index = np.where(v == 0)[0]
            index = np.where(v != 0)[0]
            X_train = np.delete(X, d_index, 0)
            y_train = np.delete(y, d_index, 0)
            svm.fit(X_train, y_train, v[index])

            support = svm.support_
            alpha_support = abs(svm.dual_coef_[0])
            alpha = np.zeros(m)
            alpha[index[support]] = alpha_support
            b = svm.intercept_[0]

            loss = C * np.maximum(np.zeros(m), 1 - y * svm.decision_function(X))
            l_0 = loss == 0
            v_new[l_0] = 1
            v_new[~l_0] = np.minimum(np.ones(m - sum(l_0)),
                                     np.maximum(np.zeros(m - sum(l_0)), gamma * (1 / np.sqrt(loss[~l_0]) - 1 / lam)))

            v_support = v_new[index[support]]
            support_E = np.where((alpha_support > 0) & (alpha_support == C * v_support))[0]
            support_S = np.where((alpha_support > 0) & (alpha_support != C * v_support))[0]
            support_E1 = np.where(v[index[support[support_E]]] == 1)
            support_E2 = np.where(v[index[support[support_E]]] != 1)
            index_R = np.delete(index, support)
            index_S = index[support[support_S]]
            index_E1 = index[support[support_E[support_E1]]]
            index_E2 = index[support[support_E[support_E2]]]

            if max(abs(v_new - v)) <= tol or ite > max_ite:
                v = v_new.copy()
                break
            v = v_new.copy()
            ite += 1
        solution_path.append(np.append(alpha, b))
        i += 1
        if i == length:
            break
        lam = lam_sequence[i]

        num_R = len(index_R)
        num_S = len(index_S)
        num_E1 = len(index_E1)
        num_E2 = len(index_E2)
        gram_E2E2 = gram[np.ix_(index_E2, index_E2)]
        gram_E2E1 = gram[np.ix_(index_E2, index_E1)]
        gram_SS = gram[np.ix_(index_S, index_S)]
        gram_E2S = gram[np.ix_(index_E2, index_S)]
        gram_SE2 = gram[np.ix_(index_S, index_E2)]
        gram_S = gram[:, index_S]
        gram_E1 = gram[:, index_E1]
        gram_E2 = gram[:, index_E2]
        y_E2 = y[index_E2].reshape(num_E2, 1)
        y_S = y[index_S].reshape(num_S, 1)

        def f(lam, active, gram_E2S, gram_E2E2, y_S, y_E2, num_S, num_E2):

            b = active[-1]
            alpha_S = active[:num_S]
            alpha_E2 = active[num_S:-1]
            l_E2 = C * (1 - gram_E2S @ alpha_S.reshape(num_S, 1) - C * gram_E2E1.sum(axis=1).reshape(num_E2,
                                                                                                     1) - gram_E2E2 @ alpha_E2.reshape(
                num_E2, 1) - y_E2.reshape(num_E2, 1) * b)
            d1 = np.zeros((num_E2, 1))

            d1[l_E2 > 0] = gamma / 2 * l_E2[l_E2 > 0] ** (-3 / 2)
            jacobian = np.block([[gram_SS, gram_SE2, y_S],
                                 [C ** 2 * d1 * gram_E2S, C ** 2 * d1 * gram_E2E2 - np.identity(num_E2),
                                  C ** 2 * d1 * y_E2],
                                 [-y_S.T, -y_E2.T, 0]])
            constant = np.block([[np.zeros((num_S, 1))],
                                 [-C * gamma / lam ** 2 * np.ones(num_E2).reshape(-1, 1)],
                                 [0]])
            return (pinv(jacobian) @ constant).reshape(num_E2 + num_S + 1, )

        active = np.append(np.append(alpha[index_S], alpha[index_E2]), b)
        sol = solve_ivp(f, [lam_sequence[i - 1], lam_sequence[-1]], active,
                        args=(gram_E2S, gram_E2E2, y_S, y_E2, num_S, num_E2), dense_output=True)

        while i < length:
            a = sol.sol(lam)
            alpha = np.zeros(m)
            alpha[index_S] = a[:num_S].reshape(num_S, )
            alpha[index_E1] = C * np.ones((num_E1,))
            alpha[index_E2] = a[num_S:-1].reshape(num_E2, )
            b = a[-1]
            alpha_S = alpha[index_S]
            alpha_E1 = alpha[index_E1]
            alpha_E2 = alpha[index_E2]
            loss = C * (1 - gram_S @ alpha_S.reshape(num_S, 1) - gram_E1 @ alpha_E1.reshape(num_E1,
                                                                                            1) - gram_E2 @ alpha_E2.reshape(
                num_E2, 1) - y.reshape(m, 1) * b)
            SR_vio = np.where(alpha[index_S] < 0)[0]
            SE_vio = np.where(loss[index_S] > 10e-7)[0]
            ES_vio = np.where(loss[index_E1] < 10e-7)[0]
            Ed_vio = np.where(loss[index_E2] > lam ** 2)[0]
            RS_vio = np.where(loss[index_R] > -10e-7)[0]
            dE_vio = np.where(loss[d_index] < lam ** 2)[0]

            S_vio = len(SR_vio) + len(SE_vio)
            E_vio = len(ES_vio) + len(Ed_vio)
            R_vio = len(RS_vio)
            d_vio = len(dE_vio)

            sum_vio = S_vio + E_vio + R_vio + d_vio

            if sum_vio == 0:
                loss = np.maximum(0, loss).reshape((-1,))
                l_0 = loss == 0
                v_new[l_0] = 1
                v_new[~l_0] = np.minimum(np.ones(m - sum(l_0)), np.maximum(np.zeros(m - sum(l_0)),
                                                                           gamma * (1 / np.sqrt(loss[~l_0]) - 1 / lam)))
                solution_path.append(np.append(alpha, b))
                i += 1
                if i == length:
                    break
                lam = lam_sequence[i]
            else:
                if sum_vio == 1:
                    if S_vio > 0:
                        if len(SR_vio) > 0:
                            index_R = np.append(index_R, index_S[SR_vio])
                            index_S = np.delete(index_S, SR_vio)
                        else:
                            index_E1 = np.append(index_E1, index_S[SE_vio])
                            index_S = np.delete(index_S, SE_vio)

                    if R_vio > 0:
                        index_S = np.append(index_S, index_R[RS_vio])
                        index_R = np.delete(index_R, RS_vio)

                    if d_vio > 0:
                        index = np.append(index, d_index[dE_vio])
                        index_E2 = np.append(index_E2, d_index[dE_vio])
                        d_index = np.delete(d_index, dE_vio)

                    if E_vio > 0:
                        if len(ES_vio) > 0:
                            index_S = np.append(index_S, index_E1[ES_vio])
                            index_E1 = np.delete(index_E1, ES_vio)
                        else:
                            index_S = np.append(index_S, index_E2[Ed_vio])
                            index_E2 = np.delete(index_E2, Ed_vio)
                            index = np.delete(index, Ed_vio)

                    num_R = len(index_R)
                    num_S = len(index_S)
                    num_E1 = len(index_E1)
                    num_E2 = len(index_E2)
                    gram_E2E2 = gram[np.ix_(index_E2, index_E2)]
                    gram_E2E1 = gram[np.ix_(index_E2, index_E1)]
                    gram_SS = gram[np.ix_(index_S, index_S)]
                    gram_E2S = gram[np.ix_(index_E2, index_S)]
                    gram_SE2 = gram[np.ix_(index_S, index_E2)]
                    gram_S = gram[:, index_S]
                    gram_E1 = gram[:, index_E1]
                    gram_E2 = gram[:, index_E2]

                break
    return solution_path


def GAGA_Logistic_linear(X, y, v, lam_sequence, C=1, tol=1e-10, max_ite=600, tol_model=1e-5):
    j = 0
    m = X.shape[0]
    n = X.shape[1]
    solution_path = []
    i = 0
    length = len(lam_sequence)

    clf = LogisticRegression(C=C, warm_start=True, tol=tol_model, max_iter=1000)
    index_D = np.where(v == 0)[0]
    index_E = np.where(v != 0)[0]
    X_E = X[index_E]
    y_E = y[index_E]
    clf.fit(X_E, y_E, v[index_E])
    lam = lam_sequence[i]
    loss = C * np.log(1 + np.e ** (-y * clf.decision_function(X)))
    v = np.maximum(np.zeros(m), 1 - loss / lam)
    while i < length and len(index_E) != m:
        ite = 1
        while True:
            v_new = np.zeros(m)
            index_D = np.where(v == 0)[0]
            index_E = np.where(v != 0)[0]
            X_E = X[index_E]
            y_E = y[index_E]
            clf.fit(X_E, y_E, v[index_E])
            loss = C * np.log(1 + np.e ** (-y * clf.decision_function(X)))
            v_new = np.maximum(np.zeros(m), 1 - loss / lam)
            if max(abs(v_new - v)) <= tol or ite > max_ite:
                break
            v = v_new.copy()
            ite += 1
        w = clf.coef_[0]
        b = clf.intercept_
        solution_path.append(np.append(w, b))

        i += 1
        if i == length or len(index_E) == m:
            break

        lam = lam_sequence[i]
        num_E = len(index_E)
        y_E = y_E.reshape(num_E, 1)

        def f(lam, active, X_E, y_E):
            b = active[-1]
            w = active[:-1].reshape((-1, 1))
            l = C * np.log(1 + np.e ** (-y_E * (X_E @ w + b)))

            tmp1 = np.e ** (-l / C) - 1
            tmp2 = y_E * (-1 / lam * tmp1 - 1 / C * (1 - l / lam) * (tmp1 + 1)) * C * y_E * tmp1

            jacobian = np.block([[np.identity(n) + C * X_E.T @ (tmp2 * X_E), C * X_E.T @ tmp2],
                                 [C * np.sum(tmp2 * X_E, axis=0), C * np.sum(tmp2)]])
            constant = np.block([[C * X_E.T @ (y_E * l / lam ** 2 * tmp1)],
                                 [C * np.sum(y_E * l / lam ** 2 * (np.e ** (-l / C) - 1), 0)]])

            return (-pinv(jacobian) @ constant).reshape(n + 1, )

        active = np.append(w, b)
        try:
            sol = solve_ivp(f, [lam_sequence[i - 1], lam_sequence[min(i + 50, length - 1)]], active, args=(X_E, y_E),
                            dense_output=True)
        except Exception:
            j += 1
            pass

        while i < length:
            a = sol.sol(lam)
            w = a[:-1]
            b = a[-1]
            ite = 1
            loss = C * np.log(1 + np.e ** (-y * (X @ w + b)))
            E_vio = np.where(loss[index_E] >= lam)[0]
            D_vio = np.where(loss[index_D] < lam)[0]
            e_vio = len(E_vio)
            d_vio = len(D_vio)

            sum_vio = e_vio + d_vio

            if sum_vio == 0:
                v = np.maximum(np.zeros(m), 1 - loss / lam)
                solution_path.append(np.append(w, b))
                i += 1
                if i == length or len(index_E) == m:
                    break
                lam = lam_sequence[i]
            else:
                break

    return solution_path


def GAGA_Logistic_mixture(X, y, v, lam_sequence, gamma=1, C=1, max_iter=6e2, max_iter_model=1e3, tol=1e-10, tol_model=1e-5):
    ite_path = []
    t_acs = 0
    t_ode = 0
    critical = 0
    j = 0
    m = X.shape[0]
    n = X.shape[1]
    solution_path = []
    i = 0
    length = len(lam_sequence)

    clf = LogisticRegression(C=C, warm_start=True, tol=tol_model, max_iter=max_iter_model)
    index_D = np.where(v == 0)[0]
    index_E = np.where(v == 1)[0]
    index_M = np.where((v > 0) & (v < 1))[0]
    index_train = np.concatenate((index_E, index_M))
    X_train = X[index_train]
    y_train = y[index_train]
    clf.fit(X_train, y_train, v[index_train])
    lam = lam_sequence[i]
    loss = C * np.log(1 + np.e ** (-y * clf.decision_function(X)))
    l_0 = loss == 0
    v[l_0] = 1
    v[~l_0] = np.minimum(np.ones(m - sum(l_0)),
                         np.maximum(np.zeros(m - sum(l_0)), gamma * (1 / np.sqrt(loss[~l_0]) - 1 / lam)))
    while i < length and len(index_E) != m:
        t1 = time.time()
        # acs找partial optimum
        ite = 1

        while True:
            v_new = np.zeros(m)
            index_D = np.where(v == 0)[0]
            index_E = np.where(v == 1)[0]
            index_M = np.where((v > 0) & (v < 1))[0]
            index_train = np.concatenate((index_E, index_M))
            X_train = X[index_train]
            y_train = y[index_train]

            clf.fit(X_train, y_train, v[index_train])
            loss = C * np.log(1 + np.e ** (-y * clf.decision_function(X)))
            l_0 = loss == 0
            v_new[l_0] = 1
            v_new[~l_0] = np.minimum(np.ones(m - sum(l_0)),
                                     np.maximum(np.zeros(m - sum(l_0)), gamma * (1 / np.sqrt(loss[~l_0]) - 1 / lam)))

            if max(abs(v_new - v)) <= tol or ite > max_iter:
                break
            v = v_new.copy()
            ite += 1
        X_E = X[index_E]
        X_M = X[index_M]
        y_E = y[index_E]
        y_M = y[index_M]
        t2 = time.time()
        t_acs += t2 - t1
        w = clf.coef_[0]
        b = clf.intercept_
        solution_path.append(np.append(w, b))
        ite_path.append(ite)

        i += 1
        if i == length or len(index_E) == m:
            break
        # l = C*np.log(1+np.e**(-y_E*(X_E@w+b)))
        # tmp1 = np.e**(-l/C)-1
        # tmp2 = y_E*(-1/lam*tmp1-1/C*(1-l/lam)*(tmp1+1))*C*y_E*tmp1

        lam = lam_sequence[i]

        def f(lam, active, X_E, y_E, X_M, y_M):

            b = active[-1]
            w = active[:-1].reshape((-1, 1))
            num_E = len(y_E)
            num_M = len(y_M)

            X_train = np.concatenate((X_E, X_M))
            y_train = np.concatenate((y_E, y_M)).reshape((-1, 1))

            y_E = y_E.reshape((-1, 1))
            y_M = y_M.reshape((-1, 1))

            num_train = num_E + num_M

            l = C * np.log(1 + np.e ** (-y_train * (X_train @ w + b)))
            l_M = l[num_E + 1:]

            tmp1 = np.e ** (-l / C) - 1
            tmp1_E = tmp1[:num_E]
            tmp1_M = tmp1[num_E + 1:]

            tmp2_E = -y_E ** 2 * (tmp1_E + 1) * tmp1_E
            tmp2_M = np.zeros((num_M, 1))
            index_M1 = np.where(l_M != 0)[0]
            # tmp2_M[index_M1] = y_M[index_M1]*(-gamma/2*l_M[index_M1]**(-3/2)*tmp1_M[index_M1]-1/C*gamma*(1/np.sqrt(l_M[index_M1])-1/lam)
            #                         *(tmp1_M[index_M1]+1))*C*y_M[index_M1]*tmp1_M[index_M1]

            l_power = l_M[index_M1] ** (-3 / 2) / 2

            tmp2_M[index_M1] = y_M[index_M1] ** 2 * gamma * (
                        l_power + (1 / lam - 1 / np.sqrt(l_M[index_M1]) - l_power) * (tmp1_M + 1)) * tmp1_M[index_M1]
            tmp2 = np.concatenate((tmp2_E, tmp2_M))

            jacobian = np.block([[np.identity(n) + C * X_train.T @ (tmp2 * X_train), C * X_train.T @ tmp2],
                                 [C * np.sum(tmp2 * X_train, axis=0), C * np.sum(tmp2)]])

            tmp3 = np.zeros((num_train, 1))
            tmp3[num_E:] = gamma
            constant = np.block([[C * X_train.T @ (y_train * tmp3 / lam ** 2 * tmp1)],
                                 [C * np.sum(y_train * tmp3 / lam ** 2 * tmp1, 0)]])

            return (-pinv(jacobian) @ constant).reshape(n + 1, )

        active = np.append(w, b)

        # method:LSODA\BDF\Radau\DOP853\RK23\RK45
        # try:

        sol = solve_ivp(fun=f, t_span=[lam_sequence[i - 1], lam_sequence[min(i + 50, length - 1)]], y0=active,
                        args=(X_E, y_E, X_M, y_M,), dense_output=True, vectorized=False, method='LSODA')
        # except Exception:
        #     j += 1
        #     pass
        while i < length:
            a = sol.sol(lam)
            w = a[:-1]
            b = a[-1]

            loss = C * np.log(1 + np.e ** (-y * (X @ w + b)))  # 0.0009

            E_vio = np.where(loss[index_E] >= (lam * gamma / (lam + gamma)) ** 2)[0]
            ME_vio = np.where(loss[index_M] < (lam * gamma / (lam + gamma)) ** 2)[0]
            MD_vio = np.where(loss[index_M] > lam ** 2)[0]
            D_vio = np.where(loss[index_D] <= lam ** 2)[0]

            e_vio = len(E_vio)
            m_vio = len(ME_vio) + len(MD_vio)
            d_vio = len(D_vio)

            sum_vio = e_vio + m_vio + d_vio

            if sum_vio == 0:
                l_0 = loss == 0
                v = np.zeros(m)
                v[l_0] = 1
                v[~l_0] = np.minimum(np.ones(m - sum(l_0)),
                                     np.maximum(np.zeros(m - sum(l_0)), gamma * (1 / np.sqrt(loss[~l_0]) - 1 / lam)))
                solution_path.append(np.append(w, b))
                i += 1
                if i == length:
                    break
                lam = lam_sequence[i]
            else:
                critical += 1
                # critical_path.append(i)
                break
    print(j)
    print(critical)
    print(f'acs:{t_acs},ode:{t_ode}')
    return solution_path, ite_path