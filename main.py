## Press the green button in the gutter to run the script.
import random
import time

import numpy as np
import ACS
import Weighted_Samples
import matplotlib.pyplot as plt
import Input_Data
import Evaluation
from sklearn import linear_model
import warnings
from sklearn.model_selection import KFold
from Input_Data import add_regression_noise_model1
from GAGA import GAGA_Lasso_linear

if __name__ == '__main__':
    RMSE_score = []
    MAE_score = []
    alpha = 0.006
    lam_start = 0.01
    lam_end = 10
    warnings.filterwarnings('ignore')
    noise_level = 0.3
    gamma = 1.0
    mod = 1 # 1 is for linear 2 is for mixture
    n_split = 8  # refers to the time it split the data
    # w_path = ACS.ACS_svm_linear(X, y, v, lam_sequence)
    # w_path = np.array(w_path)
    # for i in range(0, n):
    #     plt.plot(lam_sequence, w_path[:, i])
    # plt.show()1.

    # MOSPL.MOSPL_Lasso(X,y,regularizer='hard')
    "==============================================initialization======================================"
    X, y = Input_Data.regression_data1()
    lam_sequence__ = Weighted_Samples.lam_sequence(lam_start, lam_start+0.1, 3)
    KF = KFold(n_splits=20,random_state=40,shuffle=True)
    trail = 0
    original_clf = linear_model.Lasso(
        alpha=0.006,
        fit_intercept=False, warm_start=1,
        max_iter=2000,
        tol=1e-6,
        random_state=40)
    for train_index, test_index in KF.split(X):
        trail += 1
        print('calculationing on loop : ',trail)
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        m, n = np.shape(X_train)

        "====================================== get the partial optimum =============================================="

        v = np.random.uniform(0, 1, m)
        lam_converge = Weighted_Samples.lam_sequence(lam_start, lam_start + 0.1, 3)

        w_converge, b_converge, turning_point = ACS.ACS_Lasso(X_train, y_train, v, lam_converge, alpha, mode=mod,
                                                              gamma=gamma)

        w = np.array(w_converge[-1])
        b = np.array(b_converge[-1])
        loss_vector = np.square(y - w @ X.T - b)
        v = Weighted_Samples.linear_loss(loss_vector, lam_start + 0.1)

        "===================================== the preparation for evaluation ========================================"
        if noise_level != 0:
            X_train_, y_train_ = add_regression_noise_model1(X_train, y_train, noise_level=noise_level)
        else :
            X_train_, y_train_ = X_train, y_train
        lam_sequence = Weighted_Samples.lam_sequence(lam_start + 0.1, lam_end, 2)
        m, n = np.shape(X_train)

        "======================================train the model============================================="
        w, solution_path, lams, npoint = GAGA_Lasso_linear(X_train_, y_train_, w, lam_end, lam_start + 0.1)
        p, q = np.shape(solution_path)
        w_APSPL = np.array(solution_path)

        original_clf.fit(X_train_, y_train_)
        w_original = original_clf.coef_
        b = original_clf.intercept_
        w_original = [w_original]
        w_original = np.array(w_original)

        "===================================evluate the model=============================================="
        original_score = Evaluation.Lasso_score(w_original, X_test, y_test)
        APSPL_score = Evaluation.Lasso_score(w_APSPL, X_test, y_test)
        "=====================================create the tabel ============================================="
        RMSE_score.append([np.min(original_score[0]), np.min(APSPL_score[0])])
        MAE_score.append([np.min(original_score[1]), np.min(APSPL_score[1]),])
        "=============================form a table of breakpoints ======================================"
    "===================================statistically evaluate the model============================"
    RMSE_score = np.array(RMSE_score)
    MAE_score = np.array(MAE_score)
    rmse_score = np.sum(RMSE_score, axis=0)
    mae_score = np.sum(MAE_score, axis=0)
    rmse_average = rmse_score / 8
    rmse_std = np.sqrt(np.sum(np.square(RMSE_score - rmse_average), axis=0) / 7)
    mae_average = mae_score / 8
    mae_std = np.sqrt(np.sum(np.square(MAE_score - mae_average), axis=0) / 7)

    print('the rmse  of the model is: \n',RMSE_score)
    print('the average rmse of the model is: \n',rmse_average)
    print('the rmse std of the model is: \n',rmse_std)
    print('the mae  of the model is: \n',MAE_score)
    print('the average mae socre of the model is: \n',mae_average)
    print('the mae std of the model is: \n',mae_std)
    t = np.size(rmse_average)
    "==================================evaluate the model and plot learning curve=================="
    plt.figure(num=1)
    plt.title('The learning curve of ode algorithm')
    plt.subplot(1, 2, 1)
    plt.plot(lams, APSPL_score[0], color='b', linestyle='--', label='GAGA')
    plt.legend()
    plt.xlabel('age parameter')
    plt.ylabel('RMSE')
    plt.subplot(1, 2, 2)
    plt.xlabel('age parameter')
    plt.ylabel('MAE')
    plt.plot(lam_sequence, APSPL_score[1], color='b', linestyle='--', label='GAGA')
    plt.legend()

    print('the programme is over')