import sklearn.datasets as data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from sklearn.svm import SVC
import math
from scipy.io import arff
import pandas as pd

np.random.seed(40)
random.seed(40)

def read_science_digits(list):
    float_list = []
    for i in list:
        s = i.split('e')
        before_e = float(s[0])
        sign = s[1][0]
        after_e = int(s[1][1:])

        if sign == '+':
            float_num = before_e * math.pow(10,after_e)
        else:
            float_num = before_e * math.pow(10,-after_e)
        float_list.append(float_num)
    return float_list


def read_science_digits_num(string):
    s = string.split('e')
    before_e = float(s[0])
    sign = s[1][0]
    after_e = int(s[1][1:])

    if sign == '+':
        float_num = before_e * math.pow(10, after_e)
    else:
        float_num = before_e * math.pow(10, -after_e)
    return float_num


def add_noise_classificaiton_model1(X, y, noise_level, C=1.0):
    if noise_level != 0:
        mod = SVC(kernel='linear', tol=1e-5, cache_size=2048, C=C)
        mod.fit(np.matrix(X), y)
        w = mod.coef_
        b = mod.intercept_
        dis = np.abs(w @ X.T + b)
        rank = np.argsort(dis)
        length = int(0.3 * np.size(rank))

        pos = random.sample(range(length), int(noise_level * length))
        for j in pos:
            y[rank[j]] = -y[rank[j]]
    return X, y


def add_regression_noise_model1(X,y,noise_level):
    """

    :param X: Sample
    :param y:
    :param noise_level: the percentage of noise
    :return:
    """
    "===================================add artificial niose====================================="
    if noise_level != 0:
        m, n = np.shape(X)
        length = int(noise_level * m)
        pos = random.sample(range(m), length)  # select samples
        X[pos, :] += np.random.RandomState(40).normal(loc = 0.2,scale = 3.0,size=(length, n))
        y[pos] += np.random.RandomState(40).normal(loc =0.2,scale = 3.0,size=length)

    return X, y


def handwritten_digits1(noise_level=0, model=0):
    """

    :return: the two class digits
    """
    X, y = data.load_digits(return_X_y=True)
    m, n = np.shape(X)
    y[y <= 5] = -1
    y[y >= 5] = 1

    X, y = add_noise_classificaiton_model1(X,y,noise_level=noise_level)
    return X,y


def handwritten_digits2(noise_level=0, model=1):
    """

    :return:
    """
    file = open("datasets\mfeat-pix.txt")
    Z = file.readlines()
    X = Z[0]
    X = X.strip(' ')
    X = X.split()
    X = list(map(int, X))
    X = np.array(X)
    y = [-1]
    count = 0
    for x in Z:
        x_ = x.strip(' ')
        x_ = x_.split()
        x_ = list(map(int, x_))
        X = np.vstack([X, x_])
        if count < 500:
            y.append(-1)
        else:
            y.append(1)
        count += 1

    X, y = add_noise_classificaiton_model1(X, y, noise_level=noise_level)
    return X,y


def handwritten_digits3(noise_level=0, model=1):
    file = open('datasets\pendigits.txt')
    y = [-1]
    Z = file.readlines()
    X = Z[0]
    X = X.strip()
    X = X.split(',')
    X = list(map(int, X))
    X = X[0:-1]

    for x in Z:
        x = x.strip()
        x = x.split(',')
        x = list(map(int, x))
        y_ = x[-1]
        x = x[0:-1]
        X = np.vstack([X, x])
        if y_ >= 5:
            y.append(1)
        else:
            y.append(-1)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    X, y = add_noise_classificaiton_model1(X, y, noise_level=noise_level)
    return X,y


def regression_data1(noise_level=0,model=1):
    X, y = data.load_breast_cancer(return_X_y=True)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    "=========================split set into training set and test set==========================="
    return X, y


def regression_data2(noise_level=0,model=1):
    X, y = data.load_boston(return_X_y=True)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="

    "=========================split set into training set and test set==========================="
    return X,y


def regression_data3(noise_level=0,model=1):
    file = open('datasets/lib_regression2.txt')
    Z = file.readlines()
    y = []
    X = np.ones(6)
    for sample in Z[1:]:
        sample = sample.strip('')
        sample = sample.split()
        x = np.zeros(6)
        y.append(float(sample[0]))
        for data in sample[1:]:
            data = data.split(':')
            pos = int(data[0])
            value = float(data[1])
            x[pos-1] = value
        X = np.vstack([X,x])
    np.delete(X,0,axis=0)
    "===================================basic processing of data =============================================="
    X = np.delete(X, 0, axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="

    "=========================split set into training set and test set==========================="
    return X,y


def regression_data4(noise_level=0,model=1):
    file = open('datasets/lib_regression3.txt')
    Z = file.readlines()
    y = []
    X = np.ones(12)
    for sample in Z[1:]:
        sample = sample.strip('')
        sample = sample.split()
        x = np.zeros(12)
        y.append(float(sample[0]))
        for data in sample[1:]:
            data = data.split(':')
            pos = int(data[0])
            value = float(data[1])
            x[pos-1] = value
        X = np.vstack([X,x])
    X = np.delete(X,0,axis=0)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y))/np.max(y)
    "===================================add artificial niose====================================="

    "=========================split set into training set and test set==========================="
    return X,y


def regression_data5(noise_level=0,model=1):
    file = open('datasets/regression_1_houses.txt')
    Z = file.readlines()
    y = []
    X = np.ones(8)
    for sample in Z:
        sample = sample.strip('')
        sample = sample.split(',')
        sample = list(map(float,sample))
        X = np.vstack([X,sample[0:-1]])
        y.append(sample[-1])
    "===================================basic processing of data =============================================="
    X = np.delete(X,0,axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="

    "=========================split set into training set and test set==========================="
    return X,y


def regression_data6(noise_level=0,model=1):
    file = open('datasets/regression_3_kin8mn.txt')
    Z = file.readlines()
    y = []
    X = np.ones(8)
    for sample in Z:
        sample = sample.strip(' ')
        sample = sample.strip('\n')
        sample = sample.split(',')
        sample = read_science_digits(sample)
        y.append(sample[-1])
        X = np.vstack([X,sample[0:-1]])
    "===================================basic processing of data =============================================="
    X = np.delete(X, 0, axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    "===================================add artificial niose====================================="

    "=========================split set into training set and test set==========================="
    return X,y


def regression_data7(noise_level=0,model=1):
    file = open('datasets/regression_4_mg.txt')
    Z = file.readlines()
    y = []
    X = np.ones(6)
    for sample in Z:
        sample = sample.strip(' ')
        sample = sample.strip('\n')
        sample = sample.split()
        x = np.ones(6)
        for i in range(0,len(sample)):
            if i == 0:
                y.append(read_science_digits_num(sample[i]))
            else:
                spl = sample[i].split(':')
                pos = int(spl[0])
                val = read_science_digits_num(spl[1])
                x[pos-1] = val
        X = np.vstack([X, x])
    "===================================basic processing of data =============================================="
    X = np.delete(X, 0, axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = y - np.min(y) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="

    "=========================split set into training set and test set==========================="
    return X,y


def regression_data8(noise_level=0,model=1):
    file = open('datasets/regression_5_liver.txt')
    Z = file.readlines()
    y = []
    X = np.ones(6)
    for sample in Z:
        sample = sample.strip('')
        sample = sample.split(',')
        sample = list(map(float,sample))
        X = np.vstack([X,sample[1:]])
        y.append(sample[0])
    "===================================basic processing of data =============================================="
    X = np.delete(X,0,axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="
    "=========================split set into training set and test set==========================="
    return X,y


def regression_data9(noise_level=0,model=1):
    file = open('datasets/regression_6_air_craft.txt')
    Z = file.readlines()
    y = []
    X = np.ones(6)
    for sample in Z:
        sample = sample.strip('')
        sample = sample.split(',')
        sample = list(map(float,sample))
        X = np.vstack([X,sample[1:]])
        y.append(sample[0])
    "===================================basic processing of data =============================================="
    X = np.delete(X,0,axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="
    "=========================split set into training set and test set==========================="
    return X,y


def regression_data10(noise_level=0,model=1):
    file = open('datasets/regression_7_wind.txt')
    Z = file.readlines()
    y = []
    X = np.ones(6)
    for sample in Z:
        sample = sample.strip('')
        sample = sample.split(',')
        sample = [float(x) for x in sample]
        X = np.vstack([X,sample[0:-1]])
        y.append(sample[-1])
    "===================================basic processing of data =============================================="
    X = np.delete(X,0,axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="
    "=========================split set into training set and test set==========================="
    return X,y


def regression_data10(noise_level=0,model=1):
    file = open('datasets/regression_8_music.txt')
    Z = file.readlines()
    y = []
    X = np.ones(117)
    for sample in Z:
        sample = sample.strip('')
        sample = sample.split(',')
        sample = list(map(float,sample))
        y.append(sample[100])
        del sample[100]
        X = np.vstack([X,sample])
    "===================================basic processing of data =============================================="
    X = np.delete(X,0,axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="
    "=========================split set into training set and test set==========================="
    return X,y

def regression_data11(noise_level=0,model=1):
    file = open('datasets/soft.txt')
    Z = file.readlines()
    y = []
    X = np.ones(6)
    for sample in Z:
        sample = sample.strip('')
        sample = sample.split(',')
        sample = list(map(float, sample))
        y.append(sample[-1])
        del sample[-1]
        X = np.vstack([X, sample])
    "===================================basic processing of data =============================================="
    X = np.delete(X, 0, axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "=========================split set into training set and test set==========================="
    return X, y


def regression_data12(noise_level=0,model=1):
    file = open('datasets/house.txt')
    Z = file.readlines()
    y = []
    X = np.ones(6)
    for sample in Z:
        sample = sample.strip(' ')
        sample = sample.strip('\n')
        sample = sample.split(',')
        x = np.ones(6)
        for i in range(0,len(sample)):
            if i == 0:
                y.append(read_science_digits_num(sample[i]))
            else:
                spl = sample[i].split(':')
                pos = int(spl[0])
                val = read_science_digits_num(spl[1])
                x[pos-1] = val
        X = np.vstack([X, x])
    "===================================basic processing of data =============================================="
    X = np.delete(X, 0, axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = y - np.min(y) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="

    "=========================split set into training set and test set==========================="
    return X,y

def regression_data13(noise_level=0,model=1):
    file = open('datasets/ele.txt')
    Z = file.readlines()
    y = []
    X = np.ones(18)
    for sample in Z:
        sample = sample.strip('')
        sample = sample.split(',')
        sample = list(map(float,sample))
        y.append(sample[18])
        del sample[18]
        X = np.vstack([X,sample])
    "===================================basic processing of data =============================================="
    X = np.delete(X,0,axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="
    "=========================split set into training set and test set==========================="
    return X,y


def regression_data14(noise_level=0,model=1):
    file = open('datasets/nasa.txt')
    Z = file.readlines()
    y = []
    X = np.ones(21)
    for sample in Z:
        sample = sample.strip('')
        sample = sample.split(',')
        sample = list(map(float,sample))
        y.append(sample[-1])
        del sample[-1]
        X = np.vstack([X,sample])
    "===================================basic processing of data =============================================="
    X = np.delete(X,0,axis=0)
    y = np.array(y)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="
    "=========================split set into training set and test set==========================="
    return X,y

def regression_data15(noise_level=0,model=1):
    data, meta = arff.loadarff('./datasets/fried.arff')
    m = np.shape(data)[0]
    data = [list(data[i]) for i in range(0,m)]
    data = np.array(data)
    X = data[:,1:]
    y = data[:,0]
    y = np.array(y)
    y = y.reshape(m)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="
    "=========================split set into training set and test set==========================="
    return X,y

def regression_data16(noise_level=0,model=1):
    data, meta = arff.loadarff('./datasets/BNG_stock.arff')
    m = np.shape(data)[0]
    data = [list(data[i]) for i in range(0,m)]
    data = np.array(data)
    X = data[:, 1:]
    y = data[:, 0]
    y = np.array(y)
    y = y.reshape(m)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="
    "=========================split set into training set and test set==========================="
    return X,y

def regression_data17(noise_level=0,model=1):
    data, meta = arff.loadarff('./datasets/ailerons.arff')
    m = np.shape(data)[0]
    data = [list(data[i]) for i in range(0,m)]
    data = np.array(data)
    X = data[:, 1:]
    y = data[:, 0]
    y = np.array(y)
    y = y.reshape(m)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="
    "=========================split set into training set and test set==========================="
    return X,y

def regression_data18(noise_level=0,model=1):
    data, meta = arff.loadarff('./datasets/datano.arff')
    m = np.shape(data)[0]
    data = [list(data[i]) for i in range(0,m)]
    data = np.array(data)
    X = data[:, 1:]
    y = data[:, 0]
    y = np.array(y)
    y = y.reshape(m)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="
    "=========================split set into training set and test set==========================="
    return X,y

def regression_data19(noise_level=0,model=1):
    data, meta = arff.loadarff('./datasets/phpnf7h1l.arff')
    m = np.shape(data)[0]
    data = [list(data[i]) for i in range(0,m)]
    data = np.array(data)
    X = data[:, 1:]
    y = data[:, 0]
    y = np.array(y)
    y = y.reshape(m)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)
    y = y - np.mean(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    "===================================add artificial niose====================================="
    "=========================split set into training set and test set==========================="
    return X,y

def read_arrf(file):
    with open(file, encoding="utf-8") as f:
        header = []
        for line in f:
            if line.startswith("@attribute"):
                header.append(line.split()[1])
            elif line.startswith("@data"):
                break
        df = pd.read_csv(f, header=None)
        df.columns = header
    return df


def classification_data1():
    panda = read_arrf('./datasets/phpDYCOet.arff')
    a = 8
    X = panda.values[:,:-1]
    m,n = np.shape(X)
    y = panda.values[:,-1]
    y.reshape(m)
    y[y==2] = -1

    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)

    return X,y


def classification_data2():
    panda = read_arrf('./datasets/electricity-normalized.arff')
    X = panda.iloc[:, 0:-1].values
    y = panda.iloc[:,-1].values
    y[y=='UP'] = 1
    y[y=='DOWN'] = -1
    m, n = panda.shape
    y.reshape(m)
    y[y == 2] = -1
    y = y.astype(np.float64)

    return X, y


def classification_data3():
    panda = read_arrf('./datasets/MagicTelescope.arff')
    X = panda.iloc[:,:-1].values
    y = panda.iloc[:, -1].values
    y[y=='h'] = float(1)
    y[y=='g'] = float(-1)
    m, n = panda.shape
    y.reshape(m)
    y[y == 2] = -1

    y = y.astype(np.float64)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)

    return X, y

def classification_data4():
    panda = read_arrf('./datasets/phpfGCaQC.arff')
    X = panda.iloc[:,1:].values
    y = panda.iloc[:, 0].values
    m, n = panda.shape
    y.reshape(m)
    y[y == 0] = -1

    y = y.astype(np.float64)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)

    return X, y


def classification_data5():
    panda = read_arrf('./datasets/letter.arff')
    X = panda.iloc[:,:-1].values
    y = panda.iloc[:, -1].values
    m, n = panda.shape
    y.reshape(m)
    y[y == 'N'] = -1
    y[y == 'P'] == 1

    y = y.astype(np.float64)
    stand = StandardScaler()
    stand.fit(X)
    X = stand.transform(X)

    return X, y
