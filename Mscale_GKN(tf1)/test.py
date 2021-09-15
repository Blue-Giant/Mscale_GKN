import numpy as np
import tensorflow as tf
import matData2pLaplace
import DNN_Class_base


class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean2x = np.mean(x, axis=-1, keepdims=True)
        self.std2x = np.std(x, axis=-1, keepdims=True)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean2x) / (self.std2x + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std2x + self.eps)) + self.mean2x
        return x


def load_our_data():
    # 载入数据
    fileName2data = 'data/' + 'res258_a4_f1' + str('.mat')
    data_all = matData2pLaplace.load_Matlab_data(fileName2data)
    data_Aeps = data_all['meshA']
    data_solu = data_all['meshU']
    data_mesh = data_all['meshXY']

    trans_A = np.reshape(data_Aeps, newshape=[200, 258 * 258])
    trans_U = np.reshape(data_solu, newshape=[200, 258 * 258])

    eps = 0.00001
    mean2TA = np.mean(trans_A, axis=-1, keepdims=True)
    std2TA = np.std(trans_A, axis=-1, keepdims=True)

    temp1 = trans_A - mean2TA
    temp2 = std2TA + eps
    Gauss_TA = (trans_A-mean2TA) / (std2TA+eps)

    A_gauss_normalize = GaussianNormalizer(trans_A)
    gauss_TA = A_gauss_normalize.encode(trans_A)
    print('SSSSSSSSSS')


def load_our_data_split(batch2train=150):
    # 载入数据
    fileName2data = 'data/' + 'res258_a4_f1' + str('.mat')
    data_all = matData2pLaplace.load_Matlab_data(fileName2data)
    data_Aeps = data_all['meshA']
    data_solu = data_all['meshU']
    data_mesh = data_all['meshXY']

    trans_A = np.reshape(data_Aeps, newshape=[200, 258 * 258])
    trans_U = np.reshape(data_solu, newshape=[200, 258 * 258])
    trans_mesh = np.transpose(data_mesh, (1, 0))

    # 对数据进行归一化处理
    A_normalizer = DNN_Class_base.np_GaussianNormalizer(trans_A)
    normalize_Aeps = A_normalizer.encode(trans_A)

    # U_normalizer = DNN_base.np_GaussianNormalizer(trans_U)
    # normalize_U = A_normalizer.encode(trans_U)

    # 将数据分为训练集和测试集
    Train_Aeps = np.reshape(normalize_Aeps[0:batch2train, :], newshape=[batch2train, 258 * 258, 1])
    Train_U = np.reshape(trans_U[0:batch2train, :], newshape=[batch2train, 258 * 258, 1])
    print('AAAAAAAAAA')


def load_our_data_train_test(batch2train=150):
    res2train = 66
    res2test = 66
    # 载入数据
    fileName2train_data = 'data/' + 'res66_a4_f1_train' + str('.mat')
    # fileName2data = 'data/' + 'res258_a4_f1' + str('.mat')
    train_data = matData2pLaplace.load_Matlab_data(fileName2train_data)
    train_data_Aeps = train_data['meshA']
    train_data_solu = train_data['meshU']
    train_data_mesh = train_data['meshXY']

    shape2data = np.shape(train_data_Aeps)

    trans_A2train = np.reshape(train_data_Aeps, newshape=[shape2data[0], res2train * res2train])
    trans_U2train = np.reshape(train_data_solu, newshape=[shape2data[0], res2train * res2train])
    trans_mesh2train = np.transpose(train_data_mesh, (1, 0))

    # 对数据进行归一化处理
    A_normalizer2train = DNN_Class_base.np_GaussianNormalizer(trans_A2train)
    normalize_Aeps2train = A_normalizer2train.encode(trans_A2train)

    Train_Aeps = np.reshape(normalize_Aeps2train[0:batch2train, :], newshape=[batch2train, res2train * res2train, 1])
    Train_U = np.reshape(trans_U2train[0:batch2train, :], newshape=[batch2train, res2train * res2train, 1])

    fileName2test_data = 'data/' + 'res66_a4_f1_test' + str('.mat')
    test_data = matData2pLaplace.load_Matlab_data(fileName2test_data)
    test_data_Aeps = test_data['meshA']
    test_data_solu = test_data['meshU']
    test_data_mesh = test_data['meshXY']

    test_trans_A = np.reshape(test_data_Aeps, newshape=[1, res2test * res2test])
    test_trans_U = np.reshape(test_data_solu, newshape=[1, res2test * res2test])
    test_trans_mesh = np.transpose(test_data_mesh, (1, 0))

    A_normalizer2test = DNN_Class_base.np_GaussianNormalizer(test_trans_A)
    normalize_Aeps2test = A_normalizer2test.encode(test_trans_A)

    Test_Aeps = np.reshape(normalize_Aeps2test, newshape=[res2test * res2test, 1])
    Test_U = np.reshape(test_trans_U, newshape=[res2test * res2test, 1])
    print('AAAAAAA')


if __name__ == "__main__":
    # load_our_data()
    # load_our_data_split()
    load_our_data_train_test()