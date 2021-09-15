"""
@author: LXA
 Date: 2021 年 6 月 17 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import time
import General_Laplace
import MS_LaplaceEqs
import MS_BoltzmannEqs
import MS_ConvectionEqs
import MGKN_data
import MGKNdata_base
import matData2pLaplace
import matData2Laplace
import matData2Boltzmann
import matData2Convection
import DNN_Class_base
import DNN_base
import MGKN_tools
import MGKN_PrintLog
import saveData
import plotData


def dictionary_out2file(R_dic, log_fileout):
    MGKN_tools.log_string('PDE type for problem: %s\n' % (R_dic['PDE_type']), log_fileout)
    MGKN_tools.log_string('Equation name for problem: %s\n' % (R_dic['equa_name']), log_fileout)

    MGKN_tools.log_string('Network model for dealing the input: %s\n' % str(R_dic['model2input']), log_fileout)
    MGKN_tools.log_string('hidden layer for dealing the input:%s\n' % str(R_dic['hiddens2input']), log_fileout)
    MGKN_tools.log_string('The out-dim of input model:%s\n' % str(R_dic['output_dim']), log_fileout)
    MGKN_tools.log_string('In activate function for the input-model: %s\n' % str(R_dic['actInFunc2input']),
                          log_fileout)
    MGKN_tools.log_string('Activate function for the input-model: %s\n' % str(R_dic['actFunc2input']), log_fileout)
    MGKN_tools.log_string('Out activate function for the input-model: %s\n' % str(R_dic['actOutFunc2input']),
                          log_fileout)
    if str.lower(R_dic['model2input']) == 'fourier_dnn':
        MGKN_tools.log_string('The frequency for kernel-model: %s\n' % str(R_dic['freq2input']), log_fileout)

    MGKN_tools.log_string('Network model for kernel: %s\n' % str(R_dic['model2kernel']), log_fileout)
    MGKN_tools.log_string('hidden layer for kernel:%s\n' % str(R_dic['hiddens2kernel']), log_fileout)
    MGKN_tools.log_string('In activate function for kernel-model: %s\n' % str(R_dic['actInFunc2kernel']), log_fileout)
    MGKN_tools.log_string('Activate function for kernel-model: %s\n' % str(R_dic['actFunc2kernel']), log_fileout)
    MGKN_tools.log_string('Out activate function for kernel-model: %s\n' % str(R_dic['actOutFunc2kernel']), log_fileout)

    if str.lower(R_dic['model2kernel']) == 'fourier_dnn':
        MGKN_tools.log_string('The frequency for kernel-model: %s\n' % str(R_dic['freq2kernel']), log_fileout)

    MGKN_tools.log_string('The resolution for train-data: %s\n' % str(R['mesh_number2train']), log_fileout)
    MGKN_tools.log_string(
        'The point_num(=res*res) for train-data: %s\n' % str(R['mesh_number2train'] * R['mesh_number2train']),
        log_fileout)

    MGKN_tools.log_string('The num of sampling point for train-data: %s\n' % str(R['numSamples2train']), log_fileout)

    MGKN_tools.log_string('The resolution for test-data: %s\n' % str(R['mesh_number2test']), log_fileout)
    MGKN_tools.log_string(
        'The point_num(=res*res) for test-data: %s\n' % str(R['mesh_number2test'] * R['mesh_number2test']), log_fileout)

    MGKN_tools.log_string('Loss function: L2 loss\n', log_fileout)
    if (R_dic['optimizer_name']).title() == 'Adam':
        MGKN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        MGKN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']),
                              log_fileout)

    MGKN_tools.log_string('Init learning rate: %s\n' % str(R_dic['init_learning_rate']), log_fileout)

    MGKN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    if R_dic['activate_stop'] != 0:
        MGKN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        MGKN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']),
                              log_fileout)

def print_and_log_train_one_epoch(i_epoch, run_time, tmp_lr, pwb, loss_tmp, train_mse_tmp, train_res_tmp, log_out=None):
    # 将运行结果打印出来
    print('train epoch: %d, time: %.3f' % (i_epoch, run_time))
    print('learning rate: %f' % tmp_lr)
    print('weights and biases with  penalty: %f' % pwb)
    print('loss for training: %.10f' % loss_tmp)
    print('solution mean square error for training: %.10f' % train_mse_tmp)
    print('solution residual error for training: %.10f\n' % train_res_tmp)

    MGKN_tools.log_string('train epoch: %d,time: %.3f' % (i_epoch, run_time), log_out)
    MGKN_tools.log_string('learning rate: %f' % tmp_lr, log_out)
    MGKN_tools.log_string('weights and biases with  penalty: %f' % pwb, log_out)
    MGKN_tools.log_string('loss for training: %.10f' % loss_tmp, log_out)
    MGKN_tools.log_string('solution mean square error for training: %.10f' % train_mse_tmp, log_out)
    MGKN_tools.log_string('solution residual error for training: %.10f\n' % train_res_tmp, log_out)


class MscaleGKN(object):
    def __init__(self, in_dim2input=2, out_dim2input=1, hidden2input=None, name2Input_Model='DNN',
                 actInName2input='tanh', actName2input='tanh', actOutName2input='linear', in_dim2kernel=4,
                 outdim2kernel=1, hidden2kernel=None, name2Kernel_Model='DNN', actInName2Kernel='linear',
                 actName2Kernel='tanh', actOutName2Kernel='linear', opt2regular_WB='L2', it_neighbor2train=10,
                 it_neighbor2test=20, type2numeric='float32'):
        super(MscaleGKN, self).__init__()
        self.model2input = DNN_Class_base.Dense_Net(
            indim=in_dim2input, outdim=out_dim2input, hidden_units=hidden2input, name2Model=name2Input_Model,
            actName2in=actInName2input, actName=actName2input, actName2out=actOutName2input, scope2W='W2input',
            scope2B='B2input', type2float=type2numeric)

        self.model2kernel = DNN_Class_base.Dense_Fourier_Net(
            indim=in_dim2kernel, outdim=outdim2kernel, hidden_units=hidden2kernel,
            name2Model=name2Kernel_Model, actName2in=actInName2Kernel, actName=actName2Kernel,
            actName2out=actOutName2Kernel, scope2W='W2kernel', scope2B='B2kernel', type2float=type2numeric)

        if type2numeric == 'float32':
            self.float_type = tf.float32
        elif type2numeric == 'float64':
            self.float_type = tf.float64
        elif type2numeric == 'float16':
            self.float_type = tf.float16
        self.opt2regular_WB = opt2regular_WB
        self.it_nei2train = it_neighbor2train
        self.it_nei2test = it_neighbor2test

        self.mat2cen = tf.constant([[1, 0, 0, 0],
                                    [0, 1, 0, 0]], dtype=self.float_type)
        self.mat2nei = tf.constant([[0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=self.float_type)
        self.mat_repeat2it2train = tf.ones([it_neighbor2train, 1], dtype=self.float_type)

        self.mat_repeat2it2test = tf.ones([it_neighbor2test, 1], dtype=self.float_type)

    def train_GKN(self, input_points=None, freq2input=None, freq2kernel=None, Utrue=None, Aeps=None):
        adj_matrix = DNN_Class_base.pairwise_distance(input_points)
        knn_idx = DNN_Class_base.knn_includeself(adj_matrix, k=self.it_nei2train)  # indexes (num_points, k_neighbors)

        # obtaining the coord of neighbors according to the corresponding index, then obtaining edge-feature
        neighbors = tf.gather(input_points, knn_idx, axis=0)          # coord (num_points, k_neighbors, dim2point)
        expand_input = tf.expand_dims(input_points, axis=-2)          # (num_points,dim2point)->(num_points,1,dim2point)
        centroid = tf.matmul(self.mat_repeat2it2train, expand_input)  # (num_points, k_neighbors, dim2point)
        edges_feature = centroid - neighbors                          # (num_points, k_neighbors, dim2point)

        # calculating the wight-coefficients of neighbors by edge     # (num_points, 1, k_neighbor)
        attend_coeff2nei = DNN_Class_base.cal_attends2neighbors(edges_feature, dis_model='L2')

        xy_axy = tf.concat([input_points, Aeps], axis=-1)     # (num_points, dim+1)
        VNN = self.model2input(xy_axy, scale=freq2input)      # (num_points, 1)
        VNN_neighbors = tf.gather(VNN, knn_idx, axis=0)       # (num_points, k_neighbors,1)

        centroid_neighbors = tf.matmul(centroid, self.mat2cen) + tf.matmul(neighbors, self.mat2nei)
        # (num_points, k_neighbors, 2*dim2xy)-->(num_points, k_neighbors, 1)
        kernel_matrix = self.model2kernel(centroid_neighbors, scale=freq2kernel)

        kernel_matmul_VNN_neighbors = tf.multiply(kernel_matrix, VNN_neighbors)  # (num_points,k_neighbors,1)

        # aggregating neighbors by wight-coefficient
        attend_solu = tf.matmul(attend_coeff2nei, kernel_matmul_VNN_neighbors)

        # remove the dimension with 1 (num_points, 1)
        UGNN = tf.squeeze(attend_solu, axis=-2)

        loss_u = tf.reduce_mean(tf.square(UGNN - Utrue))
        return loss_u, UGNN

    def get_regular_WB(self):
        regular_sum2WB_input = self.model2input.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        regular_sum2WB_kernnel = self.model2kernel.get_regular_sum2WB(regular_model=self.opt2regular_WB)
        RWB = regular_sum2WB_input + regular_sum2WB_kernnel
        return RWB

    def elvaute_GKN(self, input_points=None, freq2input=None, freq2kernel=None, Aeps=None):
        adj_matrix = DNN_Class_base.pairwise_distance(input_points)
        knn_idx = DNN_Class_base.knn_includeself(adj_matrix, k=self.it_nei2test)  # indexes (num_points, k_neighbors)

        # obtaining the coord of neighbors according to the corresponding index, then obtaining edge-feature
        neighbors = tf.gather(input_points, knn_idx, axis=0)          # coord (num_points, k_neighbors, dim2point)
        expand_input = tf.expand_dims(input_points, axis=-2)          # (num_points,dim2point)->(num_points,1,dim2point)
        centroid = tf.matmul(self.mat_repeat2it2test, expand_input)   # (num_points, k_neighbors, dim2point)
        edges_feature = centroid - neighbors                          # (num_points, k_neighbors, dim2point)

        # calculating the wight-coefficients of neighbors by edge
        attend_coeff2nei = DNN_Class_base.cal_attends2neighbors(edges_feature, dis_model='L2')  # (num_points, 1, k_neighbor)

        xy_axy = tf.concat([input_points, Aeps], axis=-1)             # (num_points, dim+1)
        VNN = self.model2input(xy_axy, scale=freq2input)              # (num_points, 1)
        VNN_neighbors = tf.gather(VNN, knn_idx, axis=0)               # (num_points, k_neighbors,1)

        centroid_neighbors = tf.matmul(centroid, self.mat2cen) + tf.matmul(neighbors, self.mat2nei)
        # (num_points, k_neighbors, 2*dim2xy)-->(num_points, k_neighbors, 1)
        kernel_matrix = self.model2kernel(centroid_neighbors, scale=freq2kernel)

        kernel_matmul_VNN_neighbors = tf.multiply(kernel_matrix, VNN_neighbors)  # (num_points,k_neighbors,1)

        # aggregating neighbors by wight-coefficient
        attend_solu = tf.matmul(attend_coeff2nei, kernel_matmul_VNN_neighbors)

        # remove the dimension with 1 (num_points, 1)
        UGNN = tf.squeeze(attend_solu, axis=-2)
        return UGNN


def solve_multiScale_operator(R):
    log_out_path = R['FolderName']
    if not os.path.exists(log_out_path):
        os.mkdir(log_out_path)
    logfile_name = '%s%s.txt' % ('log2', 'train')
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')
    dictionary_out2file(R, log_fileout)

    batch2train = R['batch2train']
    batch2test = R['batch2test']

    res2train = R['mesh_number2train']
    res2test = R['mesh_number2test']

    point_num2train = res2train * res2train
    point_num2test = res2test * res2test
    numSamples = R['numSamples2train']

    penalty2WB = R['regular_weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    mgkn = MscaleGKN(in_dim2input=input_dim+1, out_dim2input=out_dim, hidden2input=R['hiddens2input'],
                     name2Input_Model=R['model2input'], actInName2input=R['actInFunc2input'],
                     actName2input=R['actFunc2input'], actOutName2input=R['actOutFunc2input'],
                     in_dim2kernel=2*input_dim, outdim2kernel=out_dim, hidden2kernel=R['hiddens2kernel'],
                     name2Kernel_Model=R['model2kernel'], actInName2Kernel=R['actInFunc2kernel'],
                     actName2Kernel=R['actFunc2kernel'], actOutName2Kernel=R['actOutFunc2kernel'],
                     opt2regular_WB=R['regular_weight_model'], it_neighbor2train=R['num_neighbor2train'],
                     it_neighbor2test=R['num_neighbor2test'])

    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.variable_scope('vscope', reuse=tf.AUTO_REUSE):
            XY2train = tf.placeholder(tf.float32, name='XY2train', shape=[point_num2train, input_dim])
            U2train = tf.placeholder(tf.float32, name='U2train', shape=[batch2train, point_num2train, 1])
            Aeps2train = tf.placeholder(tf.float32, name='Aeps2train', shape=[batch2train, point_num2train, 1])
            Indexes2train = tf.placeholder(tf.int32, name='indexes2train', shape=[numSamples, ])

            XY2test = tf.placeholder(tf.float32, name='XY2test', shape=[point_num2test, input_dim])
            U2test = tf.placeholder(tf.float32, name='U2test', shape=[point_num2test, 1])
            Aeps2test = tf.placeholder(tf.float32, name='Aeps2test', shape=[point_num2test, 1])

            in_learning_rate = tf.placeholder_with_default(input=1e-5, shape=[], name='lr')

            train_mses = 0
            train_rels = 0
            loss2U = 0
            for i_batch in range(batch2train):
                train_points = tf.gather(XY2train, axis=0, indices=Indexes2train)
                train_U = tf.reshape(U2train[i_batch, :, :], [-1, 1])
                sample_train_U = tf.gather(train_U, axis=0, indices=Indexes2train)
                train_Aeps = tf.reshape(Aeps2train[i_batch, :, :], [-1, 1])
                sample_train_A = tf.gather(train_Aeps, axis=0, indices=Indexes2train)
                loss_epoch, UMGKN2train = mgkn.train_GKN(input_points=train_points, freq2input=R['freq2input'],
                                                         freq2kernel=R['freq2kernel'], Utrue=sample_train_U,
                                                         Aeps=sample_train_A)
                loss2U = loss2U + loss_epoch

                mse2train = tf.reduce_mean(tf.square(UMGKN2train - sample_train_U))
                rel2train = mse2train/tf.reduce_mean(tf.square(sample_train_U))
                train_mses = train_mses + mse2train/batch2train
                train_rels = train_rels + rel2train/batch2train

            PWB = penalty2WB*mgkn.get_regular_WB()
            loss = loss2U/batch2train + PWB
            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

            UMGKN2test = mgkn.elvaute_GKN(input_points=XY2test, freq2input=R['freq2input'], freq2kernel=R['freq2kernel'],
                                          Aeps=Aeps2test)
            test_mse = tf.reduce_mean(tf.square(UMGKN2test - U2test))
            test_rel = test_mse / tf.reduce_mean(tf.square(U2test))

    # 载入数据
    if R['PDE_type'] == 'pLaplace_implicit':
        # fileName2data = 'data/' + 'res258_a4_f1' + str('.mat')
        if R['equa_name'] == 'multi_scale2D_2':
            fileName2train_data = 'data/' + 'res' + str(res2train) + '_a2_f1_train' + str('.mat')
            fileName2test_data = 'data/' + 'res' + str(res2test) + '_a2_f1_test' + str('.mat')
        elif R['equa_name'] == 'multi_scale2D_3':
            fileName2train_data = 'data/' + 'res' + str(res2train) + '_a3_f1_train' + str('.mat')
            fileName2test_data = 'data/' + 'res' + str(res2test) + '_a3_f1_test' + str('.mat')
        elif R['equa_name'] == 'multi_scale2D_4':
            fileName2train_data = 'data/' + 'res' + str(res2train) + '_a4_f1_train' + str('.mat')
            fileName2test_data = 'data/' + 'res' + str(res2test) + '_a4_f1_test' + str('.mat')

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

    t0 = time.time()
    loss_all, train_mse_all, train_rel_all = [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all = [], []
    test_epoch = []

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    tmp_lr = R['init_learning_rate']
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i_epoch in range(R['max_epoch'] + 1):
            tmp_lr = tmp_lr * (1 - lr_decay)
            indexes2train = np.random.randint(point_num2train, size=numSamples)
            _, loss_tmp, train_mse_tmp, train_rel_tmp, pwb = sess.run(
                [train_my_loss, loss, train_mses, train_rels, PWB],
                feed_dict={XY2train: trans_mesh2train, U2train: Train_U, Aeps2train: Train_Aeps,
                           Indexes2train: indexes2train, in_learning_rate: tmp_lr})

            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_rel_tmp)

            if i_epoch % 100 == 0:
                test_epoch.append(i_epoch / 1000)
                run_times = time.time() - t0
                print_and_log_train_one_epoch(i_epoch, run_times, tmp_lr, pwb, loss_tmp, train_mse_tmp, train_rel_tmp,
                                              log_out=log_fileout)

                unn2test, test_mse_tmp, test_rel_tmp = sess.run(
                    [UMGKN2test, test_mse, test_rel], feed_dict={XY2test: test_trans_mesh, Aeps2test: Test_Aeps,
                                                                 U2test: Test_U})
                test_mse_all.append(test_mse_tmp)
                test_rel_all.append(test_rel_tmp)
                MGKN_tools.print_and_log_test_one_epoch(test_mse_tmp, test_rel_tmp, log_out=log_fileout)

    # ------------------- save the testing results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat(loss_all, lossName='loss', outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=R['actInFunc2input'], outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=R['actInFunc2input'], seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    saveData.save_2testSolus2mat(Test_U, unn2test, actName='utrue', actName1=R['actInFunc2input'],
                                 outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=R['actInFunc2input'], outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=R['actInFunc2input'],
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)


if __name__ == "__main__":
    R={}
    # -------------------------------------- CPU or GPU 选择 -----------------------------------------------
    R['gpuNo'] = 1
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 文件保存路径设置
    # store_file = 'Laplace2D'
    store_file = 'pLaplace2D'
    # store_file = 'Boltzmann2D'
    # store_file = 'Convection2D'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])                     # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    R['activate_stop'] = int(step_stop_flag)
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    R['max_epoch'] = 200000
    if 0 != R['activate_stop']:
        epoch_stop = input('please input a stop epoch:')
        R['max_epoch'] = int(epoch_stop)

    R['input_dim'] = 2                # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1               # 输出维数

    # ---------------------------- Setup of multi-scale problem-------------------------------
    if store_file == 'Laplace2D':
        R['PDE_type'] = 'general_Laplace'
        R['equa_name'] = 'PDE1'
        # R['equa_name'] = 'PDE2'
        # R['equa_name'] = 'PDE3'
        # R['equa_name'] = 'PDE4'
        # R['equa_name'] = 'PDE5'
        # R['equa_name'] = 'PDE6'
        # R['equa_name'] = 'PDE7'
    elif store_file == 'pLaplace2D':
        R['PDE_type'] = 'pLaplace_implicit'
        # R['equa_name'] = 'multi_scale2D_1'
        # R['equa_name'] = 'multi_scale2D_2'
        # R['equa_name'] = 'multi_scale2D_3'
        R['equa_name'] = 'multi_scale2D_4'

        # R['laplace_opt'] = 'pLaplace_explicit'
        # R['equa_name'] = 'multi_scale2D_6'
        # R['equa_name'] = 'multi_scale2D_7'

    if R['PDE_type'] == 'general_Laplace':
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
    elif R['PDE_type'] == 'pLaplace_implicit':
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2
    elif R['PDE_type'] == 'pLaplace_explicit':
        R['epsilon'] = 0.1
        R['order2pLaplace_operator'] = 2

    # R['getData_model2train'] = 'random_generate'
    R['getData_model2train'] = 'load_data'

    # R['getData_model2test'] = 'random_generate'
    R['getData_model2test'] = 'load_data'

    R['mesh_number2train'] = 66
    R['mesh_number2test'] = 66

    R['num_neighbor2train'] = 250
    R['num_neighbor2test'] = 100

    R['batch2train'] = 50
    R['batch2test'] = 1

    R['numSamples2train'] = 1000

    # ---------------------------- Setup of DNN -------------------------------
    # R['regular_weight_model'] = 'L0'
    # R['regular_weight_model'] = 'L1'
    R['regular_weight_model'] = 'L2'
    # R['regular_weight_biases'] = 0.000        # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.001      # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0005       # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.0001     # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.00005  # Regularization parameter for weights
    # R['regular_weight_biases'] = 0.00001  # Regularization parameter for weights
    R['regular_weight_biases'] = 0.000001  # Regularization parameter for weights

    R['init_learning_rate'] = 2e-4                        # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['optimizer_name'] = 'Adam'                          # 优化器

    R['model2input'] = 'DNN'
    # R['model2input'] = 'Scale_DNN'
    # R['model2input'] = 'Fourier_DNN'

    # R['model2kernel'] = 'DNN'
    # R['model2kernel'] = 'Scale_DNN'
    R['model2kernel'] = 'Fourier_DNN'

    if R['model2input'] == 'DNN':
        # R['hiddens2input'] = (160, 100, 100, 50)
        R['hiddens2input'] = (300, 200, 100, 100, 50)
    elif R['model2input'] == 'Fourier_DNN':
        # R['hiddens2input'] = (80, 100, 100, 50)
        R['hiddens2input'] = (150, 200, 100, 100, 50)

    if R['model2kernel'] == 'DNN':
        R['hiddens2kernel'] = (160, 200, 100, 100, 50)
    else:
        R['hiddens2kernel'] = (80, 200, 100, 100, 50)

    R['actInFunc2input'] = 'tanh'
    # R['actInFunc2input'] = 'sin'
    # R['actInFunc2input'] = 's2relu'

    # R['actFunc2input'] = 'relu'
    # R['actFunc2input'] = 'leaky_relu'
    R['actFunc2input'] = 'tanh'
    # R['actFunc2input'] = 'sigmoid'
    # R['actFunc2input'] = 'srelu'
    # R['actFunc2input'] = 's2relu'

    R['actOutFunc2input'] = 'linear'
    # R['actOutFunc2input'] = 'relu'
    # R['actOutFunc2input'] = 'tanh'
    # R['actOutFunc2input'] = 'sigmoid'

    # R['actInFunc2kernel'] = 'relu'
    # R['actInFunc2kernel'] = 'srelu'
    # R['actInFunc2kernel'] = 's2relu'
    # R['actInFunc2kernel'] = 'sin'
    R['actInFunc2kernel'] = 'tanh'

    # R['actFunc2kernel'] = 'relu'
    # R['actFunc2kernel'] = 'leaky_relu'
    R['actFunc2kernel'] = 'tanh'
    # R['actFunc2kernel'] = 'sin'
    # R['actFunc2kernel'] = 'srelu'
    # R['actFunc2kernel'] = 's2relu'
    # R['actFunc2kernel'] = 'sigmoid'

    # R['actOutFunc2kernel'] = 'linear'
    # R['actOutFunc2kernel'] = 'relu'
    R['actOutFunc2kernel'] = 'tanh'
    # R['actOutFunc2kernel'] = 'gauss'
    # R['actOutFunc2kernel'] = 'sin'
    # R['actOutFunc2kernel'] = 's2relu'
    # R['actOutFunc2kernel'] = 'sigmoid'

    R['freq2input'] = np.array([1])
    # R['freq2input'] = np.array([1, 1, 2, 3, 4, 5])
    # R['freq2input'] = np.arange(1, 10)
    # R['freq2kernel'] = np.array([1])
    # R['freq2kernel'] = np.arange(1, 30)
    R['freq2kernel'] = np.arange(1, 50)
    # R['freq2kernel'] = np.arange(1, 100)

    solve_multiScale_operator(R)

    # 函数网络的输入既包括点，也包括A(x)，Kernel网络的输入只有点作为输入
    # 函数网络选为 一般的DNN，Kernel网络选为Fourier DNN，频率选为[1,..,60], DNN 的激活函数选为 tanh,输出激活函数选为 linear
    # Fourier_DNN的激活函数和输出激活均选为 tanh
    # 这种策略最好，但是对于多尺度问题, 与真解的高度上差些, 需要进一步研究

    # 函数网络选为Fourier DNN, 频率选为[1,....,1]，Kernel网络选为Fourier DNN，频率选为[1,..,60], 激活函数都选为tanh
    # 这种策略可行，但是与真解的高度上差些

    # 函数网络选为Fourier DNN, 频率选为[1,....,1]，激活函数选为s2relu, Kernel网络选为Fourier DNN，频率选为[1,..,60], 激活函数选为tanh
    # 这种策略可行，但是与真解的高度上差些,好于全 tanh

    # 函数网络选为Fourier DNN, 频率选为[1,....,1]，激活函数选为tanh, Kernel网络选为Fourier DNN，频率选为[1,..,60], 激活函数选为s2relu
    # 这种策略可行，但是与真解的高度上差些,好于全 tanh

    # 总结:kernel 网络可选择为Fourier DNN，函数网络选择一般的DNN网络比较好

