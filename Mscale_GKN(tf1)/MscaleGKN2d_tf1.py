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
    MGKN_tools.log_string('Out activate function for the input-model: %s\n' % str(R_dic['actOutFunc2input']), log_fileout)
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

    MGKN_tools.log_string('The batch-size for each training: %s\n' % str(R['batch2train']), log_fileout)
    MGKN_tools.log_string('The num of sampling point for each-batch-train-data: %s\n' % str(R['numSamples2train']), log_fileout)
    MGKN_tools.log_string('The num of neighbors for training process: %s\n' % str(R['num_neighbor2train']), log_fileout)

    MGKN_tools.log_string('The model for getting test-data: %s\n' % R['getData_model2test'], log_fileout)
    MGKN_tools.log_string('The resolution for test-data: %s\n' % str(R['mesh_number2test']), log_fileout)
    MGKN_tools.log_string(
        'The point_num(=res*res) for test-data: %s\n' % str(R['mesh_number2test'] * R['mesh_number2test']), log_fileout)

    MGKN_tools.log_string('The num of neighbors for testing process: %s\n' % str(R['num_neighbor2test']), log_fileout)

    MGKN_tools.log_string('Loss function: L2 loss\n', log_fileout)
    if (R_dic['optimizer_name']).title() == 'Adam':
        MGKN_tools.log_string('optimizer:%s\n' % str(R_dic['optimizer_name']), log_fileout)
    else:
        MGKN_tools.log_string('optimizer:%s  with momentum=%f\n' % (R_dic['optimizer_name'], R_dic['momentum']), log_fileout)

    MGKN_tools.log_string('Init learning rate: %s\n' % str(R_dic['init_learning_rate']), log_fileout)

    MGKN_tools.log_string('Decay to learning rate: %s\n' % str(R_dic['learning_rate_decay']), log_fileout)

    if R_dic['activate_stop'] != 0:
        MGKN_tools.log_string('activate the stop_step and given_step= %s\n' % str(R_dic['max_epoch']), log_fileout)
    else:
        MGKN_tools.log_string('no activate the stop_step and given_step = default: %s\n' % str(R_dic['max_epoch']), log_fileout)


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
    point_samples2train = R['numSamples2train']

    penalty2WB = R['regular_weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']

    penaltyTP = R['penalty2true_predict']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    hidden2input = R['hiddens2input']
    hidden2Kernel = R['hiddens2kernel']

    flag2Input = 'WB2Input'
    if R['model2input'] == 'Fourier_DNN':
        W2Input, B2Input = DNN_base.Xavier_init_NN_Fourier(input_dim + 1, out_dim, hidden2input, flag2Input)
    else:
        W2Input, B2Input = DNN_base.Xavier_init_NN(input_dim + 1, out_dim, hidden2input, flag2Input)

    flag2Kernel = 'WB2Kernel'
    if R['model2kernel'] == 'Fourier_DNN':
        W2Kernel, B2Kernel = DNN_base.Xavier_init_NN_Fourier(2 * input_dim, out_dim, hidden2Kernel, flag2Kernel)
    else:
        W2Kernel, B2Kernel = DNN_base.Xavier_init_NN(2 * input_dim, out_dim, hidden2Kernel, flag2Kernel)

    mat2cen = tf.constant([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=tf.float32)
    mat2nei = tf.constant([[0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=tf.float32)
    mat_repeat2it2train = tf.ones([R['num_neighbor2train'], 1], dtype=tf.float32)

    mat_repeat2it2test = tf.ones([R['num_neighbor2test'], 1], dtype=tf.float32)

    # training our model
    global_steps = tf.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope2train', reuse=tf.compat.v1.AUTO_REUSE):
            XY2train = tf.compat.v1.placeholder(tf.float32, name='XY2train', shape=[point_samples2train, input_dim])
            U2train = tf.compat.v1.placeholder(tf.float32, name='U2train', shape=[batch2train, point_samples2train, 1])
            Aeps2train = tf.compat.v1.placeholder(tf.float32, name='Aeps2train', shape=[batch2train, point_samples2train, 1])
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            train_mses = 0
            train_rels = 0
            loss2U = 0
            for i_batch in range(batch2train):
                sample_train_U = U2train[i_batch, :]
                sample_train_A = Aeps2train[i_batch, :]

                adj_matrix2train = DNN_base.pairwise_distance(XY2train)
                knn_idx2train = DNN_base.knn_includeself(adj_matrix2train, k=R['num_neighbor2train'])
                neighbors2train = tf.gather(XY2train, knn_idx2train, axis=0)
                expand_train_point = tf.expand_dims(XY2train, axis=-2)
                centroid2train = tf.matmul(mat_repeat2it2train, expand_train_point)
                edges2train = centroid2train - neighbors2train

                # calculating the wight-coefficients of neighbors by edge     # (num_points, 1, k_neighbor)
                attend_coeff2train = DNN_base.cal_attends2neighbors(edges2train, dis_model='L2')
                XY_Aeps2train = tf.concat([XY2train, sample_train_A], axis=-1)
                if R['model2input'] == 'DNN':
                    VNN2train = DNN_base.DNN(XY_Aeps2train, W2Input, B2Input, hidden2input,
                                             activateIn_name=R['actInFunc2input'], activate_name=R['actFunc2input'],
                                             activateOut_name=R['actOutFunc2input'])
                elif R['model2input'] == 'Scale_DNN':
                    VNN2train = DNN_base.DNN_scale(
                        XY_Aeps2train, W2Input, B2Input, hidden2input, R['freq2input'],
                        activateIn_name=R['actInFunc2input'], activate_name=R['actFunc2input'],
                        activateOut_name=R['actOutFunc2input'])
                elif R['model2input'] == 'Fourier_DNN':
                    VNN2train = DNN_base.DNN_FourierBase(
                        XY_Aeps2train, W2Input, B2Input, hidden2input, R['freq2input'], activate_name=R['actFunc2input'],
                        activateOut_name=R['actOutFunc2input'], sFourier=R['sfourier'])

                centroid_neighbors2train = tf.matmul(centroid2train, mat2cen) + tf.matmul(neighbors2train, mat2nei)
                if R['model2kernel'] == 'DNN':
                    kernel2train = DNN_base.DNN(centroid_neighbors2train, W2Kernel, B2Kernel, hidden2Kernel,
                                                activateIn_name=R['actInFunc2kernel'], activate_name=R['actFunc2kernel'],
                                                activateOut_name=R['actOutFunc2kernel'])
                elif R['model2kernel'] == 'Scale_DNN':
                    kernel2train = DNN_base.DNN_scale(
                        centroid_neighbors2train, W2Kernel, B2Kernel, hidden2Kernel, R['freq2kernel'],
                        activateIn_name=R['actInFunc2kernel'], activate_name=R['actFunc2kernel'],
                        activateOut_name=R['actOutFunc2kernel'])
                elif R['model2kernel'] == 'Fourier_DNN':
                    kernel2train = DNN_base.DNN_FourierBase(
                        centroid_neighbors2train, W2Kernel, B2Kernel, hidden2Kernel, R['freq2kernel'],
                        activate_name=R['actFunc2kernel'], activateOut_name=R['actOutFunc2kernel'],
                        sFourier=R['sfourier'])

                VNN_nei2train = tf.gather(VNN2train, knn_idx2train, axis=0)
                kernel_VNN_nei2train = tf.multiply(kernel2train, VNN_nei2train)

                # aggregating neighbors by wight-coefficient
                attend_solu2train = tf.matmul(attend_coeff2train, kernel_VNN_nei2train)

                # remove the dimension with 1 (num_points, 1)
                UMGKN2train = tf.squeeze(attend_solu2train, axis=-2)

                loss2train_U = tf.reduce_mean(tf.square(penaltyTP*UMGKN2train - penaltyTP*sample_train_U))
                loss2U = loss2U + loss2train_U/batch2train

                mse2train = tf.reduce_mean(tf.square(penaltyTP*UMGKN2train - penaltyTP*sample_train_U))
                rel2train = mse2train/tf.reduce_mean(tf.square(penaltyTP*sample_train_U))
                train_mses = train_mses + mse2train/(penaltyTP*penaltyTP*batch2train)
                train_rels = train_rels + rel2train/batch2train

            if R['regular_weight_model'] == 'L1':
                regular_WB2Input = DNN_base.regular_weights_biases_L1(W2Input, B2Input)  # 正则化权重和偏置 L1正则化
                regular_WB2Kernel = DNN_base.regular_weights_biases_L1(W2Kernel, B2Kernel)  # 正则化权重和偏置 L1正则化
            elif R['regular_weight_model'] == 'L2':
                regular_WB2Input = DNN_base.regular_weights_biases_L2(W2Input, B2Input)  # 正则化权重和偏置 L2正则化
                regular_WB2Kernel = DNN_base.regular_weights_biases_L2(W2Kernel, B2Kernel)  # 正则化权重和偏置 L2正则化
            else:
                regular_WB2Input = tf.constant(0.0)
                regular_WB2Kernel = tf.constant(0.0)
            PWB = penalty2WB*(regular_WB2Input + regular_WB2Kernel)
            loss = loss2U + PWB
            my_optimizer = tf.train.AdamOptimizer(in_learning_rate)
            train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)

    # testing our model
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope2test', reuse=tf.compat.v1.AUTO_REUSE):
            XY2test = tf.compat.v1.placeholder(tf.float32, name='XY2test', shape=[point_num2test, input_dim])
            U2test = tf.compat.v1.placeholder(tf.float32, name='U2test', shape=[point_num2test, 1])
            Aeps2test = tf.compat.v1.placeholder(tf.float32, name='Aeps2test', shape=[point_num2test, 1])

            adj_matrix2test = DNN_base.pairwise_distance(XY2test)
            knn_idx2test = DNN_base.knn_includeself(adj_matrix2test, k=R['num_neighbor2test'])
            neighbors2test = tf.gather(XY2test, knn_idx2test, axis=0)
            expand_test_point = tf.expand_dims(XY2test, axis=-2)
            centroid2test = tf.matmul(mat_repeat2it2test, expand_test_point)
            edges2test = centroid2test - neighbors2test

            # calculating the wight-coefficients of neighbors by edge
            attend_coeff2test = DNN_base.cal_attends2neighbors(edges2test, dis_model='L2')
            XY_Aeps2test = tf.concat([XY2test, Aeps2test], axis=-1)
            if R['model2input'] == 'DNN':
                VNN2test = DNN_base.DNN(XY_Aeps2test, W2Input, B2Input, hidden2input,
                                        activateIn_name=R['actInFunc2input'], activate_name=R['actFunc2input'],
                                        activateOut_name=R['actOutFunc2input'])
            elif R['model2input'] == 'Scale_DNN':
                VNN2test = DNN_base.DNN_scale(
                    XY_Aeps2test, W2Input, B2Input, hidden2input, R['freq2input'], activateIn_name=R['actInFunc2input'],
                    activate_name=R['actFunc2input'], activateOut_name=R['actOutFunc2input'])
            elif R['model2input'] == 'Fourier_DNN':
                VNN2test = DNN_base.DNN_FourierBase(
                    XY_Aeps2test, W2Input, B2Input, hidden2input, R['freq2input'], activate_name=R['actFunc2input'],
                    activateOut_name=R['actOutFunc2input'], sFourier=R['sfourier'])

            centroid_neighbors2test = tf.matmul(centroid2test, mat2cen) + tf.matmul(neighbors2test, mat2nei)
            if R['model2kernel'] == 'DNN':
                kernel2test = DNN_base.DNN(centroid_neighbors2test, W2Kernel, B2Kernel, hidden2Kernel,
                                           activateIn_name=R['actInFunc2kernel'], activate_name=R['actFunc2kernel'],
                                           activateOut_name=R['actOutFunc2kernel'])
            elif R['model2kernel'] == 'Scale_DNN':
                kernel2test = DNN_base.DNN_scale(
                    centroid_neighbors2test, W2Kernel, B2Kernel, hidden2Kernel, R['freq2kernel'],
                    activateIn_name=R['actInFunc2kernel'], activate_name=R['actFunc2kernel'],
                    activateOut_name=R['actOutFunc2kernel'])
            elif R['model2kernel'] == 'Fourier_DNN':
                kernel2test = DNN_base.DNN_FourierBase(
                    centroid_neighbors2test, W2Kernel, B2Kernel, hidden2Kernel, R['freq2kernel'],
                    activate_name=R['actFunc2kernel'], activateOut_name=R['actOutFunc2kernel'],
                    sFourier=R['sfourier'])

            VNN_nei2test = tf.gather(VNN2test, knn_idx2test, axis=0)
            kernel_VNN_nei2test = tf.multiply(kernel2test, VNN_nei2test)

            # aggregating neighbors by wight-coefficient
            attend_solu2test = tf.matmul(attend_coeff2test, kernel_VNN_nei2test)

            # remove the dimension with 1 (num_points, 1)
            UMGKN2test = tf.squeeze(attend_solu2test, axis=-2)

            test_mse = tf.reduce_mean(tf.square(UMGKN2test - U2test))
            test_rel = test_mse / tf.reduce_mean(tf.square(U2test))

            test_l2Err_fenzi = tf.reduce_sum(tf.square(UMGKN2test - U2test))
            test_l2Err_fenmu = tf.reduce_sum(tf.square(U2test))
            test_l2rel = tf.sqrt(test_l2Err_fenzi) / tf.sqrt(test_l2Err_fenmu)

    # Load the training and testing dataSet
    if R['PDE_type'] == 'pLaplace_implicit':
        # For example fileName2data = 'data/' + 'res258_a4_f1' + str('.mat')
        if R['equa_name'] == 'multi_scale2D_2':
            fileName2train_data = 'data/' + 'res' + str(res2train) + '_a2_f1_train' + str('.mat')
            fileName2test_data = 'data/' + 'res' + str(res2test) + '_a2_f1_test' + str('.mat')
        elif R['equa_name'] == 'multi_scale2D_3':
            fileName2train_data = 'data/' + 'res' + str(res2train) + '_a3_f1_train' + str('.mat')
            fileName2test_data = 'data/' + 'res' + str(res2test) + '_a3_f1_test' + str('.mat')
        elif R['equa_name'] == 'multi_scale2D_4':
            fileName2train_data = 'data/sample_data2fine_mesh/' + 'res' + str(res2train) + '_a4_f1_train' + str('.mat')
            fileName2test_data = 'data/sample_data2fine_mesh/' + 'res' + str(res2test) + '_a4_f1_test' + str('.mat')
            fileName2test_index = 'data/sample_data2fine_mesh/' + 'disorder_index' + str(res2test) + str('.mat')

    train_dataSet = matData2pLaplace.load_Matlab_data(fileName2train_data)
    train_data_Aeps = train_dataSet['meshA']
    train_data_solu = train_dataSet['meshU']
    train_data_mesh = train_dataSet['meshXY']

    shape2data = np.shape(train_data_Aeps)

    trainData2Aeps = np.reshape(train_data_Aeps, newshape=[shape2data[0], res2train * res2train])
    trainData2solu = np.reshape(train_data_solu, newshape=[shape2data[0], res2train * res2train])
    mesh2train = np.transpose(train_data_mesh, (1, 0))

    # 对数据进行归一化处理
    trainData2Aeps_normalizer = DNN_Class_base.np_GaussianNormalizer(trainData2Aeps)
    normalize_trainData2Aeps = trainData2Aeps_normalizer.encode(trainData2Aeps)

    data2indexes = matData2pLaplace.load_Matlab_data(fileName2test_index)
    disorder_index2test_data = np.reshape(data2indexes['disorder_index'], newshape=(-1))
    if 'load_test_data' == R['getData_model2test']:
        max_batch_size2train = shape2data[0]    # 训练集batch size的大小：选为全部batch作为训练集
        test_dataSet = matData2pLaplace.load_Matlab_data(fileName2test_data)
        test_data_Aeps = test_dataSet['meshA']
        test_data_solu = test_dataSet['meshU']
        test_data_mesh = test_dataSet['meshXY']

        testData2Aeps = np.reshape(test_data_Aeps, newshape=[1, res2test * res2test])
        testData2solu = np.reshape(test_data_solu, newshape=[1, res2test * res2test])

        testData2Aeps_normlizer = DNN_Class_base.np_GaussianNormalizer(testData2Aeps)
        normalize_testData2Aeps = testData2Aeps_normlizer.encode(testData2Aeps)

        shuffle_norm_testData2Aeps = normalize_testData2Aeps[:, disorder_index2test_data]
        shuffle_testData2solu = testData2solu[:, disorder_index2test_data]
        shuffle_testData2mesh = test_data_mesh[:, disorder_index2test_data]
    elif 'split_train_data' == R['getData_model2test']:
        max_batch_size2train = 400   # 训练集batch size的大小：选为前400个数据条目作为训练集

        # temp = np.reshape(normalize_trainData2Aeps[401, :], newshape=[res2test * res2test, 1])
        # shuffle_temp = temp[disorder_index2test_data, :]

        normalize_testData2Aeps = normalize_trainData2Aeps[450:450+batch2test, :]
        testData2solu = trainData2solu[450:450+batch2test, :]

        shuffle_norm_testData2Aeps = normalize_testData2Aeps[:, disorder_index2test_data]
        shuffle_testData2solu = testData2solu[:, disorder_index2test_data]
        test_data_mesh = train_data_mesh
        shuffle_testData2mesh = test_data_mesh[:, disorder_index2test_data]

    Test_Aeps = np.transpose(shuffle_norm_testData2Aeps, (1, 0))
    Test_U = np.transpose(shuffle_testData2solu, (1, 0))
    mesh2test = np.transpose(shuffle_testData2mesh, (1, 0))

    t0 = time.time()
    loss_all, train_mse_all, train_rel_all = [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all, test_l2rel_all = [], [], []
    test_epoch = []

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    tmp_lr = R['init_learning_rate']
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i_epoch in range(R['max_epoch'] + 1):
            if i_epoch % 100 == 0:
                index2sample_batch = np.random.randint(max_batch_size2train, size=batch2train)
                print('The index for sample_batch to train-data: %s\n' % str(index2sample_batch))
                MGKN_tools.log_string('The index for sample_batch to train-data: %s\n' % str(index2sample_batch), log_fileout)
                sampleBatch2Train_Aeps = normalize_trainData2Aeps[index2sample_batch, :]
                sampleBatch2Train_U = trainData2solu[index2sample_batch, :]

            tmp_lr = tmp_lr * (1 - lr_decay)
            indexes2sample_point = np.random.randint(point_num2train, size=point_samples2train)
            sample_points2train = mesh2train[indexes2sample_point, :]
            samples2sampleBatch_TrainAeps = np.expand_dims(sampleBatch2Train_Aeps[:, indexes2sample_point], axis=-1)
            samples2sampleBatch_TrainU = np.expand_dims(sampleBatch2Train_U[:, indexes2sample_point], axis=-1)
            _, loss_tmp, train_mse_tmp, train_rel_tmp, pwb = sess.run(
                [train_my_loss, loss, train_mses, train_rels, PWB],
                feed_dict={XY2train: sample_points2train, U2train: samples2sampleBatch_TrainU,
                           Aeps2train: samples2sampleBatch_TrainAeps, in_learning_rate: tmp_lr})

            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_rel_tmp)

            if i_epoch % 100 == 0:
                test_epoch.append(i_epoch / 1000)
                run_times = time.time() - t0
                print_and_log_train_one_epoch(i_epoch, run_times, tmp_lr, pwb, loss_tmp, train_mse_tmp, train_rel_tmp,
                                              log_out=log_fileout)

                unn2test, test_mse_tmp, test_rel_tmp, test_l2rel_tmp = sess.run(
                    [UMGKN2test, test_mse, test_rel, test_l2rel], feed_dict={XY2test: mesh2test, Aeps2test: Test_Aeps,
                                                                             U2test: Test_U})
                test_mse_all.append(test_mse_tmp)
                test_rel_all.append(test_rel_tmp)
                test_l2rel_all.append(test_l2rel_tmp)
                MGKN_tools.print_and_log_test1epoch(test_mse_tmp, test_rel_tmp, test_l2rel_tmp, log_out=log_fileout)

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
    saveData.save_testErr2mat(test_l2rel_all, errName='l2Err', outPath=R['FolderName'])
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
        matplotlib.use('Agg')

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 设置当前使用的GPU设备仅为第 0,1 块GPU, 设备名称为'/gpu:0'
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
    R['getData_model2train'] = 'load_train_data'

    # R['getData_model2test'] = 'random_generate'
    # R['getData_model2test'] = 'load_test_data'
    R['getData_model2test'] = 'split_train_data'

    R['mesh_number2train'] = 66
    R['mesh_number2test'] = 66

    # R['mesh_number2train'] = 130
    # R['mesh_number2test'] = 130

    if 130 == R['mesh_number2train']:
        # R['batch2train'] = 3
        # R['batch2train'] = 10
        R['batch2train'] = 15
        R['batch2test'] = 1

        # R['numSamples2train'] = 1000
        # R['numSamples2train'] = 2000
        R['numSamples2train'] = 3000
        # R['numSamples2train'] = 4000

        # R['num_neighbor2train'] = 250
        # R['num_neighbor2train'] = 400
        R['num_neighbor2train'] = 500
        # R['num_neighbor2train'] = 1000
        # R['num_neighbor2train'] = 200
    elif 66 == R['mesh_number2train']:
        # R['batch2train'] = 3
        R['batch2train'] = 20
        R['batch2test'] = 1
        # R['numSamples2train'] = 1000
        R['numSamples2train'] = 1500

        R['num_neighbor2train'] = 150
        # R['num_neighbor2train'] = 200
        # R['num_neighbor2train'] = 250

    if 130 == R['mesh_number2test']:
        # R['num_neighbor2test'] = 100
        # R['num_neighbor2test'] = 200
        # R['num_neighbor2test'] = 250
        # R['num_neighbor2test'] = 400
        # R['num_neighbor2test'] = 700
        R['num_neighbor2test'] = 800
        # R['num_neighbor2test'] = 900
        # R['num_neighbor2test'] = 1000
        # R['num_neighbor2test'] = 1500
    elif 66 == R['mesh_number2test']:
        # R['num_neighbor2test'] = 100
        # R['num_neighbor2test'] = 200
        # R['num_neighbor2test'] = 250
        R['num_neighbor2test'] = 400
        # R['num_neighbor2test'] = 500

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

    R['penalty2true_predict'] = 5

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

    # R['actInFunc2input'] = 'relu'
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
    # R['actOutFunc2kernel'] = 'softplus'

    if R['model2kernel'] == 'Fourier_DNN' and R['actFunc2kernel'] == 'tanh':
        R['sfourier'] = 1.0
    elif R['model2kernel'] == 'Fourier_DNN' and R['actFunc2kernel'] == 's2relu':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    else:
        R['sfourier'] = 1.0

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

