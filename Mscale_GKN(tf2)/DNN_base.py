# -*- coding: utf-8 -*-
"""
Created on 2021.06.17
@author: xi'an Li
"""
import tensorflow as tf
import numpy as np


def pairwise_distance(point_set):
    """Compute pairwise distance of a point cloud.
        Args:
          (x-y)^2 = x^2 - 2xy + y^2
          point_set: tensor (num_points, dims2point)
        Returns:
          pairwise distance: (num_points, num_points)
    """
    point_set_shape = point_set.get_shape().as_list()
    assert(len(point_set_shape)) == 2

    point_set_transpose = tf.transpose(point_set, perm=[1, 0])
    point_set_inner = tf.matmul(point_set, point_set_transpose)
    point_set_inner = -2 * point_set_inner
    point_set_square = tf.reduce_sum(tf.square(point_set), axis=-1, keepdims=True)
    point_set_square_transpose = tf.transpose(point_set_square, perm=[1, 0])
    return point_set_square + point_set_inner + point_set_square_transpose


def knn_includeself(dist_matrix, k=20):
    """Get KNN based on the pairwise distance.
        How to use tf.nn.top_k(): https://blog.csdn.net/wuguangbin1230/article/details/72820627
      Args:
        pairwise distance: (num_points, num_points)
        k: int

      Returns:
        nearest neighbors: (num_points, k)
      """
    neg_dist = -1.0*dist_matrix
    _, nn_idx = tf.nn.top_k(neg_dist, k=k)  # 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    return nn_idx


def knn_excludeself(dist_matrix, k=20):
    """Get KNN based on the pairwise distance.
      Args:
        pairwise distance: (num_points, num_points)
        k: int

      Returns:
        nearest neighbors index: (num_points, k)
      """
    neg_dist = -1.0*dist_matrix
    k_neighbors = k+1
    _, knn_idx = tf.nn.top_k(neg_dist, k=k_neighbors)  # 这个函数的作用是返回 input 中每行最大的 k 个数，并且返回它们所在位置的索引
    nn_idx = knn_idx[:, 1: k_neighbors]
    return nn_idx


def get_kneighbors_3D_4DTensor(point_set, nn_idx):
    """Construct neighbors feature for each point
        Args:
        point_set: (batch_size, num_points, 1, dim)
        nn_idx: (batch_size, num_points, k)
        k: int

        Returns:
        neighbors features: (batch_size, num_points, k, dim)
      """
    og_batch_size = point_set.get_shape().as_list()[0]
    og_num_dims = point_set.get_shape().as_list()[-1]
    point_set = tf.squeeze(point_set)
    if og_batch_size == 1:
        point_set = tf.expand_dims(point_set, 0)
    if og_num_dims == 1:
        point_set = tf.expand_dims(point_set, -1)

    point_set_shape = point_set.get_shape()
    batch_size = point_set_shape[0].value
    num_points = point_set_shape[1].value
    num_dims = point_set_shape[2].value

    idx_ = tf.range(batch_size) * num_points
    idx_ = tf.reshape(idx_, [batch_size, 1, 1])

    point_set_flat = tf.reshape(point_set, [-1, num_dims])
    point_set_neighbors = tf.gather(point_set_flat, nn_idx + idx_)

    return point_set_neighbors


def get_kneighbors_2DTensor(point_set, nn_idx):
    """Construct neighbors feature for each point
        Args:
        point_set: (num_points, dim)
        nn_idx: (num_points, k_num)
        num_points: the number of point
        k_num: the number of neighbor

        Returns:
        neighbors features: (num_points, k_num, dim)
      """
    shape2point_set = point_set.get_shape().as_list()
    assert(len(shape2point_set) == 2)
    point_set_neighbors = tf.gather(point_set, nn_idx)
    return point_set_neighbors


def cal_attend2neighbors(edge_point_set, dis_model='L1'):
    """
        Args:
        edge_point_set:(num_points, k_neighbors, dim2point)
        dis_model:
        return:
        atten_ceof: (num_points, 1, k_neighbors)
    """
    square_edges = tf.square(edge_point_set)                  # (num_points, k_neighbors, dim2point)
    norm2edges = tf.reduce_sum(square_edges, axis=-1)         # (num_points, k_neighbors)
    if str.lower(dis_model) == 'l1':
        norm2edges = tf.sqrt(norm2edges)
    exp_dis = tf.exp(-norm2edges)                             # (num_points, k_neighbors)
    normalize_exp_dis = tf.nn.softmax(exp_dis, axis=-1)
    atten_ceof = tf.expand_dims(normalize_exp_dis, axis=-2)   # (num_points, 1, k_neighbors)
    return atten_ceof


class np_GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(np_GaussianNormalizer, self).__init__()

        self.mean2x = np.mean(x, axis=-1, keepdims=True)
        self.std2x = np.std(x, axis=-1, keepdims=True)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean2x) / (self.std2x * self.std2x + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std2x * self.std2x + self.eps)) + self.mean2x
        return x


# ------------------------------------- my activations ----------------------------------
class my_actFunc(tf.keras.layers.Layer):
    def __init__(self, actName='linear'):
        super(my_actFunc, self).__init__()
        self.actName = actName

    def call(self, x_input):
        if str.lower(self.actName) == 'relu':
            out_x = tf.nn.relu(x_input)
        elif str.lower(self.actName) == 'leaky_relu':
            out_x = tf.nn.leaky_relu(x_input)
        elif str.lower(self.actName) == 'tanh':
            out_x = tf.nn.relu(x_input)
        elif str.lower(self.actName) == 'srelu':
            out_x = tf.nn.relu(x_input)*tf.nn.relu(1-x_input)
        elif str.lower(self.actName) == 's2relu':
            out_x = tf.nn.relu(x_input)*tf.nn.relu(1-x_input)*tf.sin(2*np.pi*x_input)
        elif str.lower(self.actName) == 'elu':
            out_x = tf.nn.elu(x_input)
        elif str.lower(self.actName) == 'sin':
            out_x = tf.sin(x_input)
        elif str.lower(self.actName) == 'gauss':
            out_x = tf.exp(-1.0*x_input*x_input)
        elif str.lower(self.actName) == 'sigmoid':
            out_x = tf.nn.sigmoid(x_input)
        elif str.lower(self.actName) == 'mish':
            out_x = x_input*tf.tanh(tf.math.log(1+tf.exp(x_input)))
        else:
            out_x = x_input
        return out_x


# ----------------Sequential dense net(constructing NN and initializing weights and bias )------------
class Dense_seqNet(tf.keras.Model):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='linear',
                 actName='tanh', actName2out='linear'):
        super(Dense_seqNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.num2NN_layers = len(hidden_units)+1
        self.name2Model = name2Model
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.dense_layers = []

        if str.lower(self.name2Model) == 'fourierdnn':
            for i_layer in range(len(hidden_units)):
                if i_layer == 0:
                    dense_hidden = tf.keras.layers.Dense(hidden_units[i_layer], use_bias=False)
                else:
                    dense_hidden = tf.keras.layers.Dense(hidden_units[i_layer])
                self.dense_layers.append(dense_hidden)
            dense_out = tf.keras.layers.Dense(self.outdim)
            self.dense_layers.append(dense_out)
        else:
            for i_layer in range(len(hidden_units)):
                dense_hidden = tf.keras.layers.Dense(hidden_units[i_layer])
                self.dense_layers.append(dense_hidden)
            dense_out = tf.keras.layers.Dense(self.outdim)
            self.dense_layers.append(dense_out)

    def call(self, inputs, scale=None, training=None, mask=None):
        dense_in = self.dense_layers[0]
        H = dense_in(inputs)
        if str.lower(self.name2Model) == 'fourier_dnn':
            Unit_num = int(self.hidden_units[0] / len(scale))
            mixcoe = np.repeat(scale, Unit_num)

            if self.repeat_Highfreq == True:
                mixcoe = np.concatenate(
                    (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
            else:
                mixcoe = np.concatenate(
                    (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

            mixcoe = mixcoe.astype(np.float32)
            H = tf.concat([tf.cos(H*mixcoe), tf.sin(H*mixcoe)], axis=-1)
        elif str.lower(self.name2Model) == 'scale_dnn':
            Unit_num = int(self.hidden_units[0] / len(scale))
            mixcoe = np.repeat(scale, Unit_num)

            if self.repeat_Highfreq == True:
                mixcoe = np.concatenate(
                    (mixcoe, np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[-1]))
            else:
                mixcoe = np.concatenate(
                    (np.ones([self.hidden_units[0] - Unit_num * len(scale)]) * scale[0], mixcoe))

            mixcoe = mixcoe.astype(np.float32)
            H = self.actFunc_in(H*mixcoe)
        else:
            H = self.actFunc_in(H)

        for i_layer in range(1, self.num2NN_layers-1):
            dense_layer = self.dense_layers[i_layer]
            H = dense_layer(H)
            H = self.actFunc(H)
        dense_out = self.dense_layers[-1]
        H =dense_out(H)
        H_out = self.actFunc_out(H)
        return H_out


# ----------------Subclass dense net(constructing NN and initializing weights and bias )------------
class Dense_Net(tf.keras.Model):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='linear', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32'):
        super(Dense_Net, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            float_type = tf.float32
        elif type2float == 'float64':
            float_type = tf.float64
        else:
            float_type = tf.float16

        if str.lower(self.name2Model) == 'fourier_dnn':
            Win = self.add_weight(shape=(indim, hidden_units[0]), initializer=tf.keras.initializers.GlorotNormal,
                                  trainable=True, name=str(scope2W) + '_in', dtype=float_type)
            Bin = self.add_weight(shape=(hidden_units[0],), initializer=tf.keras.initializers.GlorotNormal,
                                  trainable=False, name=str(scope2B) + '_in', dtype=float_type)
            self.Ws.append(Win)
            self.Bs.append(Bin)
            for i_layer in range(len(hidden_units)-1):
                if i_layer == 0:
                    W = self.add_weight(shape=(2 * hidden_units[i_layer], hidden_units[i_layer+1]),
                                        initializer=tf.keras.initializers.GlorotNormal,
                                        trainable=True, name=str(scope2W) + str(i_layer), dtype=float_type)
                    B = self.add_weight(shape=(hidden_units[i_layer+1],), initializer=tf.keras.initializers.GlorotNormal,
                                        trainable=True, name=str(scope2B) + str(i_layer), dtype=float_type)
                else:
                    W = self.add_weight(shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                        initializer=tf.keras.initializers.GlorotNormal,
                                        trainable=True, name=str(scope2W) + str(i_layer), dtype=float_type)
                    B = self.add_weight(shape=(hidden_units[i_layer + 1],),
                                        initializer=tf.keras.initializers.GlorotNormal,
                                        trainable=True, name=str(scope2B) + str(i_layer), dtype=float_type)
                self.Ws.append(W)
                self.Bs.append(B)
        else:
            Win = self.add_weight(shape=(indim, hidden_units[0]), initializer=tf.keras.initializers.GlorotNormal,
                                  trainable=True, name=str(scope2W) + '_in', dtype=float_type)
            Bin = self.add_weight(shape=(hidden_units[0],), initializer=tf.keras.initializers.GlorotNormal,
                                  trainable=True, name=str(scope2B) + '_in', dtype=float_type)
            self.Ws.append(Win)
            self.Bs.append(Bin)
            for i_layer in range(len(hidden_units)-1):
                W = self.add_weight(shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                    initializer=tf.keras.initializers.GlorotNormal,
                                    trainable=True, name=str(scope2W) + str(i_layer), dtype=float_type)
                B = self.add_weight(shape=(hidden_units[i_layer + 1],),
                                    initializer=tf.keras.initializers.GlorotNormal,
                                    trainable=True, name=str(scope2B) + str(i_layer), dtype=float_type)
                self.Ws.append(W)
                self.Bs.append(B)

        Wout = self.add_weight(shape=(hidden_units[-1], outdim), initializer=tf.keras.initializers.GlorotNormal,
                               trainable=True, name=str(scope2W) + '_out', dtype=float_type)
        Bout = self.add_weight(shape=(outdim,), initializer=tf.keras.initializers.GlorotNormal, trainable=True,
                               name=str(scope2B) + '_out', dtype=float_type)
        self.Ws.append(Wout)
        self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def call(self, inputs, scale=None, training=None, mask=None):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H = tf.add(tf.matmul(inputs, self.Ws[0]), self.Bs[0])
        if str.lower(self.name2Model) == 'fourier_dnn':
            assert (len(scale) != 0)
            repeat_num = int(self.hidden_units[0] / len(scale))
            repeat_scale = np.repeat(scale, repeat_num)

            if self.repeat_high_freq:
                repeat_scale = np.concatenate(
                    (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
            else:
                repeat_scale = np.concatenate(
                    (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

            if self.type2float == 'float32':
                repeat_scale = repeat_scale.astype(np.float32)
            elif self.type2float == 'float64':
                repeat_scale = repeat_scale.astype(np.float64)
            else:
                repeat_scale = repeat_scale.astype(np.float16)

            if self.actName == 's2relu':
                H = 0.5*tf.concat([tf.cos(H * repeat_scale), tf.sin(H * repeat_scale)], axis=-1)
            else:
                H = tf.concat([tf.cos(H * repeat_scale), tf.sin(H * repeat_scale)], axis=-1)
        elif str.lower(self.name2Model) == 'scale_dnn' or str.lower(self.name2Model) == 'wavelet_dnn':
            assert (len(scale) != 0)
            repeat_num = int(self.hidden_units[0] / len(scale))
            repeat_scale = np.repeat(scale, repeat_num)

            if self.repeat_high_freq:
                repeat_scale = np.concatenate(
                    (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
            else:
                repeat_scale = np.concatenate(
                    (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

            if self.type2float == 'float32':
                repeat_scale = repeat_scale.astype(np.float32)
            elif self.type2float == 'float64':
                repeat_scale = repeat_scale.astype(np.float64)
            else:
                repeat_scale = repeat_scale.astype(np.float16)

            H = self.actFunc_in(H*repeat_scale)
        else:  # name2Model:DNN, RBF_DNN
            H = self.actFunc_in(H)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units)-1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer+1]), self.Bs[i_layer+1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Dense_ScaleNet(tf.keras.Model):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
        if name2Model is not wavelet NN, actName2in is not same as actName; otherwise, actName2in is same as actName
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='linear', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32'):
        super(Dense_ScaleNet, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName2in = actName2in
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc_in = my_actFunc(actName=actName2in)
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            float_type = tf.float32
        elif type2float == 'float64':
            float_type = tf.float64
        else:
            float_type = tf.float16

        Win = self.add_weight(shape=(indim, hidden_units[0]), initializer=tf.keras.initializers.GlorotNormal,
                              trainable=True, name=str(scope2W) + '_in', dtype=float_type)
        Bin = self.add_weight(shape=(hidden_units[0],), initializer=tf.keras.initializers.GlorotNormal,
                              trainable=True, name=str(scope2B) + '_in', dtype=float_type)
        self.Ws.append(Win)
        self.Bs.append(Bin)
        for i_layer in range(len(hidden_units)-1):
            W = self.add_weight(shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                initializer=tf.keras.initializers.GlorotNormal,
                                trainable=True, name=str(scope2W) + str(i_layer), dtype=float_type)
            B = self.add_weight(shape=(hidden_units[i_layer + 1],),
                                initializer=tf.keras.initializers.GlorotNormal,
                                trainable=True, name=str(scope2B) + str(i_layer), dtype=float_type)
            self.Ws.append(W)
            self.Bs.append(B)

        Wout = self.add_weight(shape=(hidden_units[-1], outdim), initializer=tf.keras.initializers.GlorotNormal,
                               trainable=True, name=str(scope2W) + '_out', dtype=float_type)
        Bout = self.add_weight(shape=(outdim,), initializer=tf.keras.initializers.GlorotNormal, trainable=True,
                               name=str(scope2B) + '_out', dtype=float_type)
        self.Ws.append(Wout)
        self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def call(self, inputs, scale=None, training=None, mask=None):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H = tf.add(tf.matmul(inputs, self.Ws[0]), self.Bs[0])
        assert (len(scale) != 0)
        repeat_num = int(self.hidden_units[0] / len(scale))
        repeat_scale = np.repeat(scale, repeat_num)

        if self.repeat_high_freq:
            repeat_scale = np.concatenate(
                (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
        else:
            repeat_scale = np.concatenate(
                (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

        if self.type2float == 'float32':
            repeat_scale = repeat_scale.astype(np.float32)
        elif self.type2float == 'float64':
            repeat_scale = repeat_scale.astype(np.float64)
        else:
            repeat_scale = repeat_scale.astype(np.float16)

        H = self.actFunc_in(H*repeat_scale)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units)-1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer+1]), self.Bs[i_layer+1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        out_result = self.actFunc_out(H)
        return out_result


class Dense_Fourier_Net(tf.keras.Model):
    """
    Args:
        indim: the dimension for input data
        outdim: the dimension for output
        hidden_units: the number of  units for hidden layer, a list or a tuple
        name2Model: the name of using DNN type, DNN , ScaleDNN or FourierDNN
        actName2in: the name of activation function for input layer
        actName: the name of activation function for hidden layer
        actName2out: the name of activation function for output layer
        scope2W: the namespace of weight
        scope2B: the namespace of bias
        repeat_high_freq: repeating the high-frequency component of scale-transformation factor or not
    """
    def __init__(self, indim=1, outdim=1, hidden_units=None, name2Model='DNN', actName2in='linear', actName='tanh',
                 actName2out='linear', scope2W='Weight', scope2B='Bias', repeat_high_freq=True, type2float='float32'):
        super(Dense_Fourier_Net, self).__init__()
        self.indim = indim
        self.outdim = outdim
        self.hidden_units = hidden_units
        self.name2Model = name2Model
        self.actName = actName
        self.actName2out = actName2out
        self.actFunc = my_actFunc(actName=actName)
        self.actFunc_out = my_actFunc(actName=actName2out)
        self.repeat_high_freq = repeat_high_freq
        self.type2float = type2float
        self.Ws = []
        self.Bs = []
        if type2float == 'float32':
            float_type = tf.float32
        elif type2float == 'float64':
            float_type = tf.float64
        else:
            float_type = tf.float16

        Win = self.add_weight(shape=(indim, hidden_units[0]), initializer=tf.keras.initializers.GlorotNormal,
                              trainable=True, name=str(scope2W) + '_in', dtype=float_type)
        Bin = self.add_weight(shape=(hidden_units[0],), initializer=tf.keras.initializers.Zeros,
                              trainable=False, name=str(scope2B) + '_in', dtype=float_type)
        self.Ws.append(Win)
        self.Bs.append(Bin)
        for i_layer in range(len(hidden_units)-1):
            if i_layer == 0:
                W = self.add_weight(shape=(2 * hidden_units[i_layer], hidden_units[i_layer+1]),
                                    initializer=tf.keras.initializers.GlorotNormal,
                                    trainable=True, name=str(scope2W) + str(i_layer), dtype=float_type)
                B = self.add_weight(shape=(hidden_units[i_layer+1],), initializer=tf.keras.initializers.GlorotNormal,
                                    trainable=True, name=str(scope2B) + str(i_layer), dtype=float_type)
            else:
                W = self.add_weight(shape=(hidden_units[i_layer], hidden_units[i_layer + 1]),
                                    initializer=tf.keras.initializers.GlorotNormal,
                                    trainable=True, name=str(scope2W) + str(i_layer), dtype=float_type)
                B = self.add_weight(shape=(hidden_units[i_layer + 1],),
                                    initializer=tf.keras.initializers.GlorotNormal,
                                    trainable=True, name=str(scope2B) + str(i_layer), dtype=float_type)
            self.Ws.append(W)
            self.Bs.append(B)

        Wout = self.add_weight(shape=(hidden_units[-1], outdim), initializer=tf.keras.initializers.GlorotNormal,
                               trainable=True, name=str(scope2W) + '_out', dtype=float_type)
        Bout = self.add_weight(shape=(outdim,), initializer=tf.keras.initializers.GlorotNormal, trainable=True,
                               name=str(scope2B) + '_out', dtype=float_type)
        self.Ws.append(Wout)
        self.Bs.append(Bout)

    def get_regular_sum2WB(self, regular_model):
        layers = len(self.hidden_units)+1
        if regular_model == 'L1':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.abs(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.abs(self.Bs[i_layer]), keepdims=False)
        elif regular_model == 'L2':
            regular_w = 0
            regular_b = 0
            for i_layer in range(layers):
                regular_w = regular_w + tf.reduce_sum(tf.square(self.Ws[i_layer]), keepdims=False)
                regular_b = regular_b + tf.reduce_sum(tf.square(self.Bs[i_layer]), keepdims=False)
        else:
            regular_w = tf.constant(0.0)
            regular_b = tf.constant(0.0)
        return regular_w + regular_b

    def call(self, inputs, scale=None, training=None, mask=None):
        """
        Args
            inputs: the input point set [num, in-dim]
            scale: The scale-factor to transform the input-data
        return
            output: the output point set [num, out-dim]
        """
        # ------ dealing with the input data ---------------
        H = tf.add(tf.matmul(inputs, self.Ws[0]), self.Bs[0])
        assert (len(scale) != 0)
        repeat_num = int(self.hidden_units[0] / len(scale))
        repeat_scale = np.repeat(scale, repeat_num)

        if self.repeat_high_freq:
            repeat_scale = np.concatenate(
                (repeat_scale, np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[-1]))
        else:
            repeat_scale = np.concatenate(
                (np.ones([self.hidden_units[0] - repeat_num * len(scale)]) * scale[0], repeat_scale))

        if self.type2float == 'float32':
            repeat_scale = repeat_scale.astype(np.float32)
        elif self.type2float == 'float64':
            repeat_scale = repeat_scale.astype(np.float64)
        else:
            repeat_scale = repeat_scale.astype(np.float16)

        if self.actName == 's2relu':
            H = 0.5*tf.concat([tf.cos(H * repeat_scale), tf.sin(H * repeat_scale)], axis=-1)
        else:
            H = tf.concat([tf.cos(H * repeat_scale), tf.sin(H * repeat_scale)], axis=-1)

        #  ---resnet(one-step skip connection for two consecutive layers if have equal neurons）---
        hidden_record = self.hidden_units[0]
        for i_layer in range(len(self.hidden_units)-1):
            H_pre = H
            H = tf.add(tf.matmul(H, self.Ws[i_layer+1]), self.Bs[i_layer+1])
            H = self.actFunc(H)
            if self.hidden_units[i_layer + 1] == hidden_record:
                H = H + H_pre
            hidden_record = self.hidden_units[i_layer + 1]

        H = tf.add(tf.matmul(H, self.Ws[-1]), self.Bs[-1])
        H = self.actFunc_out(H)
        # H = tf.tanh(H)
        # H = tf.exp(-0.5*H*H)
        return H


# 在图上求关于输入变量X的导数，这里
# new_point = X + KV, K是归一化的系数， V是k近邻点集
# U = new_point[:,0] + new_point[:,1] + ....
# dU/dX = gradient(U, X), 只关于X求导，V中关联的X也求导
def test1():
    num2points = 50
    kneighbor = 3
    indim = 2
    outdim = 1
    hiddens = (4, 8, 16)
    # x0 = np.array([[1, 2],
    #               [2, 3],
    #               [3, 5],
    #               [8, 9],
    #               [5, 2],
    #               [4, 1],
    #               [3, 2],
    #               [9, 3],
    #               [8, 5],
    #               [7, 1]], dtype=np.float)
    x0 = np.array([[1, 2],
                   [2, 3],
                   [3, 5],
                   [8, 9],
                   [5, 2]], dtype=np.float)
    X_aux = tf.Variable(initial_value=x0, trainable=False, name='X_aux')
    adj_mat = pairwise_distance(X_aux)
    k_index = knn_excludeself(adj_mat, k=kneighbor)
    print('k_index:', k_index)

    neighbors_aux = get_kneighbors_2DTensor(X_aux, k_index)
    expand_point = tf.expand_dims(X_aux, axis=1)
    point_cen_tile = tf.tile(expand_point, [1, kneighbor, 1])
    edge2points = point_cen_tile - neighbors_aux
    atten_ceof = cal_attend2neighbors(edge2points)
    print('atten_ceof:', atten_ceof)

    X = tf.Variable(initial_value=x0, trainable=True, name='X')
    with tf.GradientTape(persistent=True) as t:
        t.watch(X)
        neighbors = get_kneighbors_2DTensor(X, k_index)
        agg_nei_feature = tf.matmul(atten_ceof, neighbors)
        agg_nei_feature = tf.squeeze(agg_nei_feature, axis=1)
        print('aggregate result for k_neighbors:', agg_nei_feature)
        sum_agg_nei = tf.reduce_sum(agg_nei_feature, axis=-1, keepdims=True)
        print('sum_agg_nei:', sum_agg_nei)
        d_sum_agg_x = t.gradient(sum_agg_nei, X)
        print('d_sum_agg_x:', d_sum_agg_x)

        U = tf.reduce_sum(tf.add(agg_nei_feature, X), axis=-1, keepdims=True)
        print('new point:', U)
        ux = t.gradient(U, X)
        print('ux:', ux)


# 在图上求关于输入变量X的导数，这里
# new_point = X + KV, K是归一化的系数， V是k近邻点集
# U = new_point[:,0] + new_point[:,1] + ....
# dU/dX = gradient(U, X), 只关于X求导，V中关联的X不求导
def test2():
    num2points = 50
    kneighbor = 3
    indim = 2
    outdim = 1
    hiddens = (4, 8, 16)
    x_point = np.array([[1, 2],
                  [2, 3],
                  [3, 5],
                  [8, 9],
                  [5, 2],
                  [4, 1],
                  [3, 2],
                  [9, 3],
                  [8, 5],
                  [7, 1]], dtype=np.float)
    # x_point = np.array([[1, 2],
    #                [2, 3],
    #                [3, 5],
    #                [8, 9],
    #                [5, 2]], dtype=np.float)

    X_aux = tf.Variable(initial_value=x_point, trainable=False, name='X_auxiliary')
    adj_mat = pairwise_distance(X_aux)
    k_index = knn_excludeself(adj_mat, k=kneighbor)
    print('k_index:', k_index)

    neighbors = get_kneighbors_2DTensor(X_aux, k_index)
    expand_point = tf.expand_dims(X_aux, axis=1)
    point_cen_tile = tf.tile(expand_point, [1, kneighbor, 1])
    edge2points = point_cen_tile - neighbors

    atten_ceof = cal_attend2neighbors(edge2points)
    print('atten_ceof:', atten_ceof)

    agg_nei = tf.matmul(atten_ceof, neighbors)
    agg_nei_feature = tf.squeeze(agg_nei, axis=1)
    print('aggregate result for k_neighbors:', agg_nei_feature)

    X = tf.Variable(initial_value=x_point, trainable=True, name='X')
    with tf.GradientTape(persistent=True) as t:
        t.watch(X)
        new_point = tf.add(agg_nei_feature, X)
        U = tf.reduce_sum(new_point, axis=-1, keepdims=True)
        print('new point:', U)
        ux = t.gradient(U, X)
        print('ux:', ux)


# 在图上求关于输入变量X的导数，这里
# new_point = X + KV, K是归一化的系数， V是k近邻点集
# U = new_point[:,0] + new_point[:,1] + ....
# dU/dX = gradient(U, X), 只关于X求导，V中关联的X不求导
def test_GKN():
    num2points = 50
    kneighbor = 3
    indim = 2
    outdim = 1
    hiddens = (4, 8, 16)
    x_point = np.array([[1, 2],
                  [2, 3],
                  [3, 5],
                  [8, 9],
                  [5, 2],
                  [4, 1],
                  [3, 2],
                  [9, 3],
                  [8, 5],
                  [7, 1]], dtype=np.float)
    # x_point = np.array([[1, 2],
    #                [2, 3],
    #                [3, 5],
    #                [8, 9],
    #                [5, 2]], dtype=np.float)

    X_aux = tf.Variable(initial_value=x_point, trainable=False, name='X_auxiliary')
    adj_mat = pairwise_distance(X_aux)
    k_index = knn_excludeself(adj_mat, k=kneighbor)
    print('k_index:', k_index)

    neighbors = get_kneighbors_2DTensor(X_aux, k_index)
    expand_point = tf.expand_dims(X_aux, axis=1)
    point_cen_tile = tf.tile(expand_point, [1, kneighbor, 1])
    edge2points = point_cen_tile - neighbors

    atten_ceof = cal_attend2neighbors(edge2points)
    print('atten_ceof:', atten_ceof)

    agg_nei = tf.matmul(atten_ceof, neighbors)
    agg_nei_feature = tf.squeeze(agg_nei, axis=1)
    print('aggregate result for k_neighbors:', agg_nei_feature)

    X = tf.Variable(initial_value=x_point, trainable=True, name='X')
    Wx = tf.Variable(initial_value=np.array([[10, 1], [20, 1]]), trainable=True, name='Wx')
    Wout = tf.Variable(initial_value=np.array([[10], [20]]), trainable=True, name='Wout')
    with tf.GradientTape(persistent=True) as t:
        t.watch(X)
        new_point = tf.add(agg_nei_feature, X)
        U = tf.matmul(new_point, Wout)
        print('new point:', U)
        ux = t.gradient(U, X)
        print('ux:', ux)


if __name__ == "__main__":
    test1()
    # test2()
