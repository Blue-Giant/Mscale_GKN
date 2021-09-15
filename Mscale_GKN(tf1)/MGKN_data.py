import numpy as np


# ---------------------------------------------- 数据集的生成 ---------------------------------------------------
#  方形区域[a,b]^n生成随机数, n代表变量个数
def rand_it(batch_size, variable_dim, region_a, region_b):
    # np.random.rand( )可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。
    # np.random.rand(3,2 )可以返回一个或一组服从“0~1”均匀分布的随机矩阵(3行2列)。随机样本取值范围是[0,1)，不包括1。
    x_it = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    x_it = x_it.astype(np.float32)
    return x_it


def rand_bd_1D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 1:
        x_left_bd = np.ones(shape=[batch_size, variable_dim], dtype=np.float32) * region_a
        x_right_bd = np.ones(shape=[batch_size, variable_dim], dtype=np.float32) * region_b
        return x_left_bd, x_right_bd
    else:
        return


def rand_bd_2D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    # np.random.random((100, 50)) 上方代表生成100行 50列的随机浮点数，浮点数范围 : (0,1)
    # np.random.random([100, 50]) 和 np.random.random((100, 50)) 效果一样
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 2:
        x_left_bd = (region_b-region_a) * np.random.random([batch_size, 2]) + region_a   # 浮点数都是从0-1中随机。
        for ii in range(batch_size):
            x_left_bd[ii, 0] = region_a

        x_right_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        for ii in range(batch_size):
            x_right_bd[ii, 0] = region_b

        y_bottom_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        for ii in range(batch_size):
            y_bottom_bd[ii, 1] = region_a

        y_top_bd = (region_b - region_a) * np.random.random([batch_size, 2]) + region_a
        for ii in range(batch_size):
            y_top_bd[ii, 1] = region_b

        x_left_bd = x_left_bd.astype(np.float32)
        x_right_bd = x_right_bd.astype(np.float32)
        y_bottom_bd = y_bottom_bd.astype(np.float32)
        y_top_bd = y_top_bd.astype(np.float32)
        return x_left_bd, x_right_bd, y_bottom_bd, y_top_bd
    else:
        return


def rand_bd_3D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 3:
        bottom_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            bottom_bd[ii, 2] = region_a

        top_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            top_bd[ii, 2] = region_b

        left_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            left_bd[ii, 1] = region_a

        right_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            right_bd[ii, 1] = region_b

        front_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            front_bd[ii, 0] = region_b

        behind_bd = (region_b - region_a) * np.random.rand(batch_size, 3) + region_a
        for ii in range(batch_size):
            behind_bd[ii, 0] = region_a

        bottom_bd = bottom_bd.astype(np.float32)
        top_bd = top_bd.astype(np.float32)
        left_bd = left_bd.astype(np.float32)
        right_bd = right_bd.astype(np.float32)
        front_bd = front_bd.astype(np.float32)
        behind_bd = behind_bd.astype(np.float32)
        return bottom_bd, top_bd, left_bd, right_bd, front_bd, behind_bd
    else:
        return


def rand_bd_4D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    variable_dim = int(variable_dim)

    x0a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x0a[ii, 0] = region_a

    x0b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x0b[ii, 0] = region_b

    x1a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x1a[ii, 1] = region_a

    x1b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x1b[ii, 1] = region_b

    x2a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x2a[ii, 2] = region_a

    x2b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x2b[ii, 2] = region_b

    x3a = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x3a[ii, 3] = region_a

    x3b = (region_b - region_a) * np.random.rand(batch_size, variable_dim) + region_a
    for ii in range(batch_size):
        x3b[ii, 3] = region_b

    x0a = x0a.astype(np.float32)
    x0b = x0b.astype(np.float32)

    x1a = x1a.astype(np.float32)
    x1b = x1b.astype(np.float32)

    x2a = x2a.astype(np.float32)
    x2b = x2b.astype(np.float32)

    x3a = x3a.astype(np.float32)
    x3b = x3b.astype(np.float32)

    return x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b


def rand_bd_5D(batch_size, variable_dim, region_a, region_b):
    # np.asarray 将输入转为矩阵格式。
    # 当输入是列表的时候，更改列表的值并不会影响转化为矩阵的值
    # [0,1] 转换为 矩阵，然后
    # reshape(-1,1):数组新的shape属性应该要与原来的配套，如果等于-1的话，那么Numpy会根据剩下的维度计算出数组的另外一个shape属性值。
    region_a = float(region_a)
    region_b = float(region_b)
    if variable_dim == 5:
        x0a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x0a[ii, 0] = region_a

        x0b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x0b[ii, 0] = region_b

        x1a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x1a[ii, 1] = region_a

        x1b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x1b[ii, 1] = region_b

        x2a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x2a[ii, 2] = region_a

        x2b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x2b[ii, 2] = region_b

        x3a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x3a[ii, 3] = region_a

        x3b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x3b[ii, 3] = region_b

        x4a = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x4a[ii, 4] = region_a

        x4b = (region_b - region_a) * np.random.rand(batch_size, 5) + region_a
        for ii in range(batch_size):
            x4b[ii, 4] = region_b

        x0a = x0a.astype(np.float32)
        x0b = x0b.astype(np.float32)

        x1a = x1a.astype(np.float32)
        x1b = x1b.astype(np.float32)

        x2a = x2a.astype(np.float32)
        x2b = x2b.astype(np.float32)

        x3a = x3a.astype(np.float32)
        x3b = x3b.astype(np.float32)

        x4a = x4a.astype(np.float32)
        x4b = x4b.astype(np.float32)
        return x0a, x0b, x1a, x1b, x2a, x2b, x3a, x3b, x4a, x4b
    else:
        return


# --------- sampling from existed data ------------------
# 随机地从给定的数据中采样
# numpy.size(a, axis=None) ---- a：输入的矩阵  axis：int型的可选参数，指定返回哪一维的元素个数。当没有指定时，返回整个矩阵的元素个数。
# numpy.random.randint(low, high=None, size=None, dtype='l')
# 返回一个随机整型数或数组，范围从低（包括）到高（不包括），即[low, high)。如果没有写参数high的值，则返回[0,low)的值。
def randSample_existData2(coordData=None, soluData=None, numSamples=10, unique_index=True):
    """
        Args:
        coordData:  the coords of point in d-dimension space with shape [num2points, dim]
        soluData:   the true solution with shape [num2points, 1]
        numSamples: the num of sample points

        return:
        coordData_sample: [numSamples, dim]
        soluData_sample: [numSamples, 1]
    """
    coords_temp = []
    solus_temp = []
    shape2coordData = np.shape(coordData)
    shape2soluData = np.shape(soluData)
    assert (len(shape2coordData) == 2)
    assert (len(shape2soluData) == 2)
    assert (np.size(soluData, 1) == 1)
    data_length = np.size(coordData, 0)
    assert(numSamples < data_length)
    indexes = np.random.randint(data_length, size=numSamples)  # generating the index-array
    if unique_index:
        indexes = np.unique(indexes)                               # removing the repeating index in array

    for i_index in indexes:
        coords_temp.append(coordData[i_index, :])
        solus_temp.append(soluData[i_index, 0])
    coordData_sample = np.array(coords_temp)
    soluData_sample = np.array(solus_temp)

    # batchsize = len(indexes)
    coordData_sample = np.reshape(coordData_sample, (-1, shape2coordData[1]))
    soluData_sample = np.reshape(soluData_sample, (-1, 1))
    return coordData_sample, soluData_sample


def randSample_existData3(coordData=None, soluData=None, coefData=None, numSamples=10, unique_index=True):
    """
        Args:
        coordData:  the coords of point in d-dimension space with shape [num2point, dim]
        soluData:   the true solution with shape [num2point, 1]
        coeffData:  the value of coefficient function in coord points, shape [num2point, 1]
        numSamples: the num of sample points

        return:
        coordData_sample: [numSamples, dim]
        soluData_sample: [numSamples, 1]
        coefData_sample: [numSamples, 1]
        """
    coords_temp = []
    solus_temp = []
    coef_temp = []
    shape2coordData = np.shape(coordData)
    shape2soluData = np.shape(soluData)
    shape2coefData = np.shape(coefData)
    assert (len(shape2coordData) == 2)
    assert (len(shape2soluData) == 2)
    assert (len(shape2coefData) == 2)
    assert (np.size(soluData, 1) == 1)
    assert (np.size(coefData, 1) == 1)
    data_length = np.size(coordData, 0)
    assert (numSamples < data_length)
    indexes = np.random.randint(data_length, size=numSamples)  # generating the index-array
    if unique_index:
        indexes = np.unique(indexes)                               # removing the repeating index in array

    for i_index in indexes:
        coords_temp.append(coordData[i_index, :])
        solus_temp.append(soluData[i_index, 0])
        coef_temp.append(coefData[i_index, 0])
    coordData_sample = np.array(coords_temp)
    soluData_sample = np.array(solus_temp)
    coefData_sample = np.array(coef_temp)

    batchsize = len(indexes)
    coordData_sample = coordData_sample.reshape(batchsize, shape2coordData(1))
    soluData_sample = soluData_sample.reshape(batchsize, 1)
    coefData_sample = coefData_sample.reshape(batchsize, 1)
    return coordData_sample, soluData_sample, coefData_sample


# 根据索引的上下界，随机地从给定的数据中采样
def rangeIndexSample_existData2(coordData=None, soluData=None, beginIndex=0, endIndex=10, unique_index=True):
    """
        Args:
        coordData:  the coords of point in d-dimension space with shape [num2point, dim]
        soluData:   the true solution with shape [num2point, 1]
        beginIndex: the begin index for sampling
        endIndex: the end index for sampling

        return:
        coordData_sample: [numSamples, dim]
        soluData_sample: [numSamples, 1]
    """
    coords_temp = []
    solus_temp = []
    shape2coordData = np.shape(coordData)
    shape2soluData = np.shape(soluData)
    assert (len(shape2coordData) == 2)
    assert (len(shape2soluData) == 2)
    assert (np.size(soluData, 1) == 1)
    data_length = np.size(coordData, 0)
    assert (endIndex < data_length)
    batchsize = int(endIndex - beginIndex)
    indexes = np.random.randint(beginIndex, high=endIndex, size=batchsize)  # generating the index-array
    if unique_index:
        indexes = np.unique(indexes)

    for i_index in indexes:
        coords_temp.append(coordData[i_index, :])
        solus_temp.append(soluData[i_index, 0])
    coordData_sample = np.array(coords_temp)
    soluData_sample = np.array(solus_temp)

    batchsize = len(indexes)
    coordData_sample = coordData_sample.reshape(batchsize, shape2coordData(1))
    soluData_sample = soluData_sample.reshape(batchsize, 1)
    return coordData_sample, soluData_sample


def rangeIndexSample_existData3(coordData=None, soluData=None, coefData=None, beginIndex=0, endIndex=10,
                                unique_index=True):
    """
        Args:
        coordData:  the coords of point in d-dimension space with shape [num2point, dim]
        soluData:   the true solution with shape [num2point, 1]
        coeffData:  the value of coefficient function in coord points, shape [num2point, 1]
        beginIndex: the begin index for sampling
        endIndex: the end index for sampling

        return:
        coordData_sample: [numSamples, dim]
        soluData_sample: [numSamples, 1]
        coefData_sample: [numSamples, 1]
    """
    coords_temp = []
    solus_temp = []
    coef_temp = []
    shape2coordData = np.shape(coordData)
    shape2soluData = np.shape(soluData)
    shape2coefData = np.shape(coefData)
    assert (len(shape2coordData) == 2)
    assert (len(shape2soluData) == 2)
    assert (len(shape2coefData) == 2)
    assert (np.size(soluData, 1) == 1)
    assert (np.size(coefData, 1) == 1)
    data_length = np.size(coordData, 0)
    assert (endIndex < data_length)
    batchsize = int(endIndex - beginIndex)
    indexes = np.random.randint(beginIndex, high=endIndex, size=batchsize)  # generating the index-array
    if unique_index:
        indexes = np.unique(indexes)

    for i_index in indexes:
        coords_temp.append(coordData[i_index, :])
        solus_temp.append(soluData[i_index, 0])
        coef_temp.append(coefData[i_index, 0])
    coordData_sample = np.array(coords_temp)
    soluData_sample = np.array(solus_temp)
    coefData_sample = np.array(coef_temp)

    batchsize = len(indexes)
    coordData_sample = coordData_sample.reshape(batchsize, shape2coordData(1))
    soluData_sample = soluData_sample.reshape(batchsize, 1)
    coefData_sample = coefData_sample.reshape(batchsize, 1)
    return coordData_sample, soluData_sample, coefData_sample


# 根据给定的索引从给定的数据中采样
def indexSample_existData(data=None, indexes=None):
    """
        Args:
        data:  [num2data, dim]
        indexes: [num2index, 1]

        return:
        data_sample: [num2index, dim]
    """
    data_temp = []
    shape2data = np.shape(data)
    assert (len(shape2data) == 2)
    data_length = shape2data[0]
    assert(max(indexes) < data_length)
    # indexes = np.random.randint(data_length, size=batchsize)
    for i_index in indexes:
        data_temp.append(data[i_index, :])
    data_samples = np.array(data_temp)

    batchsize = len(indexes)
    data_samples = data_samples.reshape(batchsize, shape2data[1])
    return data_samples