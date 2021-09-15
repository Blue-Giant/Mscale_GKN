import tensorflow as tf
import scipy.io
import numpy as np
import sklearn
import h5py


def pairwise_distance(point_set):
    """Compute pairwise distance of a point cloud.
        Args:
          (x-y)^2 = x^2 - 2xy + y^2
          point_set: tensor (num_points, dims2point)
        Returns:
          pairwise distance: (num_points, num_points)
    """
    point_set = tf.cast(point_set, dtype=tf.float32)
    point_set_shape = point_set.get_shape().as_list()
    assert(len(point_set_shape)) == 2

    point_set_transpose = tf.transpose(point_set, perm=[1, 0])
    point_set_inner = tf.matmul(point_set, point_set_transpose)
    point_set_inner = -2 * point_set_inner
    point_set_square = tf.reduce_sum(tf.square(point_set), axis=-1, keepdims=True)
    point_set_square_transpose = tf.transpose(point_set_square, perm=[1, 0])
    return point_set_square + point_set_inner + point_set_square_transpose


class MatReader(object):
    def __init__(self, file_path, to_tensor=True, to_float=True):
        super(MatReader, self).__init__()
        """
        Args:
            file_path: the path of mat-file
            to_tensor: transform mat-form into tensor-form
            to_float: transform data-type into float data-type
        """

        self.to_tensor = to_tensor
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        """
        Args:
             load the mat-file according to path-file of mat-file
        """
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        """
        Args:
             load the mat-file according to path-file of mat-file
        """
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        """
        Args:
             read the data according to attribute from data-item
        return:
             the need-data
        """
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_tensor:
            x = tf.convert_to_tensor(x)

        return x

    def set_tensor(self, to_tensor):
        """
        Args:
          to_tensor: given the set, transform numpy into tf.tensor
        """
        self.to_tensor = to_tensor

    def set_float(self, to_float):
        """
        Args:
          to_float: given the set, transform data-type into tf.float32
        """
        self.to_float = to_float


class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        mean2x = tf.reduce_mean(x, axis=0, keepdims=True)
        self.mean = tf.reshape(mean2x, [-1])
        var2x = tf.reduce_mean(tf.square(x - self.mean), axis=0, keepdims=True)
        self.std = tf.reshape(var2x, [-1])

        self.eps = eps

    def encode(self, x):
        s = tf.shape(x)
        x = tf.reshape(s[0], -1)
        x = (x - self.mean) / (self.std + self.eps)
        x = tf.reshape(x, s)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps   # n
            mean = self.mean
        else:
            std = self.std[sample_idx] + self.eps  # batch * n
            mean = self.mean[sample_idx]

        s = tf.shape(x)
        x = tf.reshape(s[0], -1)
        x = (x * std) + mean
        x = tf.reshape(x, s)
        return x


class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        mean2x = tf.reduce_mean(x)
        self.mean = mean2x
        var2x = tf.reduce_mean(tf.square(x-mean2x))
        self.std = var2x
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x


class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = tf.reshape(tf.argmin(x, axis=0)[0], [-1])
        mymax = tf.reshape(tf.argmax(x, axis=0)[0], [-1])

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = tf.shape(x)
        x = tf.reshape(s[0], [-1])
        x = self.a*x + self.b
        x = tf.reshape(x, s)
        return x

    def decode(self, x):
        s = tf.shape(x)
        x = tf.reshape(s[0], [-1])
        x = (x - self.b)/self.a
        x = tf.reshape(x, s)
        return x


class SquareMeshGenerator(object):
    def __init__(self, real_space, mesh_size):
        super(SquareMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.s = mesh_size[0]

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

    def ball_connectivity(self, r):
        pwd = pairwise_distance(self.grid)
        # pwd = sklearn.metrics.pairwise_distances(self.grid)
        temp_edge_index = np.where(pwd <= r)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return tf.cast(self.edge_index, dtype=tf.int32)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return tf.cast(self.edge_index, dtype=tf.int32)

    def get_grid(self):
        return tf.cast(self.grid, dtype=tf.float32)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                edge_attr = np.zeros((self.n_edges, 3 * self.d))
                edge_attr[:, 0:2 * self.d] = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return tf.cast(edge_attr, dtype=tf.float32)

    def get_boundary(self):
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]

        return tf.cast(self.edge_index_boundary, dtype=tf.int32)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 3 * self.d))
                edge_attr_boundary[:, 0:2 * self.d] = self.grid[self.edge_index_boundary.T].reshape(
                    (self.n_edges_boundary, -1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d + 1] = theta[self.edge_index_boundary[1]]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary, -1))
            if theta is None:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                edge_attr_boundary = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index_boundary[0]],
                                       theta[self.edge_index_boundary[1]])

        return tf.cast(edge_attr_boundary, dtype=tf.float32)


def test_data_pre():
    TRAIN_PATH = 'data/piececonst_r241_N1024_smooth1.mat'
    TEST_PATH = 'data/piececonst_r241_N1024_smooth2.mat'

    r = 4
    s = int(((241 - 1)/r) + 1)
    n = s**2
    m = 100
    k = 1

    radius_train = 0.1
    radius_test = 0.1

    print('resolution', s)

    ntrain = 100
    ntest = 40

    batch_size = 1
    batch_size2 = 2
    width = 64
    ker_width = 1024
    depth = 6
    edge_features = 6
    node_features = 6

    epochs = 200
    learning_rate = 0.0001
    scheduler_step = 50
    scheduler_gamma = 0.8

    path = 'UAI1_r'+str(s)+'_n'+ str(ntrain)
    path_model = 'model/'+path+''
    path_train_err = 'results/'+path+'train.txt'
    path_test_err = 'results/'+path+'test.txt'
    path_image = 'image/'+path+''
    path_train_err = 'results/'+path+'train'
    path_test_err16 = 'results/'+path+'test16'
    path_test_err31 = 'results/'+path+'test31'
    path_test_err61 = 'results/'+path+'test61'
    path_image_train = 'image/'+path+'train'
    path_image_test16 = 'image/'+path+'test16'
    path_image_test31 = 'image/'+path+'test31'
    path_image_test61 = 'image/'+path+'test61'

    reader = MatReader(TRAIN_PATH)
    # temp = reader.read_field('coeff')
    # temp_1 = temp[:ntrain, ::r, ::r]  # 双冒号的作用,步长step=n;代表从start开始（start也算）每隔step间隔，取一个数，一直到结尾end
    # temp2 = tf.reshape(temp_1, [ntrain, -1])
    train_a = tf.reshape(reader.read_field('coeff')[:ntrain, ::r, ::r], [ntrain, -1])
    train_a_smooth = tf.reshape(reader.read_field('Kcoeff')[:ntrain, ::r, ::r], [ntrain, -1])
    train_a_gradx = tf.reshape(reader.read_field('Kcoeff_x')[:ntrain, ::r, ::r], [ntrain, -1])
    train_a_grady = tf.reshape(reader.read_field('Kcoeff_y')[:ntrain, ::r, ::r], [ntrain, -1])
    train_u = tf.reshape(reader.read_field('sol')[:ntrain, ::r, ::r], [ntrain, -1])
    train_u64 = tf.reshape(reader.read_field('sol')[:ntrain, ::r, ::r], [ntrain, -1])

    reader.load_file(TEST_PATH)
    test_a = tf.reshape(reader.read_field('coeff')[:ntest, ::4, ::4], [ntest, -1])
    test_a_smooth = tf.reshape(reader.read_field('Kcoeff')[:ntest, ::4, ::4], [ntest, -1])
    test_a_gradx = tf.reshape(reader.read_field('Kcoeff_x')[:ntest, ::4, ::4], [ntest, -1])
    test_a_grady = tf.reshape(reader.read_field('Kcoeff_y')[:ntest, ::4, ::4], [ntest, -1])
    test_u = tf.reshape(reader.read_field('sol')[:ntest, ::4, ::4], [ntest, -1])

    a_normalizer = GaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)
    as_normalizer = GaussianNormalizer(train_a_smooth)
    train_a_smooth = as_normalizer.encode(train_a_smooth)
    test_a_smooth = as_normalizer.encode(test_a_smooth)
    agx_normalizer = GaussianNormalizer(train_a_gradx)
    train_a_gradx = agx_normalizer.encode(train_a_gradx)
    test_a_gradx = agx_normalizer.encode(test_a_gradx)
    agy_normalizer = GaussianNormalizer(train_a_grady)
    train_a_grady = agy_normalizer.encode(train_a_grady)
    test_a_grady = agy_normalizer.encode(test_a_grady)

    test_a = tf.reshape(test_a, [ntest, 61, 61])
    test_a_smooth = tf.reshape(test_a_smooth, [ntest, 61, 61])
    test_a_gradx = tf.reshape(test_a_gradx, [ntest, 61, 61])
    test_a_grady = tf.reshape(test_a_grady, [ntest, 61, 61])
    test_u = tf.reshape(test_u, [ntest, 61, 61])

    test_a16 = tf.reshape(test_a[:ntest, ::4, ::4], [ntest, -1])
    test_a_smooth16 = tf.reshape(test_a_smooth[:ntest, ::4, ::4], [ntest, -1])
    test_a_gradx16 = tf.reshape(test_a_gradx[:ntest, ::4, ::4], [ntest, -1])
    test_a_grady16 = tf.reshape(test_a_grady[:ntest, ::4, ::4], [ntest, -1])
    test_u16 = tf.reshape(test_u[:ntest, ::4, ::4], [ntest, -1])
    test_a31 = tf.reshape(test_a[:ntest, ::2, ::2], [ntest, -1])
    test_a_smooth31 = tf.reshape(test_a_smooth[:ntest, ::2, ::2], [ntest, -1])
    test_a_gradx31 = tf.reshape(test_a_gradx[:ntest, ::2, ::2], [ntest, -1])
    test_a_grady31 = tf.reshape(test_a_grady[:ntest, ::2, ::2], [ntest, -1])
    test_u31 = tf.reshape(test_u[:ntest, ::2, ::2], [ntest, -1])
    test_a = tf.reshape(test_a, [ntest, -1])
    test_a_smooth = tf.reshape(test_a_smooth, [ntest, -1])
    test_a_gradx = tf.reshape(test_a_gradx, [ntest, -1])
    test_a_grady = tf.reshape(test_a_grady, [ntest, -1])
    test_u = tf.reshape(test_u, [ntest, -1])

    u_normalizer = GaussianNormalizer(train_u)
    train_u = u_normalizer.encode(train_u)
    # test_u = y_normalizer.encode(test_u)

    meshgenerator = SquareMeshGenerator([[0, 1], [0, 1]], [s, s])
    grid = meshgenerator.get_grid()


    data_train = []
    for j in range(ntrain):
        x = tf.concat([grid, tf.reshape(train_a[j, :], [-1, 1]),
                       tf.reshape(train_a_smooth[j, :], [-1, 1]),
                       tf.reshape(train_a_gradx[j, :], [-1, 1]),
                       tf.reshape(train_a_grady[j, :], [-1, 1])], axis=-1)
        data_train.append(x)

    edge_index = meshgenerator.ball_connectivity(radius_test)
    print('shdsfhsdjhfsdj')


def test_reshape():
    X = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8]])
    Xtensor = tf.convert_to_tensor(X)
    Ytensor = tf.reshape(Xtensor, [-1])  # 行向量(n,1)省略了1
    print('Y:', Ytensor)

    Ytensor_cvec = tf.reshape(Xtensor, [8, -1])
    print('Y_row_vec:', Ytensor_cvec)

    mean_x2row = tf.reduce_mean(Xtensor, axis=-1, keepdims=True)
    print('mean_x2row:', mean_x2row)


if __name__ == "__main__":
    test_data_pre()
    # test_reshape()