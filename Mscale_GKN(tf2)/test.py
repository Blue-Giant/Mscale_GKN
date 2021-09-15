import numpy as np
import tensorflow as tf

import torch as t


def test_0():
    A = t.tensor([[1,2,3,4],[5,6,7,8]])
    B = A
    A[0] = t.tensor([10,11,12,13])
    print("A:",A)
    print('B:',B)


def test_1():
    A = tf.ones([4, 4])
    print(A)

    B = A


if __name__ == "__main__":
    test_0()
    test_1()