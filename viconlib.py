import numpy as np


def rotx(x):
    re = np.matrix([[1.0, 0.0, 0.0, 0.0],
                    [0, np.cos(x), -np.sin(x), 0],
                    [0, np.sin(x), np.cos(x), 0],
                    [0, 0, 0, 1]])
    return re


def roty(x):
    re = np.matrix([[np.cos(x), 0, np.sin(x), 0],
                   [0, 1, 0, 0],
                   [-np.sin(x), 0, np.cos(x), 0],
                   [0, 0, 0, 1]])
    return re


def rotz(x):
    re = np.matrix([[np.cos(x), -np.sin(x), 0, 0],
                   [np.sin(x), np.cos(x), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return re


def transx(x):
    re = np.matrix([[1, 0, 0, x],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return re


def transy(y):
    re = np.matrix([[1, 0, 0, 0],
                   [0, 1, 0, y],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return re


def transz(z):
    re = np.matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, z],
                   [0, 0, 0, 1]])
    return re


# calculates the inverse of transformation function
def inver_t(t):  # is matrix
    r = t[0:3, 0:3]
    r_inver = r.transpose()
    p = t[0:3, 3]
    p_inver = -np.matmul(r_inver, p)
    t_inver = np.matrix([[r_inver[0, 0], r_inver[0, 1], r_inver[0, 2], p_inver[0, 0]],
                        [r_inver[1, 0], r_inver[1, 1], r_inver[1, 2], p_inver[1, 0]],
                        [r_inver[2, 0], r_inver[2, 1], r_inver[2, 2], p_inver[2, 0]],
                        [0, 0, 0, 1]])
    return t_inver


