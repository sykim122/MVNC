__author__ = 'xinchou'
import numpy as np
import theano


def statusMatrix(W):

    """ We got matrix W(similarity matrix)
        P is status matrix
        P(i, j) = W(i, j) / \sum(W(i, k))
        diag = 1/2
    """

    (row, col) = W.shape

    rowsumW = np.add.reduce(W, axis=1).transpose()

    # repeat matrix by W.shape
    dominateW = 2 * np.repeat(rowsumW, col).reshape((row, col))

    # defined W's diagonal = 0
    # Wd is matrix diagonal of W equal zero

    Wd = W - np.diag(np.diag(W))

    P = Wd / dominateW
    np.fill_diagonal(P, 1. / 2)

    return P


def kernelMatrix(W, param):

    """ We got matrix W(similarity matrix)
        S is kernel matrix
        diag = 0
    """

    (row, col) = W.shape
    K = param.K

    sortedW = np.sort(W, axis=1)

    # repeat matrix by W.shape
    rowKsum = np.add.reduce(sortedW[:, :K], axis=1).transpose()
    dominateW = np.repeat(rowKsum, col).reshape((row, col))

    Sd = W - np.diag(np.diag(W))

    S = Sd / dominateW

    return S

def t_status_matrix(W):
    (row, col) = W.shape
    rowsum_w = theano.tensor.sum(W, axis=1).T
    dominate_w = 2 * theano.tensor.repeat(rowsum_w, col).reshape((row,col))

    Wd = W - theano.tensor.diag(theano.tensor.diag(W))
    P = Wd / dominate_w
    theano.tensor.fill_diagonal(P, 1. / 2)

    return P

def t_kernel_matrix(W, param):
    (row, col) = W.shape
    K = param.K

    sortedW = theano.tensor.sort(W, axis=1)

    rowKsum = theano.tensor.sum(sortedW[:, :K], axis=1).T
    dominate_w = theano.tensor.repeat(rowKsum, col).reshape((row,col))

    Sd = W - theano.tensor.diag(theano.tensor.diag(W))
    S = Sd / dominate_w

    return S

