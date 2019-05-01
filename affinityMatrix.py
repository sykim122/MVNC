__author__ = 'xinchou'
import numpy as np
import theano
import theano

import scipy

Machine_Epsilon = np.finfo('float32').eps

def standardNormalization(origin_d):

    """ normalization each column of origin_d to
        have mean = 0 and standard deviation = 1
        standn = (x - mean / std)
    """
    normal_dat = []
    for data in origin_d:
        mean = np.mean(data, axis=0)
        sd = np.std(data, axis=0)
        sd[sd == 0] = 1.0
        normal_dat.append(np.asmatrix((data - mean)/sd))

    return np.array(normal_dat)


# use decorator to defined different distance function

def dist2matrix(func):

    """ from origin_data to pair-wised disatance matrix
        n sample x n sample
    """
    def __wrapper(normal_dat):

        w = []
        for data in normal_dat:
            dist_matrix = func(np.asmatrix(data))
            w.append(dist_matrix)

        return np.array(w)

    return __wrapper


@dist2matrix
def euclidDist(x):

    """ calculate \sum(x_{i}{k}-x_{j}{k})^2
    """
    s_x = theano.shared(x.astype('float32'), borrow=True)
    s_sumsqX = (s_x*s_x).sum(axis=1)
    s_square_x = s_sumsqX + s_sumsqX.T
    f_square_x = theano.function([], s_square_x)
    square_x = f_square_x()
    print 'compiled square_x'
    s_product_x = 2 * theano.tensor.dot(s_x , s_x.T)
    f_product_x = theano.function([], s_product_x)
    product_x = f_product_x()
    print 'compiled product_x'

    return square_x - product_x


@dist2matrix
def manhatDist(x):

    """ calculated \sum(x_{i}{k}-x_{j}{k})
    """
    sumsqX = np.add.reduce(x, axis=1)

    return np.abs(sumsqX - sumsqX.transpose())


def affinityMatrix(Diff, param):

    """ similarity matrix generated
        w(i,j) = exp(-dist(xi, xj) / mu / eps)
        eps = (mean(D(xi, Ni)) + mean(D(xj, Nj)) + Dist(xi, xj)) / 3
    """
 
    row, col = Diff.shape

    Diff_mat = (Diff + Diff.transpose()) / 2            # the distance pair-wised matrix


    Diff_mat_sort = Diff_mat - np.diag(np.diag(Diff_mat))    # set Diff's diagonal to 0
    Diff_mat_sort = np.sort(Diff_mat_sort, axis=0)

    # calculated the epsilon
    K_dist = np.mean(Diff_mat_sort[:param.K], axis=0)
    epsilon = (K_dist + K_dist.transpose()) / 3 * 2 + Diff_mat / 3 + Machine_Epsilon

    W = np.exp(-(Diff_mat / (param.eps * epsilon)))

    W = (W + W.transpose()) / 2
 
    return W

def t_euclid_dist(x):
    sumsq_x = (x*x).sum(axis=1)
    square_x = sumsq_x + sumsq_x.T
    product_x = 2 * theano.tensor.dot(x, x.T)

    return square_x - product_x
    
def t_affinity_matrix(diff, param):
    row, col = diff.shape
    diff_mat = (diff + diff.T)

    diff_mat_sort = theano.tensor.sort(diff_mat - theano.tensor.diag(theano.tensor.diag(diff_mat)), axis = 0)
    k_diff = theano.tensor.mean(diff_mat_sort[:param.K], axis=0)
    epsilon = (k_diff + k_diff.T) / 3 * 2 + diff_mat / 3 + Machine_Epsilon

    W = theano.tensor.exp(-(diff_mat / (param.eps * epsilon)))
    W = (W + W.T) / 2

    return W

def euclidAffinity(Ps, param):
    #s_data = theano.shared(data, borrow=True)
    x = theano.tensor.fmatrix('x')

    eu_dist = t_euclid_dist(x)
    aff_mat = t_affinity_matrix(eu_dist, param)

    for P in Ps:
	s_x = theano.shared(P, borrow=True)
        print theano.function([], aff_mat, givens=[(x, s_x)])()



