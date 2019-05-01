from __future__ import division
import theano
import matrixOperate as mO
import affinityMatrix as aM
import numpy as np
import scipy

__author__ = 'xinchou'


Machine_Epsilon = np.finfo(float).eps

def normalized(W):

    """ normalized calculated P
    """

    W = W / np.add.reduce(W, axis=1).transpose()

    return W


def SNF_ORIGINAL(Ws, param):

    """ main function in SNFtools
        input : param -- K : k nearest neighbour
                      +- T : iterations
        return unified similariy matrix

    """

    T = param.iterate

    # Ps, and Ss used for store P^{v} and S^{v} -- initialized
    Ps = []
    Ss = []

    for W in Ws:
        Ss.append(mO.kernelMatrix(W, param))
        Ps.append(mO.statusMatrix(W))

    Ps = np.array(Ps)
    Ss = np.array(Ss)

    for iterate in range(T):
        for idx in range(Ps.__len__()):
            Ps_diag = np.diag(np.diag(Ps[idx]))

            Ps[idx] = np.matrix(Ss[idx]) * np.matrix(np.add.reduce(Ps) - Ps[idx]) / (Ps.__len__() - 1) * np.matrix(Ss[idx]).transpose()
            Ps[idx] = normalized(Ps[idx])

            Ps[idx] = Ps[idx] - np.diag(np.diag(Ps[idx])) + Ps_diag

    P = np.matrix(np.add.reduce(Ps)) / Ps.__len__()

    return (P + P.transpose()) / 2      # ensure affinity matrix is symmetrical

def SNF_SPARSE(Ws, param):

    """ main function in SNFtools
        input : param -- K : k nearest neighbour
                      +- T : iterations
        return unified similariy matrix

    """

    T = param.iterate

    # Ps, and Ss used for store P^{v} and S^{v} -- initialized
    sPs = []
    sSs = []

    for W in Ws:
        k = np.int(.9 * W.shape[1])
        S = mO.kernelMatrix(W, param)
        P = mO.statusMatrix(W)
        S[S < np.partition(S, k, axis=1)[:,k]] = 0
        P[P < np.partition(P, k, axis=1)[:,k]] = 0
        
        sS = scipy.sparse.dia_matrix(S)
        sP = scipy.sparse.dia_matrix(P)

        sSs.append(sS)
        sPs.append(sP)

    Ps = np.array(sPs)
    Ss = np.array(sSs)

    for iterate in range(T):
        for idx in range(Ps.__len__()):
            
            #Ps_diag = Ps[idx].multiply(scipy.sparse.eye(Ps[idx].shape[0], Ps[idx].shape[1]))
            Ps_diag = scipy.sparse.diags([Ps[idx].data[Ps[idx].offsets[0]:(Ps[idx].shape[0]+Ps[idx].offsets[0])]], [0])
            #print Ps_diag

            Ps[idx] = Ss[idx] * (np.add.reduce(Ps) - Ps[idx]) / (Ps.__len__() - 1) *Ss[idx].transpose()
            #Ps[idx] = normalized(Ps[idx])
            Ps[idx] = Ps[idx].multiply(scipy.sparse.dia_matrix(( 1 / Ps[idx].sum(axis=1))))

            Ps[idx] = Ps[idx] - scipy.sparse.diags([Ps[idx].data[Ps[idx].offsets[0]:(Ps[idx].shape[0]+Ps[idx].offsets[0])]], [0]) + Ps_diag

    sP = (np.add.reduce(Ps)) / Ps.__len__()

    return (sP + sP.transpose()) / 2      # ensure affinity matrix is symmetrical

def SNF(Ws, param):

    """ main function in SNFtools
        input : param -- K : k nearest neighbour
                      +- T : iterations
        return unified similariy matrix

    """

    T = param.iterate

    # Ps, and Ss used for store P^{v} and S^{v} -- initialized
    Ps = []
    Ss = []

    for W in Ws:
        Ss.append(np.matrix(mO.kernelMatrix(W, param)))
        Ps.append(mO.statusMatrix(W))

    Ps = np.array(Ps)
    #Ss = np.array(Ss)
    

    for iterate in range(T):
        for idx in range(len(Ws)):
            Ps_diag = np.diag(np.diag(Ps[idx]))

            # Ps[idx] = Ss[idx] * (np.add.reduce(Ps) - Ps[idx]) / (len(Ws) - 1) * Ss[idx].transpose()
            Ps[idx] = Ss[idx] * (np.sum(Ps) - Ps[idx]) / (len(Ws) - 1) * Ss[idx].transpose()
            Ps[idx] = normalized(Ps[idx])

            Ps[idx] = Ps[idx] - np.diag(np.diag(Ps[idx])) + Ps_diag

    P = np.matrix(np.mean(Ps, axis=0))

    return (P + P.transpose()) / 2      # ensure affinity matrix is symmetrical


def SNF_GPU(Ws, param):

    """ main function in SNFtools
        input : param -- K : k nearest neighbour
                      +- T : iterations
        return unified similariy matrix

    """

    T = param.iterate

    # Ps, and Ss used for store P^{v} and S^{v} -- initialized
    Ps = []
    Ss = []
    s_Ss = []

    for W in Ws:
        S = np.matrix(mO.kernelMatrix(W, param))
        s_Ss.append(theano.shared(S, borrow=True))
        Ps.append(mO.statusMatrix(W))

    Ps = np.array(Ps)

    s_Ps = theano.shared(Ps, borrow=True)

    Fs = []
    for idx in range(len(Ws)):
        Fs.append(theano.function([], theano.tensor.dot(theano.tensor.dot(s_Ss[idx],(theano.tensor.sum(s_Ps) - s_Ps[idx]) / (len(Ws) - 1)), s_Ss[idx].T)))

    print "compile done"

    for iterate in range(T):
        for idx in range(len(Ws)):
            Ps_diag = np.diag(np.diag(Ps[idx]))

            Ps[idx] = Fs[idx]()
            Ps[idx] = normalized(Ps[idx])

            Ps[idx] = Ps[idx] - np.diag(np.diag(Ps[idx])) + Ps_diag

    P = np.matrix(np.mean(Ps, axis=0))

    return (P + P.transpose()) / 2      # ensure affinity matrix is symmetrical

def SNF_GPU2(Ws, param):
    """ main function in SNFtools
        input : param -- K : k nearest neighbour
                      +- T : iterations
        return unified similariy matrix

    """

    T = param.iterate

    # Ps, and Ss used for store P^{v} and S^{v} -- initialized
    Ps = []
    Ss = []
    

    for W in Ws:
        S = np.matrix(mO.kernelMatrix(W, param))
        Ss.append(theano.shared(S, borrow=True))
        P = np.matrix(mO.statusMatrix(W))
        Ps.append(theano.shared(P, borrow=True))

    #Ps = np.array(Ps)
    #Ss = np.array(Ss)

    s_P = theano.tensor.fmatrix('s_P')
    s_S = theano.tensor.fmatrix('s_S')

    GetPsDiag = []
    UpdatePs = []
    NormalizePs = []
    AdjustPs = []
    prev, now = theano.tensor.fmatrices('prev','now')
    for idx in range(len(Ws)):
        GetPsDiag.append(
            theano.function([],
                            theano.tensor.diag(theano.tensor.diag(Ps[idx])),
            )
        )
        UpdatePs.append(
            theano.function([],
                            (),
                            updates=[(Ps[idx], theano.tensor.dot(theano.tensor.dot(s_S,(theano.tensor.sum(Ps, axis=0) - s_P)/(len(Ws) - 1)),  s_S.T))],
                            givens=[(s_P, Ps[idx]), (s_S, Ss[idx])]
            )
        )
        NormalizePs.append(
            theano.function([],
                            (),
                            updates=[(Ps[idx], s_P / theano.tensor.sum(s_P, axis=1).transpose())],
                            givens=[(s_P, Ps[idx])]
            )
        )
        AdjustPs.append(
            theano.function([prev, now],
                            (),
                            updates=[(Ps[idx], s_P - now + prev)],
                            givens=[(s_P, Ps[idx])]
            )
        )

    print "compile done gpu2"

    for iterate in range(T):
        for idx in range(len(Ws)):
            prev_P_diag = GetPsDiag[idx]()
            UpdatePs[idx]()
            NormalizePs[idx]()
            now_P_diag = GetPsDiag[idx]()
            AdjustPs[idx](prev_P_diag, now_P_diag)

    P = np.matrix(np.mean([p.get_value() for p in Ps], axis=0))

    return (P + P.transpose()) / 2      # ensure affinity matrix is symmetrical

def SNF_GPU_NOSHARE(Ws, param):

    """ main function in SNFtools
        input : param -- K : k nearest neighbour
                      +- T : iterations
        return unified similariy matrix

    """

    T = param.iterate

    # Ps, and Ss used for store P^{v} and S^{v} -- initialized
    Ps = []
    Ss = []

    for W in Ws:
        S = np.matrix(mO.kernelMatrix(W, param))
        Ss.append(S)
        P = mO.statusMatrix(W)
        Ps.append(P)

    Ps = np.array(Ps)

    s_P = theano.tensor.fmatrix('s_P')
    s_S = theano.tensor.fmatrix('s_S')

    F = theano.function([s_P, s_S], theano.tensor.dot(theano.tensor.dot(s_S, (theano.tensor.sum(Ps, axis=0) - s_P) / (len(Ws) - 1)), s_S.T))

    print "compile done"

    for iterate in range(T):
        for idx in range(len(Ws)):
            Ps_diag = np.diag(np.diag(Ps[idx]))

            Ps[idx] = F(Ps[idx], Ss[idx])
            Ps[idx] = normalized(Ps[idx])

            Ps[idx] = Ps[idx] - np.diag(np.diag(Ps[idx])) + Ps_diag

    P = np.matrix(np.mean(Ps, axis=0))

    return (P + P.transpose()) / 2      # ensure affinity matrix is symmetrical
