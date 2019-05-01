from __future__ import division
import theano
import theano.tensor
import matrixOperate as mO
import numpy as np

__author__ = 'xinchou'


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
        Ss.append(mO.kernelMatrix(W, param))
        Ps.append(mO.statusMatrix(W))

    Ps = np.array(Ps)
    Ss = np.array(Ss)
    

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

    for W in Ws:
        Ss.append(mO.kernelMatrix(W, param))
        Ps.append(mO.statusMatrix(W))

    Ps = np.array(Ps)
    Ss = np.array(Ss)

    s_Ps = theano.shared(Ps, borrow=True)
    s_Ss = theano.shared(Ss, borrow=True)

    Fs = []
    for idx in range(len(Ws)):
        Fs.append(theano.function([], (s_Ss[idx] * (theano.tensor.sum(s_Ps) - s_Ps[idx]) / (len(Ws) - 1) * s_Ss[idx].transpose())))

    print "compile done"

    for iterate in range(T):
        for idx in range(len(Ws)):
            Ps_diag = np.diag(np.diag(Ps[idx]))

            # Ps[idx] = Ss[idx] * (np.add.reduce(Ps) - Ps[idx]) / (len(Ws) - 1) * Ss[idx].transpose()
            # Ps[idx] = Ss[idx] * (np.sum(Ps) - Ps[idx]) / (len(Ws) - 1) * Ss[idx].transpose()
            
            #Ps[idx] = (s_Ss[idx] * (theano.tensor.sum(s_Ps) - s_Ps[idx]) / (len(Ws) - 1) * s_Ss[idx].transpose() ).eval()
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
    shared_Ps = []
    shared_Ss = []
    

    for W in Ws:
        S = mO.kernelMatrix(W, param)
        Ss.append(S)
        shared_Ss.append(theano.shared(S, borrow=True))
        P = mO.statusMatrix(W)
        Ps.append(P)
        shared_Ps.append(theano.shared(P, borrow=True))

    Ps = np.array(Ps)
    Ss = np.array(Ss)

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
                            theano.tensor.diag(theano.tensor.diag(shared_P)),
            )
        )
        UpdatePs.append(
            theano.function([],
                            (),
                            updates=[(shared_Ps[idx], s_S*(theano.tensor.sum(shared_Ps) - s_P)/(len(Ws) - 1) * s_S.transpose())],
                            givens=[(s_P, shared_Ps[idx]), (s_S, shared_Ss[idx])]
            )
        )
        W = W / np.add.reduce(W, axis=1).transpose()
        NormalizePs.append(
            theano.function([],
                            (),
                            updates=[(shared_Ps[idx], s_P / theano.tensor.sum(s_P, axis=1).transpose())],
                            givens=[(s_P, shared_Ps[idx])]
            )
        )
        AdjustPs.append(
            theano.function([prev, now],
                            (),
                            updates=[(shared_Ps[idx], s_P - now + prev)],
                            givens=[(s_P, shared_Ps[idx])]
            )
        )

        GetPsDiag.append(
            theano.function([],
                            theano.tensor.diag(theano.tensor.diag(s_P)),
                            givens=[(s_P, shared_Ps[idx])]
            )
        )
        UpdatePs.append(
            theano.function([],
                            (),
                            updates=[(shared_Ps[idx], s_S*(theano.tensor.sum(shared_Ps) - s_P)/(len(Ws) - 1) * s_S.transpose())],
                            givens=[(s_P, shared_Ps[idx]), (s_S, shared_Ss[idx])]
            )
        )
        W = W / np.add.reduce(W, axis=1).transpose()
        NormalizePs.append(
            theano.function([],
                            (),
                            updates=[(shared_Ps[idx], s_P / theano.tensor.sum(s_P, axis=1).transpose())],
                            givens=[(s_P, shared_Ps[idx])]
            )
        )
        AdjustPs.append(
            theano.function([prev, now],
                            (),
                            updates=[(shared_Ps[idx], s_P - now + prev)],
                            givens=[(s_P, shared_Ps[idx])]
            )
        )

    print "compile done gpu2"

    """
    for iterate in range(T):
        for idx in range(len(Ws)):
            Ps_diag = np.diag(np.diag(Ps[idx]))

            # Ps[idx] = Ss[idx] * (np.add.reduce(Ps) - Ps[idx]) / (len(Ws) - 1) * Ss[idx].transpose()
            # Ps[idx] = Ss[idx] * (np.sum(Ps) - Ps[idx]) / (len(Ws) - 1) * Ss[idx].transpose()
            
            #Ps[idx] = (s_Ss[idx] * (theano.tensor.sum(s_Ps) - s_Ps[idx]) / (len(Ws) - 1) * s_Ss[idx].transpose() ).eval()
            Ps[idx] = Fs[idx](Ps[idx], Ss[idx])
            #normalize
            Ps[idx] = normalized(Ps[idx])

            Ps[idx] = Ps[idx] - np.diag(np.diag(Ps[idx])) + Ps_diag
    """

    for iterate in range(T):
        #print shared_Ps[0].get_value()
        for idx in range(len(Ws)):
            prev_P_diag = GetPsDiag[idx]()
            UpdatePs[idx]()
            NormalizePs[idx]()
            now_P_diag = GetPsDiag[idx]()
            AdjustPs[idx](prev_P_diag, now_P_diag)

    P = np.matrix(np.mean([p.get_value() for p in shared_Ps], axis=0))

    return (P + P.transpose()) / 2      # ensure affinity matrix is symmetrical

