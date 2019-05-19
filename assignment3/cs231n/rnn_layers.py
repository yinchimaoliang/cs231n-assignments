from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    temp1 = np.dot(x, Wx)
    temp2 = np.dot(prev_h, Wh)
    cache = (x, prev_h, Wx, Wh, temp1 + temp2 + b)
    next_h = np.tanh(temp1 + temp2 + b)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x = cache[0]
    h = cache[1]
    Wx = cache[2]
    Wh = cache[3]
    # cache[4]对应着公式中的a
    cacheD=cache[4]
    N,H=h.shape
    # 计算激活函数的导数
    temp = np.ones((N,H))-np.square(np.tanh(cacheD))
    delta = np.multiply(temp,dnext_h)
    # 计算x的梯度
    tempx = np.dot(Wx,delta.T)
    dx=tempx.T
    # h的提督
    temph = np.dot(Wh,delta.T)
    dprev_h=temph.T
    # Wxh的梯度
    dWx = np.dot(x.T,delta)
    # Whh
    dWh = np.dot(h.T,delta)
    # b的梯度
    tempb = np.sum(delta,axis=0)
    db=tempb.T
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, D = x.shape
    N, H = h0.shape
    prev_h = h0  # 之前公式中对应的a
    h1=np.empty([N,T,H]) # 隐藏状态h的序列
    h2=np.empty([N,T,H]) # 滞后h一个时间点
    h3=np.empty([N,T,H])
    for i in range(0, T): #单步前向传播
        temp_h, cache_temp = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b) #记录下需要的变量
        h3[:,i,:]=prev_h
        prev_h=temp_h
        h2[:,i,:]=temp_h
        h1[:,i,:]=cache_temp[4]
    cache=(x,h3,Wx,Wh,h1)
    h = h2
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H).

    NOTE: 'dh' contains the upstream gradients produced by the
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    x = cache[0]
    N, T, D = x.shape
    N, T, H = dh.shape
    # 初始化
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros(H)
    dout = dh
    dx = np.empty([N, T, D])
    dh = np.empty([N, T, H])
    # 当前时刻隐藏状态对应的梯度
    hnow = np.zeros([N, H])

    for k in range(0, T):
        i = T - 1 - k
        # 我们要注意，除了上一层传来的梯度，我们每一层都有输出，对应的误差函数也会传入梯度
        hnow = hnow + dout[:, i, :]
        cacheT = (cache[0][:, i, :], cache[1][:, i, :], cache[2], cache[3], cache[4][:, i, :])
        # 单步反向传播
        dx_temp, dprev_h, dWx_temp, dWh_temp, db_temp = rnn_step_backward(hnow, cacheT)
        hnow = dprev_h
        dx[:, i, :] = dx_temp
        # 将每一层共享的参数对应的梯度相加
        dWx = dWx + dWx_temp
        dWh = dWh + dWh_temp
        db = db + db_temp

    dh0 = hnow
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    out = W[x]
    cache = (x, W)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    dW = np.zeros(W.shape)


    np.add.at(dW, x, dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Note that a sigmoid() function has already been provided for you in this file.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    N, H = prev_h.shape
    A = x.dot(Wx) + prev_h.dot(Wh) + b
    ai = A[:, 0:H]
    af = A[:, H:2 * H]
    ao = A[:, 2 * H:3 * H]
    ag = A[:, 3 * H:4 * H]

    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)

    next_c = np.multiply(f, prev_c) + np.multiply(i, g)
    next_h = np.multiply(o, np.tanh(next_c))

    cache = (x, prev_h, prev_c, i, f, o, g, Wx, Wh, next_c, A)
    return next_h, next_c, cache
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    输入:
    - dnext_h: 下一层传来关于h的梯度, 维度 (N, H)
    - dnext_c: 下一层传来关于c的梯度, 维度 (N, H)
    - cache: 前向传播存的变量

    输出
    - dx: 输入x的梯度, 维度(N, D)
    - dprev_h: 上一层隐藏状态的梯度, 维度(N, H)
    - dprev_c: 上一层单元状态的梯度, 维度(N, H)
    - dWx: 权值矩阵Wxh的梯度, 维度(D, 4H)
    - dWh: 权值矩阵Whh的梯度, 维度(H, 4H)
    - db: 偏差值b的梯度，维度（4H,）
    """
    #提取cache中的变量
    N,H=dnext_h.shape
    f=cache[4]
    o=cache[5]
    i=cache[3]
    g=cache[6]
    nc=cache[9]
    prev_c=cache[2]
    prev_x=cache[0]
    prev_h=cache[1]
    A=cache[10]
    ai=A[:,0:H]
    af=A[:,H:2*H]
    ao=A[:,2*H:3*H]
    ag=A[:,3*H:4*H]
    Wx=cache[7]
    Wh=cache[8]
    #计算到c_t-1的梯度
    dc_c=np.multiply(dnext_c,f)
    dc_h_temp=np.multiply(dnext_h,o)
    temp = np.ones_like(nc)-np.square(np.tanh(nc))
    temp2=np.multiply(temp,f)
    dprev_c=np.multiply(temp2,dc_h_temp)+dc_c

    #计算(dE/dh)(dh/dc)
    dc_from_h=np.multiply(dc_h_temp,temp)

    dtotal_c=dc_from_h+dnext_c

    #计算到o,f,i,g的梯度
    tempo=np.multiply(np.tanh(nc),dnext_h)
    tempf=np.multiply(dtotal_c,prev_c)
    tempi=np.multiply(dtotal_c,g)
    tempg=np.multiply(dtotal_c,i)

    #计算到ao,ai,af,ag的梯度
    tempao=np.multiply(tempo,np.multiply(o,np.ones_like(o)-o))
    tempai=np.multiply(tempi,np.multiply(i,np.ones_like(o)-i))
    tempaf=np.multiply(tempf,np.multiply(f,np.ones_like(o)-f))
    dtanhg = np.ones_like(ag)-np.square(np.tanh(ag))
    tempag=np.multiply(tempg,dtanhg)

    #计算各参数的梯度
    TEMP=np.concatenate((tempai,tempaf,tempao,tempag),axis=1)
    dx=TEMP.dot(Wx.T)
    dprev_h=TEMP.dot(Wh.T)
    xt=prev_x.T
    dWx=xt.dot(TEMP)
    ht=prev_h.T
    dWh=ht.dot(TEMP)
    db=np.sum(TEMP,axis=0).T

    return dx, dprev_h, dprev_c, dWx, dWh, db



def lstm_forward(x, h0, Wx, Wh, b):
    """

    输入:
    - x: 整个序列的输入数据, 维度 (N, T, D).
    - h0: 初始隐藏层, 维度 (N, H)
    - Wx: 权值矩阵Wxh, 维度 (D, H)
    - Wh: 权值矩阵Whh, 维度 (H, H)
    - b: 偏差值，维度 (H,)

    输出:
    - h: 整个序列的隐藏状态, 维度 (N, T, H).
    - cache: 反向传播时需要的变量

    """
    N,T,D=x.shape
    N,H=h0.shape
    prev_h=h0
    #以下的变量为反向传播时所需
    h3=np.empty([N,T,H])
    h4=np.empty([N,T,H])
    I=np.empty([N,T,H])
    F=np.empty([N,T,H])
    O=np.empty([N,T,H])
    G=np.empty([N,T,H])
    NC=np.empty([N,T,H])
    AT=np.empty([N,T,4*H])
    h2=np.empty([N,T,H])
    prev_c=np.zeros_like(prev_h)
    for i in range(0, T):
        h3[:,i,:]=prev_h
        h4[:,i,:]=prev_c
        #单步前向传播
        next_h, next_c, cache_temp = lstm_step_forward(x[:,i,:], prev_h, prev_c, Wx, Wh, b)
        prev_h=next_h
        prev_c=next_c
        h2[:,i,:]=prev_h
        I[:,i,:]=cache_temp[3]
        F[:,i,:]=cache_temp[4]
        O[:,i,:]=cache_temp[5]
        G[:,i,:]=cache_temp[6]
        NC[:,i,:]=cache_temp[9]
        AT[:,i,:]=cache_temp[10]

    cache=(x,h3,h4,I,F,O,G,Wx,Wh,NC,AT)



    return h2, cache



def lstm_backward(dh, cache):
    """
    输入:
    - dh: 损失函数关于每一个隐藏层的梯度, 维度 (N, T, H)
    - cache: 前向传播时存的变量
    输出
    - dx: 每一层输入x的梯度, 维度(N, T, D)
    - dh0: 初始隐藏状态的梯度, 维度(N, H)
    - dWx: 权值矩阵Wxh的梯度, 维度(D, 4H)
    - dWh: 权值矩阵Whh的梯度, 维度(H, 4H)
    - db: 偏差值b的梯度，维度（4H,）
    """
    x=cache[0]
    N,T,D=x.shape
    N,T,H=dh.shape

    dWx=np.zeros((D,4*H))
    dWh=np.zeros((H,4*H))
    db=np.zeros(4*H)
    dout=dh
    dx=np.empty([N,T,D])
    hnow=np.zeros([N,H])
    cnow=np.zeros([N,H])
    for k in range(0, T):
        i=T-1-k
        hnow=hnow+dout[:,i,:]
        cacheT=(cache[0][:,i,:],cache[1][:,i,:],cache[2][:,i,:],cache[3][:,i,:],cache[4][:,i,:],cache[5][:,i,:],cache[6][:,i,:],cache[7],cache[8],cache[9][:,i,:],cache[10][:,i,:])

        dx_temp, dprev_h, dprev_c, dWx_temp, dWh_temp, db_temp = lstm_step_backward(hnow, cnow, cacheT)
        hnow=dprev_h
        cnow=dprev_c
        dx[:,i,:]=dx_temp
        dWx=dWx+dWx_temp
        dWh=dWh+dWh_temp
        db=db+db_temp

    dh0=hnow

    return dx, dh0, dWx, dWh, db



def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
