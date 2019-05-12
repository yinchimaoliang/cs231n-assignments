import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  X_num = X.shape[0]
  class_num = W.shape[1]
  loss = 0
  for i in range(X_num):
    score = X[i].dot(W)
    # 为避免数值不稳定的问题，每个分值向量都减去向量中的最大值
    score -= np.max(score)
    # print(score)
    exp = np.exp(score)
    sum_temp = np.sum(exp)
    p = lambda k: np.exp(score[k]) / sum_temp
    loss += -np.log(p(y[i]))  # 每一个图像的损失值都要加一起，之后再求均值

    # 计算梯度
    for k in range(class_num):
      p_k = p(k)
      dW[:, k] += (p_k - (k == y[i])) * X[i]

  loss /= X_num
  loss += 0.5 * reg * np.sum(W * W)  # 参见知识点中的loss函数公式
  dW /= X_num
  dW += reg * W
# print(X_num)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # 插播softmax.py里的第二个函数段
  num_train = X.shape[0]
  f = X.dot(W)
  f -= np.max(f, axis=1, keepdims=True)
  sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
  p = np.exp(f) / sum_f

  loss = np.sum(-np.log(p[np.arange(num_train), y]))

  ind = np.zeros_like(p)
  ind[np.arange(num_train), y] = 1
  dW = X.T.dot(p - ind)

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  # 插播结束
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

