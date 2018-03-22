
import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class AccuracyTopNum(function.Function):

    def __init__(self, num=5):
        if num < 1:
            raise ValueError("num is >= 1. {} is set". format(num))
        else:
            self.num       = num

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype == numpy.int32
        )

        t_ndim = type_check.eval(t_type.ndim)
        type_check.expect(
            x_type.ndim >= t_type.ndim,
            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2: t_ndim + 1] == t_type.shape[1:]
        )
        for i in six.moves.range(t_ndim + 1, type_check.eval(x_type.ndim)):
            type_check.expect(x_type.shape[i] == 1)

    def forward_cpu(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        pred = y.argsort().T[-self.num:]
        broad = xp.broadcast(t, pred)
        boolean = xp.array([te==pe for (te,pe) in broad]).reshape(pred.shape)
        return xp.array(boolean.max(axis=0).mean()),

    def forward_gpu(self, inputs):
        xp = cuda.get_array_module(*inputs)
        y, t = inputs

        pred = y.argsort().T[-self.num:]
        t_b, pred_b = xp.broadcast(t, pred).values
        return xp.array(t_b==pred_b).max(axis=0).mean(),


def accuracy_top_num(y, t, num=5):
    """Computes multiclass classification accuracy of the minibatch.

    For example, if teach label is in ``num`` of top of ``y``,
    the result is correct(i.e. this function is extended for compute top5)

    Args:
        y (Variable): Variable holding a matrix whose (i, j)-th element
            indicates the score of the class j at the i-th example.
        t (Variable): Variable holding an int32 vector of ground truth labels.
        num(int): Range which include correct label in top of ``y``.

    Returns:
        Variable: A variable holding a scalar array of the accuracy.

    .. note:: This function is non-differentiable.

    """
    return AccuracyTopNum(num=num)(y, t)



