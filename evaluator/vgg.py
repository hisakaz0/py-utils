
import sys

from chainer import Variable
from chainer import configuration
from chainer import cuda
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function
from chainer.functions import accuracy
from chainer import link
from chainer import reporter as reporter_module
from chainer import report
from chainer.training import extension
from chainer.training.extensions import Evaluator
import numpy

from functions.accuracy_top_num import accuracy_top_num


class VGGTestEvaluator(Evaluator):

    def __init__(self, iterator, target, shape,
            converter=convert.concat_examples,
            device=None, eval_hook=None, eval_func=None):

        super(VGGTestEvaluator, self).__init__(iterator, target,
                converter, device, eval_hook, eval_func=self.eval_func)

        xp = target.xp
        # number of crop of one image: 50
        # number of class in imagenet: 1000
        self.pool = Queue(Variable(xp.ndarray(shape).astype('f')))


    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func


        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)


        summary = reporter_module.DictSummary()
        flag, count = 0, 0

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if not isinstance(in_arrays, tuple):
                        raise TypeError
                    images, labels = in_arrays
                    res = Queue(target(images))

                    """ VGGのテストのフロ(ー)チャ(ート)
                    1. activationsが空になるまで繰り返す
                    2. pool_indexより、pool_activationsの空き領域にactivations
                        の先頭から入れる。pool_indexが0の場合、pool_labelを
                        同じlabelsで設定する。
                    3. pool_activationsの状態をチェック
                    4. 満タンなら5、そうでなければ1に移動する。
                    5. eval_funcを実行、pool_activationsを空にし、1へ移動する。
                    """

                    # activations_index = 0
                    while not res.is_full:
                        if self.pool.is_empty:
                            label = labels[res.index]

                        set_remain(self.pool, res)

                        if self.pool.is_full:
                            eval_func(label)
                            self.pool.index = 0 # reset
            flag += 1
            count += len(batch)
            if flag % 100 == 0:
                sys.stdout.write("\r{}/{} : {}".format(count, len(it.dataset),
                    {k:"{0:.5f}".format(v) for k,v in summary.compute_mean().items()} ))

            summary.add(observation)

        return summary.compute_mean()


    def eval_func(self, label):
        # chainer.functionのtype_checkを通過させるため、[]のwrapperをつける
        xp = cuda.get_array_module(self.pool.variable.data)
        mean_activations = Variable(xp.array(
            [xp.mean(self.pool.variable.data, axis=0)],
            dtype=xp.float32))
        label = Variable(xp.array(label, dtype=xp.int32))

        # cpuに移す
        mean_activations.to_cpu()
        label.to_cpu()
        mean_activations = mean_activations.data
        label = label.data

        # top1, top5の算出
        res_top1 = mean_activations.argmax()
        res_top5 = mean_activations.argsort()[0][-5:]

        top1 = 1.0 if label[0] == res_top1 else 0.0
        top5 = 1.0 if label[0] in res_top5 else 0.0

        report({'top1': top1, 'top5': top5})


class Queue:
    # TODO: QueueをPoolへ名前を変更する。適切じゃねぇ。

    """Manage length of the activations

    `data` is array of data.
    `begin` is first index of activations. This value is initialized with
    `begin` in input args of __init___. In default, this value is 0.
    `index` is marker which indicates empty item of last occupied item.
    `end`   is last index of activations.

    If `index` is 0, array of data is empty. Else if `index` is greater than
    `end`, array is full. Else, data has empty space with only `remain_len`.
    """


    def __init__(self, variable):
        if type(variable) != Variable:
            raise TypeError
        self.variable = variable
        self.begin  = 0
        self.index  = 0 # which indicates next empty location of last item in 'activations'
        self.end    = len(self.variable.data) - 1


    def __len__(self):
        return len(self.variable.data)


    @property
    def remain_len(self):
        return self.end - self.index + 1


    @property
    def is_empty(self):
        return self.index == 0


    @property
    def is_full(self):
        return self.index > self.end


def set_remain(a, b):
    """Set remain of a with remain of b

    Input args:
        a, b: instance of ActivationsBottle class"""
    remain_len = min(a.remain_len, b.remain_len)

    a_begin = a.index
    a_end   = a.index + remain_len - 1

    b_begin = b.index
    b_end   = b.index + remain_len - 1

    a.variable.data[a_begin : a_end] = b.variable.data[b_begin : b_end]

    a.index += remain_len # update index
    b.index += remain_len # update index


class VGGCaffeTestEvaluator(Evaluator):

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, num_labels=1000):

        super(VGGCaffeTestEvaluator, self).__init__(iterator, target,
                converter, device, eval_hook, eval_func=self.eval_func)

        xp = target.xp
        # number of crop of one image: 50
        # number of class in imagenet: 1000
        self.num_labels = num_labels
        self.pool = Queue(Variable(xp.ndarray((50, num_labels)).astype('f')))


    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func


        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)


        summary = reporter_module.DictSummary()

        count = 0
        flag = 0
        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if not isinstance(in_arrays, tuple):
                        raise TypeError
                    images, labels = in_arrays
                    prob, = target(inputs={'data': images}, outputs=['prob'])
                    res = Queue(prob)

                    """ VGGのテストのフロ(ー)チャ(ート)
                    1. activationsが空になるまで繰り返す
                    2. pool_indexより、pool_activationsの空き領域にactivations
                        の先頭から入れる。pool_indexが0の場合、pool_labelを
                        同じlabelsで設定する。
                    3. pool_activationsの状態をチェック
                    4. 満タンなら5、そうでなければ1に移動する。
                    5. eval_funcを実行、pool_activationsを空にし、1へ移動する。
                    """

                    # activations_index = 0
                    while not res.is_full:
                        if self.pool.is_empty:
                            label = labels[res.index]

                        set_remain(self.pool, res)

                        if self.pool.is_full:
                            eval_func(label)
                            self.pool.index = 0 # reset

            summary.add(observation)
            # XXX: print_reportで書き直す
            flag += 1
            count += len(batch)
            if flag % 100 == 0:
                sys.stdout.write("\r{}/{} : {}".format(count, len(it.dataset),
                    summary.compute_mean()))

        return summary.compute_mean()


    def eval_func(self, label):
        # chainer.functionのtype_checkを通過させるため、[]のwrapperをつける
        xp = cuda.get_array_module(self.pool.variable.data)
        sum_activations = xp.array([xp.sum(self.pool.variable.data, axis=0)],
                dtype=xp.float32)
        label = xp.array([label], dtype=xp.int32)

        top1_accuracy = accuracy(sum_activations, label)
        top5_accuracy = accuracy_top_num(sum_activations, label, 5)
        report({'top1': top1_accuracy, 'top5': top5_accuracy})


class VGGIndividualCaffeTestEvaluator(Evaluator):

    def __init__(self, iterator, target, converter=convert.concat_examples,
                 device=None, eval_hook=None, eval_func=None, num_labels=1000, print_interval=100):

        super(VGGIndividualCaffeTestEvaluator, self).__init__(iterator, target,
                converter, device, eval_hook, eval_func=self.eval_func)
        xp = target.xp
        # number of crop of one image: 50
        # number of class in imagenet: 1000
        self.num_labels = num_labels
        self.pool = Queue(Variable(xp.ndarray((50, 1000)).astype('f')))
        self.interval =  print_interval - 1

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func


        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        if it.shuffle == True:
            print("This evaluator accepts only sequential iterator")
            raise ValueError

        observers = [object() for _ in range(0, self.num_labels)]
        prefix = 'validation' + '/'
        reporter = reporter_module.get_current_reporter()
        for index, observer in enumerate(observers):
            o = (str(index), observer)
            reporter.add_observer(*o)
            reporter.add_observers(prefix, [o])

        summary = reporter_module.DictSummary()
        len_it  = len(it.dataset) // it.batch_size

        interval = 0
        for index, batch in enumerate(it):
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if not isinstance(in_arrays, tuple):
                        raise TypeError
                    images, labels = in_arrays
                    prob, = target(inputs={'data': images}, outputs=['prob'])
                    res = Queue(prob)

                    """ VGGのテストのフロ(ー)チャ(ート)
                    1. activationsが空になるまで繰り返す
                    2. pool_indexより、pool_activationsの空き領域にactivations
                        の先頭から入れる。pool_indexが0の場合、pool_labelを
                        同じlabelsで設定する。
                    3. pool_activationsの状態をチェック
                    4. 満タンなら5、そうでなければ1に移動する。
                    5. eval_funcを実行、pool_activationsを空にし、1へ移動する。
                    """

                    # activations_index = 0
                    while not res.is_full:
                        if self.pool.is_empty:
                            label = labels[res.index]

                        set_remain(self.pool, res)

                        if self.pool.is_full:
                            acc = eval_func(label)
                            report(acc, observers[int(label)])
                            self.pool.index = 0 # reset

            summary.add(observation)
            # print_reportだと、trainingに合わせないとだめなので、自作
            if interval >= self.interval:
                sys.stdout.write("\riteration: {}/{}". format(index, len_it))
                interval = 0
            else:
                interval += 1

        return summary.compute_mean()


    def eval_func(self, label):
        # chainer.functionのtype_checkを通過させるため、[]のwrapperをつける
        xp = cuda.get_array_module(self.pool.variable.data)
        sum_activations = xp.array([xp.sum(self.pool.variable.data, axis=0)],
                dtype=xp.float32)
        label = xp.array([label], dtype=xp.int32)

        top1_accuracy = accuracy(sum_activations, label)
        top5_accuracy = accuracy_top_num(sum_activations, label, 5)
        return {'top1': top1_accuracy, 'top5': top5_accuracy}

class VGGMultiLabelsTestEvaluator(Evaluator):

    def __init__(self, iterator, target, shape,
            extract_func=None, converter=convert.concat_examples,
            device=None, eval_hook=None, eval_func=None, num_labels=1000):

        super(VGGMultiLabelsTestEvaluator, self).__init__(iterator, target,
                converter, device, eval_hook, eval_func=self.eval_func)

        xp = target.xp
        self.extract_func = extract_func
        self.num_labels = num_labels
        self.pool = Queue(Variable(xp.ndarray(shape).astype('f')))

        if extract_func is None: # 'prob'の場合は自分で書く必要あり
            def extract(model, x):
                return model(x)
            self.extract_func = extract

    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        count = 0
        flag = 0
        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if not isinstance(in_arrays, tuple):
                        raise TypeError
                    images, labels = in_arrays
                    res = Queue(self.extract_func(target, images))

                    """ VGGのテストのフロ(ー)チャ(ート)
                    1. activationsが空になるまで繰り返す
                    2. pool_indexより、pool_activationsの空き領域にactivations
                        の先頭から入れる。pool_indexが0の場合、pool_labelを
                        同じlabelsで設定する。
                    3. pool_activationsの状態をチェック
                    4. 満タンなら5、そうでなければ1に移動する。
                    5. eval_funcを実行、pool_activationsを空にし、1へ移動する。
                    """

                    # activations_index = 0
                    while not res.is_full:
                        if self.pool.is_empty:
                            label = labels[res.index]

                        set_remain(self.pool, res)

                        if self.pool.is_full:
                            eval_func(label)
                            self.pool.index = 0 # reset

            summary.add(observation)
            # TODO: print_reportで書き直すことは、無理そう
            # evaluateはiterやepochの概念外。なのでロガーで書こう。
            flag += 1
            count += len(batch)
            if flag % 100 == 0:
                sys.stdout.write("\r{}/{} : {}".format(count, len(it.dataset),
                    summary.compute_mean()))

        return summary.compute_mean()

    def eval_func(self, label):
        # chainer.functionのtype_checkを通過させるため、[]のwrapperをつける
        xp = cuda.get_array_module(self.pool.variable.data)

        mean_activations = Variable(xp.array(
            [xp.mean(self.pool.variable.data, axis=0)],
            dtype=xp.float32))
        label = Variable(xp.array(label, dtype=xp.int32))

        # cpuに移す
        mean_activations.to_cpu()
        label.to_cpu()
        mean_activations = mean_activations.data
        label = label.data

        # top1, top5の算出
        res_top1 = mean_activations.argmax()
        res_top5 = mean_activations.argsort()[0][-5:]

        # ndarray(numpy) -> list(pure python)
        label.tolist()
        res_top1 = int(res_top1)
        res_top5.tolist()

        # label[0] は imageに対応する元々のlabel
        # label はcommon_root_labels
        ref_top1 = 1.0 if res_top1 in label             else 0.0
        ref_top5 = 1.0 if set(res_top5) & set(label)    else 0.0
        org_top1 = 1.0 if label[0] == res_top1          else 0.0
        org_top5 = 1.0 if label[0] in res_top5          else 0.0

        report({
            'ref/top1': ref_top1,
            'ref/top5': ref_top5,
            'org/top1': org_top1,
            'org/top5': org_top5 })

