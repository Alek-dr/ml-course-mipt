from itertools import pairwise

import numpy as np

# import torch


class Module(object):
    """
    Basically, you can think of a module as of a something (black box)
    which can process `input` data and produce `ouput` data.
    This is like applying a function which is called `forward`:

        output = module.forward(input)

    The module should be able to perform a backward pass: to differentiate the `forward` function.
    More, it should be able to differentiate it if is a part of chain (chain rule).
    The latter implies there is a gradient from previous step of a chain rule.

        gradInput = module.backward(input, gradOutput)
    """

    def __init__(self):
        self.output = None
        self.gradInput = None
        self.training = True

    def forward(self, inputs):
        """
        Takes an input object, and computes the corresponding output of the module.
        """
        return self.updateOutput(inputs)

    def backward(self, input, gradOutput):
        """
        Performs a backpropagation step through the module, with respect to the given input.

        This includes
         - computing a gradient w.r.t. `input` (is needed for further backprop),
         - computing a gradient w.r.t. parameters (to update parameters while optimizing).
        """
        self.updateGradInput(input, gradOutput)
        self.accGradParameters(input, gradOutput)
        return self.gradInput

    def updateOutput(self, inputs):
        """
        Computes the output using the current parameter set of the class and input.
        This function returns the result which is stored in the `output` field.

        Make sure to both store the data in `output` field and return it.
        """

        # The easiest case:

        # self.output = input
        # return self.output

        pass

    def updateGradInput(self, inputs, gradOutput):
        """
        Computing the gradient of the module with respect to its own input.
        This is returned in `gradInput`. Also, the `gradInput` state variable is updated accordingly.

        The shape of `gradInput` is always the same as the shape of `input`.

        Make sure to both store the gradients in `gradInput` field and return it.
        """

        # The easiest case:

        # self.gradInput = gradOutput
        # return self.gradInput

        pass

    def accGradParameters(self, inputs, gradOutput):
        """
        Computing the gradient of the module with respect to its own parameters.
        No need to override if module has no parameters (e.g. ReLU).
        """
        pass

    def zeroGradParameters(self):
        """
        Zeroes `gradParams` variable if the module has params.
        """
        pass

    def getParameters(self):
        """
        Returns a list with its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def getGradParameters(self):
        """
        Returns a list with gradients with respect to its parameters.
        If the module does not have parameters return empty list.
        """
        return []

    def train(self):
        """
        Sets training mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = True

    def evaluate(self):
        """
        Sets evaluation mode for the module.
        Training and testing behaviour differs for Dropout, BatchNorm.
        """
        self.training = False

    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Module"


class Dropout(Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None

    def updateOutput(self, inputs):
        if self.training:
            self.mask = (np.random.rand(*inputs.shape) >= self.p).astype(np.float32)
            self.output = inputs * self.mask / (1.0 - self.p)
        else:
            self.output = inputs
        return self.output

    def updateGradInput(self, input, gradOutput):
        if self.training:
            self.gradInput = gradOutput * self.mask / (1.0 - self.p)
        else:
            self.gradInput = gradOutput
        return self.gradInput

    def __repr__(self):
        return "Dropout"


class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def updateOutput(self, inputs):
        # start with normalization for numerical stability
        inputs = np.subtract(inputs, inputs.max(axis=1, keepdims=True))

        batch_size = inputs.shape[0]
        exp_inputs = np.exp(inputs.astype(np.float128))
        self.output = exp_inputs / np.sum(exp_inputs, axis=1).reshape(batch_size, -1)
        return self.output

    def updateGradInput(self, inputs, gradOutput):
        grad = np.sum(gradOutput * self.output, axis=1, keepdims=True)
        return self.output * (gradOutput - grad)

    def __repr__(self):
        return "SoftMax"


class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 2D input of shape (n_samples, n_feature).
    """

    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        # This is a nice initialization
        stdv = 1.0 / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size=(n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, inputs):
        # Your code goes here. ################################################
        self.output = np.matmul(inputs, self.W.T) + self.b
        return self.output

    def updateGradInput(self, inputs, gradOutput):
        # Your code goes here. ################################################
        return np.matmul(gradOutput, self.W)

    def accGradParameters(self, inputs, gradOutput):
        # Your code goes here. ################################################
        self.gradW = np.matmul(inputs.T, gradOutput).T
        self.gradb = gradOutput.sum(axis=0)

    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def getParameters(self):
        return [self.W, self.b]

    def getGradParameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        s = self.W.shape
        q = "Linear %d -> %d" % (s[1], s[0])
        return q


class BatchNormalization(Module):
    EPS = 1e-3

    def __init__(self, alpha=0.0):
        super(BatchNormalization, self).__init__()
        self.alpha = alpha
        self.moving_mean = None
        self.moving_variance = None
        self.inv_std = None

    def updateOutput(self, inputs):
        if self.moving_mean is None:
            self.moving_mean = np.zeros(inputs.shape[1], dtype=np.float32)
            self.moving_variance = np.ones(inputs.shape[1], dtype=np.float32)

        if self.training:
            mean = inputs.mean(axis=0)
            var = inputs.var(axis=0)
            self.centered_input = inputs - mean
            self.inv_std = 1 / np.sqrt(var + self.EPS)
            self.output = self.centered_input * self.inv_std

            momentum = 1.0 - self.alpha
            self.moving_mean = self.alpha * self.moving_mean + (1.0 - self.alpha) * mean
            self.moving_variance = (
                momentum * self.moving_variance + (1 - momentum) * var
            )
        else:
            self.output = (inputs - self.moving_mean) / np.sqrt(
                self.moving_variance + self.EPS
            )

        return self.output

    def updateGradInput(self, inputs, gradOutput):
        if self.training:
            n = inputs.shape[0]

            grad_sum = np.sum(gradOutput, axis=0)
            grad_sum_ = np.sum(gradOutput * self.output, axis=0)

            self.gradInput = (
                (1.0 / n)
                * self.inv_std
                * (n * gradOutput - grad_sum - self.output * grad_sum_)
            )
            return self.gradInput

        self.gradInput = gradOutput / np.sqrt(self.moving_variance + self.EPS)
        return self.gradInput

    def __repr__(self):
        return "BatchNormalization"


class ChannelwiseScaling(Module):
    """
    Implements linear transform of input y = \gamma * x + \beta
    where \gamma, \beta - learnable vectors of length x.shape[-1]
    """

    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1.0 / np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)

    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output

    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput

    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis=0)
        self.gradGamma = np.sum(gradOutput * input, axis=0)

    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)

    def getParameters(self):
        return [self.gamma, self.beta]

    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]

    def __repr__(self):
        return "ChannelwiseScaling"


class Sequential(Module):
    """
    This class implements a container, which processes `input` data sequentially.

    `input` is processed by each module (layer) in self.modules consecutively.
    The resulting array is called `output`.
    """

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, inputs):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})


        Just write a little loop.
        """
        for module in self.modules:
            inputs = module.updateOutput(inputs)
        self.output = inputs
        return self.output

    def backward(self, inputs, gradOutput):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)


        !!!

        To ech module you need to provide the input, module saw while forward pass,
        it is used while computing gradients.
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass)
        and NOT `input` to this Sequential module.

        !!!

        """
        for curr_module, prev_module in pairwise(self.modules[::-1]):
            mod_inputs = prev_module.output
            gradOutput = curr_module.backward(mod_inputs, gradOutput)

        gradOutput = self.modules[0].backward(inputs, gradOutput)
        self.gradInput = gradOutput
        return self.gradInput

    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getParameters(self):
        """
        Should gather all parameters in a list.
        """
        return [x.getParameters() for x in self.modules]

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]

    def __repr__(self):
        string = "".join([str(x) + "\n" for x in self.modules])
        return string

    def __getitem__(self, x):
        return self.modules.__getitem__(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()

    def updateOutput(self, inputs):
        # start with normalization for numerical stability
        inputs = np.subtract(inputs, inputs.max(axis=1, keepdims=True))

        exp_ = np.exp(inputs)
        self.output = inputs - np.log(np.sum(exp_, axis=1))[..., None]
        return self.output

    def updateGradInput(self, input, gradOutput):
        exp_ = np.exp(self.output)
        grad_sum = np.sum(gradOutput, axis=1, keepdims=True)
        self.gradInput = gradOutput - exp_ * grad_sum
        return self.gradInput

    def __repr__(self):
        return "LogSoftMax"


class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super(LeakyReLU, self).__init__()

        self.slope = slope

    def updateOutput(self, inputs):
        less_zero = inputs < 0
        output = inputs.copy()
        output[less_zero] = self.slope * inputs[less_zero]
        self.output = output
        return self.output

    def updateGradInput(self, inputs, gradOutput):
        res = gradOutput.copy()
        ind_less_zero = inputs < 0
        res[ind_less_zero] = gradOutput[ind_less_zero] * self.slope
        self.gradInput = res
        return self.gradInput

    def __repr__(self):
        return "LeakyReLU"


class ELU(Module):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()

        self.alpha = alpha

    def updateOutput(self, inputs):
        less_zero = inputs <= 0
        output = inputs.copy()
        output[less_zero] = self.alpha * (np.exp(inputs[less_zero]) - 1.0)
        self.output = output
        return self.output

    def updateGradInput(self, inputs, gradOutput):
        less_zero = inputs <= 0
        grad = gradOutput.copy()
        grad[less_zero] *= self.output[less_zero] + self.alpha
        self.gradInput = grad
        return self.gradInput

    def __repr__(self):
        return "ELU"


class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()

    def updateOutput(self, inputs):
        self.output = np.log1p(np.exp(inputs))
        return self.output

    def updateGradInput(self, inputs, gradOutput):
        derr = np.zeros_like(inputs)
        pos_mask = inputs >= 0
        less_zero = inputs < 0
        derr[pos_mask] = 1.0 / (1.0 + np.exp(-inputs[pos_mask]))
        derr[less_zero] = np.exp(inputs[less_zero]) / (1.0 + np.exp(inputs[less_zero]))
        self.gradInput = gradOutput * derr
        return self.gradInput

    def __repr__(self):
        return "SoftPlus"
