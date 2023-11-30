import torch
from torch import nn
import numpy as np

from typing import List, Tuple, Optional


class AbstractTransformer:

    def __init__(self):
        # lower and upper bounds for the output of this layer
        self.lb = None
        self.ub = None

        # weights and biases of the equation y <= W_ub x + b_ub and y >= W_lb x + b_lb
        # where x is the activation of the layer after self.equation_transformer
        self.equation_ub_weights = None
        self.equation_lb_weights = None

        self.equation_ub_bias = None
        self.equation_lb_bias = None

        # the layer the above equation is referring to
        self.equation_transformer = None

        # the transformer for the layer before this one
        self.previous_transformer = None

        # the depth of this layer in the network
        self.depth = None

        # the depth of the layer we are backsubstituting into
        self.backsub_depth = None

        # this is only used for ReLU and LeakyReLU
        self.alphas = None

    def calculate(self):
        pass

    def backward(self):
        pass

    def __str__(self):
        return f"{type(self).__name__} at depth {self.depth}"


class InputTransformer(AbstractTransformer):

    def __init__(self, inputs: torch.Tensor, eps: float, flatten: bool):
        super().__init__()
        self.inputs = inputs
        self.eps = eps

        self.lb = inputs - eps
        self.ub = inputs + eps

        if flatten:
            self.lb = self.lb.flatten()
            self.ub = self.ub.flatten()

        self.lb = torch.clamp(self.lb, min=0, max=1)
        self.ub = torch.clamp(self.ub, min=0, max=1)

        self.depth = 0
        self.backsub_depth = 0


class LinearTransformer(AbstractTransformer):

    def __init__(self, module: nn.Linear, previous_transformer: AbstractTransformer, depth: int):

        super().__init__()
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.in_features = module.in_features
        self.out_features = module.out_features

        self.previous_transformer = previous_transformer
        self.depth = depth
        self.backsub_depth = depth

    def calculate(self):

        # self.equation_transformer = self.previous_transformer
        # we backsub recursively until we reach the layer we are backsubstituting into
        if self.backsub_depth < self.previous_transformer.depth:
            # print(f"Backsub called from {self}")
            (self.equation_transformer,
             self.equation_ub_weights,
             self.equation_lb_weights,
             self.equation_ub_bias,
             self.equation_lb_bias) = self.previous_transformer.backward(self.weight,
                                                                         self.weight,
                                                                         self.bias,
                                                                         self.bias,
                                                                         self.backsub_depth)
        else:
            self.equation_transformer = self.previous_transformer
            self.equation_ub_weights = self.weight
            self.equation_lb_weights = self.weight
            self.equation_ub_bias = self.bias
            self.equation_lb_bias = self.bias

        lb = self.equation_transformer.lb
        ub = self.equation_transformer.ub

        # use previous upper bound and lower bound for positive weights, otherwise swap
        positive_ub_weights = (self.equation_ub_weights >= 0).int() * self.equation_ub_weights
        negative_ub_weights = (self.equation_ub_weights < 0).int() * self.equation_ub_weights

        positive_lb_weights = (self.equation_lb_weights >= 0).int() * self.equation_lb_weights
        negative_lb_weights = (self.equation_lb_weights < 0).int() * self.equation_lb_weights

        self.lb = positive_lb_weights @ lb + negative_lb_weights @ ub + self.equation_lb_bias
        self.ub = positive_ub_weights @ ub + negative_ub_weights @ lb + self.equation_ub_bias

    def backward(self,
                 ub_weights: torch.Tensor,
                 lb_weights: torch.Tensor,
                 ub_bias: torch.Tensor,
                 lb_bias: torch.Tensor,
                 backsub_depth: int) -> Tuple[
        AbstractTransformer, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        new_ub_bias = ub_bias + ub_weights @ self.bias
        new_lb_bias = lb_bias + lb_weights @ self.bias

        new_ub_weights = ub_weights @ self.weight
        new_lb_weights = lb_weights @ self.weight

        if backsub_depth >= self.previous_transformer.depth:
            return self.previous_transformer, new_ub_weights, new_lb_weights, new_ub_bias, new_lb_bias
        else:
            return self.previous_transformer.backward(new_ub_weights, new_lb_weights, new_ub_bias, new_lb_bias,
                                                      backsub_depth)


class ReLUTransformer(AbstractTransformer):

    def __init__(self, module: nn.ReLU, previous_transformer: AbstractTransformer, depth: int):

        super().__init__()
        self.equation_transformer = previous_transformer
        self.previous_transformer = previous_transformer
        self.depth = depth
        self.backsub_depth = depth

        self.alphas = None

    def calculate(self):

        lb = self.previous_transformer.lb
        ub = self.previous_transformer.ub

        crossing = (lb < 0) & (ub > 0)
        negative = (ub <= 0)
        positive = (lb >= 0)

        lmbda = ub / (ub - lb)
        self.ub_weights = negative.int() * torch.zeros(ub.shape) + positive.int() * torch.ones(
            ub.shape) + crossing.int() * lmbda
        self.ub_weights = torch.diag(self.ub_weights)
        self.ub_bias = -crossing.int() * lmbda * lb

        # initialize self.alphas uniformly random between 0 and 1
        if self.alphas is None:
            self.alphas = torch.empty(lb.shape, requires_grad=True)
            nn.init.uniform_(self.alphas, 0, 1)
            #self.alphas = torch.rand(lb.shape, requires_grad=True) 
        self.lb_weights = negative.int() * torch.zeros(ub.shape) + positive.int() * torch.ones(
            ub.shape) + crossing.int() * self.alphas
        self.lb_weights = torch.diag(self.lb_weights)
        self.lb_bias = torch.zeros(lb.shape)

        # we backsub recursively until we reach the layer we are backsubstituting into
        if self.backsub_depth < self.previous_transformer.depth:
            (self.equation_transformer,
             self.equation_ub_weights,
             self.equation_lb_weights,
             self.equation_ub_bias,
             self.equation_lb_bias) = self.previous_transformer.backward(self.ub_weights,
                                                                         self.lb_weights,
                                                                         self.ub_bias,
                                                                         self.lb_bias,
                                                                         self.backsub_depth)
        else:
            self.equation_transformer = self.previous_transformer
            self.equation_ub_weights = self.ub_weights
            self.equation_lb_weights = self.lb_weights
            self.equation_ub_bias = self.ub_bias
            self.equation_lb_bias = self.lb_bias

        lb = self.equation_transformer.lb
        ub = self.equation_transformer.ub

        # use previous upper bound and lower bound for positive weights, otherwise swap
        positive_ub_weights = (self.equation_ub_weights >= 0).int() * self.equation_ub_weights
        negative_ub_weights = (self.equation_ub_weights < 0).int() * self.equation_ub_weights

        positive_lb_weights = (self.equation_lb_weights >= 0).int() * self.equation_lb_weights
        negative_lb_weights = (self.equation_lb_weights < 0).int() * self.equation_lb_weights

        self.lb = positive_lb_weights @ lb + negative_lb_weights @ ub + self.equation_lb_bias
        self.ub = positive_ub_weights @ ub + negative_ub_weights @ lb + self.equation_ub_bias

    def backward(self,
                 ub_weights: torch.Tensor,
                 lb_weights: torch.Tensor,
                 ub_bias: torch.Tensor,
                 lb_bias: torch.Tensor,
                 backsub_depth: int) -> Tuple[
        AbstractTransformer, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # backward function looks similar to LinearTransformer - possibly refactor somehow to superclass

        positive_ub_weights = (ub_weights >= 0).int() * ub_weights
        negative_ub_weights = (ub_weights < 0).int() * ub_weights

        positive_lb_weights = (lb_weights >= 0).int() * lb_weights
        negative_lb_weights = (lb_weights < 0).int() * lb_weights

        new_ub_bias = ub_bias + positive_ub_weights @ self.ub_bias + negative_ub_weights @ self.lb_bias
        new_lb_bias = lb_bias + positive_lb_weights @ self.lb_bias + negative_lb_weights @ self.ub_bias

        new_ub_weights = positive_ub_weights @ self.ub_weights + negative_ub_weights @ self.lb_weights
        new_lb_weights = positive_lb_weights @ self.lb_weights + negative_lb_weights @ self.ub_weights

        if backsub_depth >= self.previous_transformer.depth:
            return self.previous_transformer, new_ub_weights, new_lb_weights, new_ub_bias, new_lb_bias
        else:
            return self.previous_transformer.backward(new_ub_weights, new_lb_weights, new_ub_bias, new_lb_bias,
                                                      backsub_depth)
        
    def clamp_alphas(self):
        self.alphas.data.clamp_(min=0, max=1)

    def reinitialize_alphas(self):
        nn.init.uniform_(self.alphas, 0, 1)


class LeakyReLUTransformer(AbstractTransformer):

    def __init__(self, module: nn.LeakyReLU, previous_transformer: AbstractTransformer, depth: int):
        super().__init__()
        self.negative_slope = 1.0 * module.negative_slope  # always positive (!), two different cases <1 and >=1
        self.equation_transformer = previous_transformer
        self.previous_transformer = previous_transformer
        self.depth = depth
        self.backsub_depth = depth
        self.alphas = None

    def calculate(self):
        lb = self.previous_transformer.lb
        ub = self.previous_transformer.ub

        crossing = (lb < 0) & (ub > 0)
        negative = (ub <= 0)
        positive = (lb >= 0)

        lmbda = (ub - self.negative_slope * lb) / (ub - lb)  # TODO: think if ub=lb can happen and deal with that
        if self.alphas is None:
            self.alphas = torch.empty(lb.shape, requires_grad=True)
            if self.negative_slope <= 1:
                nn.init.uniform_(self.alphas, self.negative_slope, 1)
            else:
                nn.init.uniform_(self.alphas, 1, self.negative_slope)

        self.ub_weights = negative.int() * torch.full(ub.shape, self.negative_slope) + positive.int() * torch.ones(
            ub.shape) + crossing.int() * lmbda
        self.ub_weights = torch.diag(self.ub_weights)
        self.ub_bias = crossing.int() * (lb * ub * (self.negative_slope - 1) / (ub - lb))

        self.lb_weights = negative.int() * torch.full(ub.shape, self.negative_slope) + positive.int() * torch.ones(
            ub.shape) + crossing.int() * self.alphas
        self.lb_weights = torch.diag(self.lb_weights)
        self.lb_bias = torch.zeros(lb.shape)
        if 1 < self.negative_slope:  # if slope is bigger than 1 then upper and lower bound lines are swapped
            self.ub_weights, self.ub_bias, self.lb_weights, self.lb_bias = self.lb_weights, self.lb_bias, self.ub_weights, self.ub_bias

        # we backsub recursively until we reach the layer we are backsubstituting into
        if self.backsub_depth < self.previous_transformer.depth:
            (self.equation_transformer,
             self.equation_ub_weights,
             self.equation_lb_weights,
             self.equation_ub_bias,
             self.equation_lb_bias) = self.previous_transformer.backward(self.ub_weights,
                                                                         self.lb_weights,
                                                                         self.ub_bias,
                                                                         self.lb_bias,
                                                                         self.backsub_depth)
        else:
            self.equation_transformer = self.previous_transformer
            self.equation_ub_weights = self.ub_weights
            self.equation_lb_weights = self.lb_weights
            self.equation_ub_bias = self.ub_bias
            self.equation_lb_bias = self.lb_bias

        lb = self.equation_transformer.lb
        ub = self.equation_transformer.ub

        # use previous upper bound and lower bound for positive weights, otherwise swap
        positive_ub_weights = (self.equation_ub_weights >= 0).int() * self.equation_ub_weights
        negative_ub_weights = (self.equation_ub_weights < 0).int() * self.equation_ub_weights

        positive_lb_weights = (self.equation_lb_weights >= 0).int() * self.equation_lb_weights
        negative_lb_weights = (self.equation_lb_weights < 0).int() * self.equation_lb_weights

        self.lb = positive_lb_weights @ lb + negative_lb_weights @ ub + self.equation_lb_bias
        self.ub = positive_ub_weights @ ub + negative_ub_weights @ lb + self.equation_ub_bias

    def backward(self,
                 ub_weights: torch.Tensor,
                 lb_weights: torch.Tensor,
                 ub_bias: torch.Tensor,
                 lb_bias: torch.Tensor,
                 backsub_depth: int) -> Tuple[
        AbstractTransformer, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        positive_ub_weights = (ub_weights >= 0).int() * ub_weights
        negative_ub_weights = (ub_weights < 0).int() * ub_weights

        positive_lb_weights = (lb_weights >= 0).int() * lb_weights
        negative_lb_weights = (lb_weights < 0).int() * lb_weights

        new_ub_bias = ub_bias + positive_ub_weights @ self.ub_bias + negative_ub_weights @ self.lb_bias
        new_lb_bias = lb_bias + positive_lb_weights @ self.lb_bias + negative_lb_weights @ self.ub_bias

        new_ub_weights = positive_ub_weights @ self.ub_weights + negative_ub_weights @ self.lb_weights
        new_lb_weights = positive_lb_weights @ self.lb_weights + negative_lb_weights @ self.ub_weights

        if backsub_depth >= self.previous_transformer.depth:
            return self.previous_transformer, new_ub_weights, new_lb_weights, new_ub_bias, new_lb_bias
        else:
            return self.previous_transformer.backward(new_ub_weights, new_lb_weights, new_ub_bias, new_lb_bias,
                                                      backsub_depth)

    def clamp_alphas(self):
        if self.negative_slope <= 1:
            self.alphas.data.clamp_(min=self.negative_slope, max=1)
        else:
            self.alphas.data.clamp_(min=1, max=self.negative_slope)

    def reinitialize_alphas(self):
        if self.negative_slope <= 1:
            nn.init.uniform_(self.alphas, self.negative_slope, 1)
        else:
            nn.init.uniform_(self.alphas, 1, self.negative_slope)



class Conv2dTransformer(AbstractTransformer):

    def __init__(self, module: nn.Linear, previous_transformer: AbstractTransformer, depth: int):

        super().__init__()

        # Convolution attributes
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation

        # Filter weights are tensor of shape (out_channels, in_channels/group, kernel_size[0], kernel_size[1])
        self.conv_filter_weights = module.weights
        assert self.conv_filter_weights.size() == (self.out_channels, self.kernel_size[0], self.kernel_size[1]), \
                                                    f"Wrong conv wonv weiths size: {module.weights.size()}"
        # Biases are tensor of shape (out_channels)
        self.conv_filter_bias = module.bias
        assert module.bias.size() == (self.out_channels), f"Wrong conv biases size: {module.bias.size()}"
    
        self.in_height = int(np.sqrt(self.in_features / self.in_channels))
        self.in_width = int(np.sqrt(self.in_features / self.in_channels))
        self.out_height = int((self.in_height + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0]-1) - 1) / self.stride[0] + 1)
        self.out_width  = int((self.in_width  + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1]-1) - 1) / self.stride[1] + 1)
        
        # Convolution final weights
        self.weights = torch.zeros((self.out_channels, self.in_channels * self.in_height * self.in_width))

        ## TODO: ADAPT THIS to multiple out_channels!!! 
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.in_features = module.in_features
        self.out_features = module.out_features

        self.previous_transformer = previous_transformer
        self.depth = depth
        self.backsub_depth = depth

    def create_weight_matrices(self):
        weight_matrix = []
        bias_tensor = torch.zeros((self.out_channels, self.out_height, self.out_width))
        for out_channel_idx in range(self.out_channels):
            for out_height_idx in range(self.out_height):
                for out_width_idx in range(self.out_width):
                    bias_tensor[out_channel_idx, out_height_idx, out_width_idx] = self.bias[out_channel_idx]
                    # create the row
                    row = torch.zeros((self.in_channels, self.in_height + 2*self.padding[0], self.in_width+2*self.padding[1]))
                    in_height_idx = out_height_idx * self.stride[0]
                    in_width_idx = out_width_idx * self.stride[1]
                    row[:, in_height_idx:in_height_idx + self.kernel_size[0], in_width_idx:in_width_idx+self.kernel_size[1]] = self.weights[out_channel_idx]
                    # remove padding
                    row = row[:, self.padding[0]: self.padding[0]+self.in_height, self.padding[1]:self.padding[1]+self.in_width]
                    # Flatten the row
                    row = torch.reshape(row, (self.in_channels s* self.in_height * self.in_width,))
                    # Store as sparse tensor
                    row = row.to_sparse()
                    weight_matrix.append(row)

        # Final shape of self.weights: (out_channels, in_channels * in_height * in_width)
        assert torch.stack(weight_matrix, dim=0).to_dense().size() == ((self.out_channels, self.in_channels * self.in_height * self.in_width)), \
                            f"Wrong final weights shape: {torch.stack(weight_matrix, dim=0).to_dense().size()}"
        self.weights = torch.stack(weight_matrix, dim=0).to_dense() 

    def calculate(self):

        # self.equation_transformer = self.previous_transformer
        # we backsub recursively until we reach the layer we are backsubstituting into
        if self.backsub_depth < self.previous_transformer.depth:
            # print(f"Backsub called from {self}")
            (self.equation_transformer,
             self.equation_ub_weights,
             self.equation_lb_weights,
             self.equation_ub_bias,
             self.equation_lb_bias) = self.previous_transformer.backward(self.weight,
                                                                         self.weight,
                                                                         self.bias,
                                                                         self.bias,
                                                                         self.backsub_depth)
        else:
            self.equation_transformer = self.previous_transformer
            self.equation_ub_weights = self.weight
            self.equation_lb_weights = self.weight
            self.equation_ub_bias = self.bias
            self.equation_lb_bias = self.bias

        lb = self.equation_transformer.lb
        ub = self.equation_transformer.ub

        # use previous upper bound and lower bound for positive weights, otherwise swap
        positive_ub_weights = (self.equation_ub_weights >= 0).int() * self.equation_ub_weights
        negative_ub_weights = (self.equation_ub_weights < 0).int() * self.equation_ub_weights

        positive_lb_weights = (self.equation_lb_weights >= 0).int() * self.equation_lb_weights
        negative_lb_weights = (self.equation_lb_weights < 0).int() * self.equation_lb_weights

        self.lb = positive_lb_weights @ lb + negative_lb_weights @ ub + self.equation_lb_bias
        self.ub = positive_ub_weights @ ub + negative_ub_weights @ lb + self.equation_ub_bias

    def backward(self,
                 ub_weights: torch.Tensor,
                 lb_weights: torch.Tensor,
                 ub_bias: torch.Tensor,
                 lb_bias: torch.Tensor,
                 backsub_depth: int) -> Tuple[
        AbstractTransformer, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        new_ub_bias = ub_bias + ub_weights @ self.bias
        new_lb_bias = lb_bias + lb_weights @ self.bias

        new_ub_weights = ub_weights @ self.weight
        new_lb_weights = lb_weights @ self.weight

        if backsub_depth >= self.previous_transformer.depth:
            return self.previous_transformer, new_ub_weights, new_lb_weights, new_ub_bias, new_lb_bias
        else:
            return self.previous_transformer.backward(new_ub_weights, new_lb_weights, new_ub_bias, new_lb_bias,
                                                      backsub_depth)
