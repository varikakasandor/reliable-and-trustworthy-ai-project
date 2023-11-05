import torch
from torch import nn

from typing import List, Tuple, Optional

class AbstractTransformer:

    def __init__():
        pass

    def forward():
        pass


class InputTransformer(AbstractTransformer):

    def __init__(self, inputs: torch.Tensor, eps: float, flatten: bool):

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
    
    def calculate(self):
        pass


class LinearTransformer(AbstractTransformer):

    def __init__(self, module: nn.Linear, previous_transformer: AbstractTransformer, depth: int):

        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.in_features = module.in_features
        self.out_features = module.out_features

        self.lb = torch.zeros(self.out_features)
        self.ub = torch.zeros(self.out_features)

        self.ub_weights = self.weight
        self.lb_weights = self.weight

        self.ub_bias = self.bias
        self.lb_bias = self.bias

        self.previous_transformer = previous_transformer
        self.depth = depth
        self.backsub_depth = depth

    def calculate(self):

        print("Calculating layer: ", self.depth)
        print("Backsub depth: ", self.backsub_depth)
        print("Previous transformer depth: ", self.previous_transformer.depth)

        if self.backsub_depth < self.previous_transformer.depth:
            (self.previous_transformer, 
            self.ub_weights, 
            self.lb_weights, 
            self.ub_bias, 
            self.lb_bias) = self.previous_transformer.backward(self.ub_weights, 
                                                               self.lb_weights, 
                                                               self.ub_bias, 
                                                               self.lb_bias, 
                                                               self.backsub_depth)
        
        lb = self.previous_transformer.lb
        ub = self.previous_transformer.ub

        #print("Previous transformer lb: ", lb)
        #print("Previous transformer ub: ", ub)

        positive_ub_weights = (self.ub_weights>=0).int() * self.ub_weights
        negative_ub_weights = (self.ub_weights<0).int() * self.ub_weights

        positive_lb_weights = (self.lb_weights>=0).int() * self.lb_weights
        negative_lb_weights = (self.lb_weights<0).int() * self.lb_weights

        '''
        print("Lb bias: ", self.lb_bias.shape)
        print("Ub bias: ", self.ub_bias.shape)
        print("Lb weights: ", self.lb_weights.shape)
        print("Ub weights: ", self.ub_weights.shape)
        '''

        self.lb = positive_lb_weights @ lb + negative_lb_weights @ ub + self.lb_bias
        self.ub = positive_ub_weights @ ub + negative_ub_weights @ lb + self.ub_bias

        #print("Lower bound: ", self.lb)
        #print("Upper bound: ", self.ub)

    def backward(self, 
                lb_weights: torch.Tensor, 
                ub_weights: torch.Tensor, 
                lb_bias: torch.Tensor, 
                ub_bias: torch.Tensor, 
                backsub_depth: int) -> Tuple[AbstractTransformer, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:


        new_ub_bias = ub_bias + ub_weights @ self.ub_bias
        new_lb_bias = lb_bias + lb_weights @ self.lb_bias

        new_ub_weights = ub_weights @ self.ub_weights
        new_lb_weights = lb_weights @ self.lb_weights
        
        if backsub_depth >= self.previous_transformer.depth:

            return self.previous_transformer, new_ub_weights, new_lb_weights, new_ub_bias, new_lb_bias

        else:

            return self.previous_transformer.backward(new_ub_weights, new_lb_weights, new_ub_bias, new_lb_bias, backsub_depth)
       

    def forward(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert lb.shape == (self.in_features,)
        assert ub.shape == (self.in_features,)

        positive_weights = (self.weight>=0).int() * self.weight
        negative_weights = (self.weight<0).int() * self.weight

        self.lb = positive_weights @ lb + negative_weights @ ub + self.bias
        self.ub = positive_weights @ ub + negative_weights @ lb + self.bias

        return self.lb, self.ub


class ReLUTransformer(AbstractTransformer):

    def __init__(self, module: nn.ReLU, previous_transformer: Optional[AbstractTransformer] = None, backward_depth: int = 0):

        self.module = module
        self.previous_transformer = previous_transformer
        self.backward_depth = backward_depth

    def forward(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        self.lb = torch.max(lb, torch.zeros(lb.shape))
        self.ub = torch.max(ub, torch.zeros(ub.shape))

        crossing = (lb < 0) & (ub > 0)
        negative = (ub <= 0)
        positive = (lb >= 0) 

        lmbda = ub / (ub - lb)

        # if negative, then upper and lower bound are 0

        self.upper_bound_weights = negative.int() * torch.zeros(ub.shape) + positive.int() * torch.ones(ub.shape) + crossing.int() * lmbda
        self.upper_bound_bias = -crossing.int() * lmbda * lb

        # placeholder value for alpha
        alpha_placeholder = 0.1
        self.alphas = alpha_placeholder * torch.ones(lb.shape)
        assert torch.all(self.alphas >= 0)
        assert torch.all(self.alphas <= 1)

        self.lower_bound_weights = negative.int() * torch.zeros(ub.shape) + positive.int() * torch.ones(ub.shape) + crossing.int() * self.alphas
        self.lower_bound_bias = torch.zeros(lb.shape)

        return self.lower_bound, self.upper_bound

    def backward():
        pass
