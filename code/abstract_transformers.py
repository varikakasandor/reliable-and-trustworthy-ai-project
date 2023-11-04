import torch
from torch import nn

from typing import List, Tuple, Optional

class AbstractTransformer:

    def __init__(self, module: nn.Module, previous_transformer: Optional['AbstractTransformer'] = None):
        pass

    def forward(self, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class LinearTransformer(AbstractTransformer):

    def __init__(self, module: nn.Linear, previous_transformer: Optional[AbstractTransformer] = None):

        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.in_features = module.in_features
        self.out_features = module.out_features

        self.lower_bound = torch.zeros(self.out_features)
        self.upper_bound = torch.zeros(self.out_features)

        self.upper_bound_weights = self.weight
        self.lower_bound_weights = self.weight

        self.upper_bound_bias = self.bias
        self.lower_bound_bias = self.bias

        self.previous_transformer = previous_transformer

    def forward(self, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        assert lower_bounds.shape == (self.in_features,)
        assert upper_bounds.shape == (self.in_features,)

        positive_weights = (self.weight>=0).int() * self.weight
        negative_weights = (self.weight<0).int() * self.weight

        self.lower_bound = positive_weights @ lower_bounds + negative_weights @ upper_bounds + self.bias
        self.upper_bound = positive_weights @ upper_bounds + negative_weights @ lower_bounds + self.bias

        return self.lower_bound, self.upper_bound

class ReLUTransformer(AbstractTransformer):

    def __init__(self, module: nn.ReLU, previous_transformer: Optional[AbstractTransformer] = None):

        self.module = module
        self.previous_transformer = previous_transformer

    def forward(self, lower_bounds: torch.Tensor, upper_bounds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        self.lower_bound = torch.max(lower_bounds, torch.zeros(lower_bounds.shape))
        self.upper_bound = torch.max(upper_bounds, torch.zeros(upper_bounds.shape))

        return self.lower_bound, self.upper_bound
    


