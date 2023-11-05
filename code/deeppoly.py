import torch
from torch import nn

from typing import List, Tuple, Optional
from abstract_transformers import *

class DeepPoly:

    def __init__(self, net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int):

        self.net = net
        self.inputs = inputs
        self.eps = eps
        self.true_label = true_label

        self.transformers = []

        for name, layer in net.named_children():

            if len(self.transformers) == 0:
                if isinstance(layer, nn.Flatten):
                    self.transformers.append(InputTransformer(inputs, eps, flatten=True))
                    continue
                else:
                    self.transformers.append(InputTransformer(inputs, eps, flatten=False))

            if isinstance(layer, nn.Flatten):
                raise Exception("Flatten layer can only be the first layer in the network")
            
            elif isinstance(layer, nn.Linear):
                self.transformers.append(LinearTransformer(layer, self.transformers[-1], len(self.transformers)))
            
            elif isinstance(layer, nn.ReLU):
                self.transformers.append(ReLUTransformer(layer, self.transformers[-1], len(self.transformers)))

            else:
                print(f"Layers of type {type(layer).__name__} are not yet supported")

        self.num_layers = len(self.transformers)

    def run_deeppoly(self):
        
        verified = False
        backsub_complete = False
        while not verified and not backsub_complete:
            backsub_complete = True
            for transformer in self.transformers:
                transformer.backsub_depth = max(0, transformer.backsub_depth - 1)
                backsub_complete = False if transformer.backsub_depth > 0 else backsub_complete
                transformer.calculate()
            
            verified = self.verify()
        
        return verified

    def verify(self) -> bool:

        final_transformer = self.transformers[-1]
        final_lb = final_transformer.lb
        final_ub = final_transformer.ub

        print("True label:", self.true_label)
        print("Lower bounds:", final_lb)
        print("Upper bounds:", final_ub)

        final_ub[self.true_label] = float('-inf')
        largest_upper_bound = torch.max(final_ub)
        verified = final_lb[self.true_label] > largest_upper_bound
        print("Verified:", verified)
        return bool(verified)

    
