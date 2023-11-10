import argparse
import torch
from torch import nn

from deeppoly import DeepPoly
from abstract_transformers import *

DEVICE = "cpu"

def test1():
    # NOT WORKING ANYMORE
    inputs = torch.tensor([0.5, 0.5])
    eps = 0.25

    linear_1 = nn.Linear(2, 2)
    linear_1.weight = nn.Parameter(torch.tensor([[1, 1], [2, -1]], dtype=torch.float))
    linear_1.bias = nn.Parameter(torch.tensor([0, -1], dtype=torch.float))

    net = nn.Sequential(
        linear_1,
        nn.ReLU(),
    )

    net.eval()

    true_label = 1

    deeppoly = DeepPoly(net, inputs, eps, true_label)
    deeppoly.run(backsub = False)

    linear_1_transformer = deeppoly.transformers[1]
    relu_1_transformer = deeppoly.transformers[2]

    assert torch.allclose(linear_1_transformer.lb, torch.tensor([0.5, -1.25]))
    assert torch.allclose(linear_1_transformer.ub, torch.tensor([1.5, 0.25]))
    
    assert torch.allclose(relu_1_transformer.lb, torch.tensor([0.5, 0]))
    assert torch.allclose(relu_1_transformer.ub, torch.tensor([1.5, 0.25]))

    assert torch.allclose(relu_1_transformer.ub_weights, torch.tensor([[1, 0],[0,1/6]], dtype=torch.float))
    print(relu_1_transformer.lb_weights)
    assert torch.allclose(relu_1_transformer.lb_weights, torch.tensor([[1, 0],[0,0.25]], dtype=torch.float))

    assert torch.allclose(relu_1_transformer.ub_bias, torch.tensor([0., 5/24]))
    assert torch.allclose(relu_1_transformer.lb_bias, torch.tensor([0., 0.]))

def test2():
    # NOT WORKING ANYMORE
    inputs = torch.tensor([0.5, 0.5])
    eps = 0.25

    linear_1 = nn.Linear(2, 2)
    linear_1.weight = nn.Parameter(torch.tensor([[1, 1], [2, -1]], dtype=torch.float))
    linear_1.bias = nn.Parameter(torch.tensor([0, -1], dtype=torch.float))

    net = nn.Sequential(
        linear_1,
        nn.ReLU(),
    )

    net.eval()

    # the false label to enable backsubstitution
    true_label = 1

    deeppoly = DeepPoly(net, inputs, eps, true_label)
    deeppoly.run(backsub = True)

    linear_1_transformer = deeppoly.transformers[1]
    relu_1_transformer = deeppoly.transformers[2]

    assert torch.allclose(linear_1_transformer.lb, torch.tensor([0.5, -1.25]))
    assert torch.allclose(linear_1_transformer.ub, torch.tensor([1.5, 0.25]))

    assert torch.allclose(relu_1_transformer.ub_weights, torch.tensor([[1, 1],[1/3,-1/6]], dtype=torch.float))
    assert torch.allclose(relu_1_transformer.lb_weights, torch.tensor([[1, 1],[1/2,-1/4]], dtype=torch.float))

    #print(relu_1_transformer.ub_bias)
    assert torch.allclose(relu_1_transformer.ub_bias, torch.tensor([0., 1/24]))
    assert torch.allclose(relu_1_transformer.lb_bias, torch.tensor([0., -1/4]))

    assert torch.allclose(relu_1_transformer.lb, torch.tensor([0.5, 0]))
    assert torch.allclose(relu_1_transformer.ub, torch.tensor([1.5, 0.25]))

def test3():
    # example from the course slides
    # make sure to comment the clamping part in InputTransformer

    inputs = torch.tensor([0, 0], dtype=torch.float)
    eps = 1

    linear_1 = nn.Linear(2, 2)
    linear_1.weight = nn.Parameter(torch.tensor([[1, 1], [1, -1]], dtype=torch.float))
    linear_1.bias = nn.Parameter(torch.tensor([0, 0], dtype=torch.float))

    linear_2 = nn.Linear(2, 2)
    linear_2.weight = nn.Parameter(torch.tensor([[1, 1], [1, -1]], dtype=torch.float))
    linear_2.bias = nn.Parameter(torch.tensor([-1/2, 0], dtype=torch.float))

    linear_3 = nn.Linear(2, 2)
    linear_3.weight = nn.Parameter(torch.tensor([[-1, 1], [0, 1]], dtype=torch.float))
    linear_3.bias = nn.Parameter(torch.tensor([3, 0], dtype=torch.float))

    net = nn.Sequential(
        linear_1,
        nn.ReLU(),
        linear_2,
        nn.ReLU(),
        linear_3,
    )

    net.eval()

    # the false label to enable backsubstitution
    true_label = 0

    deeppoly = DeepPoly(net, inputs, eps, true_label)
    deeppoly.run(backsub = True)

    linear_1_transformer = deeppoly.transformers[1]
    relu_1_transformer = deeppoly.transformers[2]
    linear_2_transformer = deeppoly.transformers[3]
    relu_2_transformer = deeppoly.transformers[4]
    linear_3_transformer = deeppoly.transformers[5]
    final_transformer = deeppoly.transformers[6]
    
    assert torch.allclose(linear_1_transformer.lb, torch.tensor([-2., -2.]))    
    assert torch.allclose(linear_1_transformer.ub, torch.tensor([2., 2.]))
    assert torch.allclose(relu_1_transformer.lb, torch.tensor([0., 0.]))
    assert torch.allclose(relu_1_transformer.ub, torch.tensor([2., 2.]))
    assert torch.allclose(linear_2_transformer.lb, torch.tensor([-0.5, -2.]))
    assert torch.allclose(linear_2_transformer.ub, torch.tensor([2.5, 2.]))
    assert torch.allclose(relu_2_transformer.lb, torch.tensor([0., 0.]))
    assert torch.allclose(relu_2_transformer.ub, torch.tensor([2.5, 2.]))
    assert torch.allclose(linear_3_transformer.lb, torch.tensor([0.5, 0.]))
    assert torch.allclose(linear_3_transformer.ub, torch.tensor([5., 2.]))
    assert torch.allclose(final_transformer.lb, torch.tensor([0., 0.5]))
    assert torch.allclose(final_transformer.ub, torch.tensor([0., 3.]))   


def main(): 
    # TEST 1 and 2 DOESNT WORK ANYMORE!!

    #test1()
    #test2()
    test3()




if __name__ == "__main__":
    main()
