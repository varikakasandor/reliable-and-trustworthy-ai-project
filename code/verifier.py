import argparse
import torch
from torch import nn

from networks import get_network
from utils.loading import parse_spec
from abstract_transformers import *
from deeppoly import DeepPoly

DEVICE = "cpu"


def analyze(net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int) -> bool:

    # print layers in nn.sequential
    '''
    for name, layer in net.named_children():
        print(name, layer)

    return True


    '''
    print(eps)

    deeppoly = DeepPoly(net, inputs, eps, true_label)
    return deeppoly.run_deeppoly()

    '''
    lb = inputs - eps
    ub = inputs + eps
    lb = torch.clamp(lb, min=0, max=1)
    ub = torch.clamp(ub, min=0, max=1)

    transformers = []

    for name, layer in net.named_children():
        previous_transformer = transformers[-1] if len(transformers) > 0 else None
        if isinstance(layer, nn.Flatten):
            if len(transformers) == 0:
                lb = lb.flatten()
                ub = ub.flatten()
                continue
            else:
                raise Exception("Flatten layer can only be the first layer in the network")

        elif isinstance(layer, nn.Linear):
            print(layer.weight.shape)
            print(layer.bias.shape)
            transformers.append(LinearTransformer(layer, previous_transformer))

        elif isinstance(layer, nn.ReLU):
            transformers.append(ReLUTransformer(layer, previous_transformer))

        else:
            print(f"Layers of type {type(layer).__name__} are not yet supported")


    for transformer in transformers:
        lb, ub = transformer.forward(lb, ub)
    
    print("True label:", true_label)
    print("Lower bounds:", lb)
    print("Upper bounds:", ub)

    ub[true_label] = float('-inf')
    largest_upper_bound = torch.max(ub)
    return lb[true_label] > largest_upper_bound
    '''
    
def main():
    parser = argparse.ArgumentParser(
        description="Neural network verification using DeepPoly relaxation."
    )
    parser.add_argument(
        "--net",
        type=str,
        choices=[
            "fc_base",
            "fc_1",
            "fc_2",
            "fc_3",
            "fc_4",
            "fc_5",
            "fc_6",
            "fc_7",
            "conv_base",
            "conv_1",
            "conv_2",
            "conv_3",
            "conv_4",
        ],
        required=True,
        help="Neural network architecture which is supposed to be verified.",
    )
    parser.add_argument("--spec", type=str, required=True, help="Test case to verify.")
    args = parser.parse_args()

    true_label, dataset, image, eps = parse_spec(args.spec)

    # print(args.spec)

    net = get_network(args.net, dataset, f"models/{dataset}_{args.net}.pt").to(DEVICE)  # TODO: remove the ../

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label):
        print("verified")
    else:
        print("not verified")


if __name__ == "__main__":
    main()
