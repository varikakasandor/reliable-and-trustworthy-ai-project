import argparse
import torch
from torch import nn

from networks import get_network
from utils.loading import parse_spec
from abstract_transformers import *

DEVICE = "cpu"


def analyze(net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int) -> bool:
    curr_lower_bounds = inputs.flatten() - eps
    curr_upper_bounds = inputs.flatten() + eps
    for name, layer in net.named_children():
        if isinstance(layer, nn.Flatten):  # No need to do anything in this case
            pass
        elif isinstance(layer, nn.Linear):
            weights = layer.weight
            next_lower_bounds, next_upper_bounds = torch.zeros(weights.shape[0]), torch.zeros(weights.shape[0])
            for i in range(weights.shape[0]):  # TODO: vectorise
                for j in range(weights.shape[1]):
                    if weights[i, j] >= 0:
                        next_lower_bounds[i] += weights[i, j] * curr_lower_bounds[j]
                        next_upper_bounds[i] += weights[i, j] * curr_upper_bounds[j]
                    else:
                        next_lower_bounds[i] += weights[i, j] * curr_upper_bounds[j]
                        next_upper_bounds[i] += weights[i, j] * curr_lower_bounds[j]
            curr_lower_bounds, curr_upper_bounds = next_lower_bounds, next_upper_bounds
        else:
            print(f"Layers of type {type(layer).__name__} are not yet supported")
    highest_incorrect_prediction = None
    for i in range(len(curr_upper_bounds)):
        if i != true_label and (
                highest_incorrect_prediction is None or curr_upper_bounds[i] > highest_incorrect_prediction):
            highest_incorrect_prediction = curr_upper_bounds[i]
    print(f"The category lower bounds are {curr_lower_bounds}")
    print(f"The category upper bounds are {curr_upper_bounds}")
    print(f"The correct category is {true_label}")
    return curr_lower_bounds[true_label] > highest_incorrect_prediction


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

    net = get_network(args.net, dataset, f"../models/{dataset}_{args.net}.pt").to(DEVICE)  # TODO: remove the ../

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
