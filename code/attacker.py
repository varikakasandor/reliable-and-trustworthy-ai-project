import argparse
import torch
from torch import nn

from networks import get_network
from utils.loading import parse_spec
from abstract_transformers import *

DEVICE = "cpu"

def check_region(x: torch.Tensor, x_adv: torch.Tensor, eps: float) -> bool:

    # check if x_adv is within the eps sized box around x
    # Find out where the discrepancy occurs
    return torch.all((x_adv >= x - eps) & (x_adv <= x + eps))

def pgd(model, x_batch, true_label: int, eps: float, eps_step: float):

    loss_fn = nn.CrossEntropyLoss()

    x_adv = x_batch + eps * (2 * torch.rand_like(x_batch) - 1)
    x_adv.clamp_(min=0., max=1.)

    for _ in range(100):
        x_adv.detach_().requires_grad_()

        model.zero_grad()
        out = model(x_adv)

        loss = loss_fn(out, torch.tensor([true_label]))
        loss.backward() 

        step = eps_step * x_adv.grad.sign()
        x_adv = x_batch + (x_adv + step - x_batch).clamp_(min=-eps, max=eps)

        x_adv.clamp_(min=0, max=1)

        assert check_region(x_batch, x_adv, eps)
        # check if the attack is successful
        if out.max(dim=1)[1].item() != true_label:
            return x_adv.detach()

    return x_adv.detach()


def attack(net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int) -> bool:


    for i in range(100):
        x_adv = pgd(net, inputs, true_label, eps, 0.01)
        assert check_region(inputs, x_adv, eps)
        out = net(x_adv.unsqueeze(0))
        if out.max(dim=1)[1].item() != true_label:
            print(torch.abs(x_adv - inputs))
            print(out)
            print(torch.max(torch.abs(inputs - x_adv)))
            return True
    
    return False


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
    net = net.eval()

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if attack(net, image, eps, true_label):
        print("Attack successful")
    else:
        print("Attack failed")


if __name__ == "__main__":
    main()
