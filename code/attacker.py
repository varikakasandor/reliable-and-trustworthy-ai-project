import argparse
import torch
from pathlib import Path

from abstract_transformers import *
from networks import get_network
from utils.loading import parse_spec
from utils.general import *


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


def attack(net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int, n_epochs: int, print_debug: bool) -> bool:
    for i in range(n_epochs):
        x_adv = pgd(net, inputs, true_label, eps, 0.01)
        
        assert check_region(inputs, x_adv, eps)
        #print(x_adv.shape)
        out = net(x_adv)
        if out.max(dim=1)[1].item() != true_label:
            if print_debug:
                print(torch.abs(x_adv - inputs))
                print(out)
                print(torch.max(torch.abs(inputs - x_adv)))
            return True
    return False


def attack_main_body(parser_args, n_epochs=100, print_debug=True):
    if print_debug:
        print(parser_args)
    true_label, dataset, image, eps = parse_spec(parser_args.spec)

    # print(args.spec)

    net = get_network(parser_args.net, dataset, f"../models/{dataset}_{parser_args.net}.pt").to(
        DEVICE)  # TODO: remove the ../

    image = image.to(DEVICE)
    #print(image.shape)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if attack(net, image.unsqueeze(0), eps, true_label, n_epochs=n_epochs, print_debug=print_debug):
        print("Attack successful")
        return True
    else:
        print("Attack failed")
        return False


def attack_main():
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

    attack_main_body(args)


def attack_non_console_main(net, spec, n_epochs=100, print_debug=True):
    model_config = ModelConfig(spec=spec, net=net)
    return attack_main_body(model_config, n_epochs=n_epochs, print_debug=print_debug)

def run_all_attacks(forbidden_networks=("fc",)):
    current_file_path = Path(__file__).resolve()
    parent_directory = current_file_path.parent.parent
    test_cases_folder = parent_directory / 'test_cases'
    for folder_path in test_cases_folder.iterdir():
        if folder_path.is_dir() and all(s not in folder_path.name for s in forbidden_networks):
            for file_path in folder_path.glob("*.txt"):
                relative_file_path = f"../test_cases/{folder_path.name}/{file_path.name}"
                print(f"Model: {folder_path.name}")
                print(f"Image: {file_path.name}")
                attacked = attack_non_console_main(folder_path.name, relative_file_path, n_epochs=100, print_debug=False)
                print()


if __name__ == "__main__":
    #attack_non_console_main("conv_1", "../test_cases/conv_1/img4_mnist_0.1241.txt", print_debug=False, n_epochs=1000)
    run_all_attacks()