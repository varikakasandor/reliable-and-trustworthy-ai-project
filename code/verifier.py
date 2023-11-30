import argparse
from pathlib import Path

from abstract_transformers import *
from deeppoly import DeepPoly
from networks import get_network
from utils.general import *
from utils.loading import parse_spec
from attacker import attack_non_console_main


def analyze(net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int, n_epochs: int,
            print_debug: bool) -> bool:
    deeppoly = DeepPoly(net, inputs, eps, true_label, print_debug=print_debug)
    return deeppoly.run(n_epochs=n_epochs)


def main_body(parser_args, n_epochs=1000, print_debug=True):
    if print_debug:
        print(parser_args)
    true_label, dataset, image, eps = parse_spec(parser_args.spec)

    # print(args.spec)

    net = get_network(parser_args.net, dataset, f"../models/{dataset}_{parser_args.net}.pt").to(
        DEVICE)  # TODO: remove the ../

    image = image.to(DEVICE)
    out = net(image.unsqueeze(0))

    pred_label = out.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, image, eps, true_label, n_epochs=n_epochs, print_debug=print_debug):
        print("Verified")
        return True
    else:
        print("Not verified")
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
    main_body(args)


def non_console_main(net, spec, n_epochs=1000, print_debug=True):
    model_config = ModelConfig(spec=spec, net=net)
    return main_body(model_config, n_epochs=n_epochs, print_debug=print_debug)


def run_all_test_cases(forbidden_networks=("conv",)):
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
                verified = non_console_main(folder_path.name, relative_file_path, n_epochs=1000, print_debug=False)
                assert verified or attacked
                print()


if __name__ == "__main__":
    non_console_main("conv_4", "../test_cases/conv_4/img2_mnist_0.1797.txt", print_debug=True, n_epochs=100)
    # main()
    #run_all_test_cases()
