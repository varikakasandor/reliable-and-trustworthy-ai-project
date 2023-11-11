from abstract_transformers import *


class DeepPoly:

    def __init__(self, net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int):

        self.net = net
        self.inputs = inputs
        self.eps = eps
        self.true_label = true_label

        # we have an abstract transformer for each layer
        self.transformers: List[AbstractTransformer] = []

        for name, layer in net.named_children():

            print(name, layer)

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

        print()
        # add a final linear layer to be able to backsubstitue the difference between x_{true_label} - x_i for all i
        # \neq true_label
        out = self.net(self.inputs.unsqueeze(0))
        num_classes = out.shape[1]

        final_linear = nn.Linear(num_classes, num_classes)
        final_linear.bias = nn.Parameter(torch.zeros(num_classes))
        final_linear.bias.requires_grad = False
        weight_matrix = -1 * torch.eye(num_classes)
        weight_matrix[:, true_label] += 1
        # print(weight_matrix)
        final_linear.weight = nn.Parameter(weight_matrix)
        final_linear.weight.requires_grad = False
        final_linear.eval()
        final_transformer = LinearTransformer(final_linear, self.transformers[-1], len(self.transformers))

        self.transformers.append(final_transformer)

        self.num_layers = len(self.transformers)
        # layers are backsubstituted at least until the layer with depth = backsub_depth
        # and with no backsubstitution this is the penultimate layer (-2 due to InputTransformer)
        self.backsub_depth = self.num_layers - 2

    def verify(self) -> bool:

        self.final_transformer = self.transformers[-1]
        self.final_lb = self.final_transformer.lb
        self.final_ub = self.final_transformer.ub

        print(f"True label: {self.true_label}")
        print(f"Lower bounds: {self.final_lb}")

        verified = bool(torch.all(self.final_lb >= 0))
        print(f"Verified: {verified}\n")
        return verified

    def run(self, backsub=True) -> bool:

        # try to verify - if we can't, backsub every layer by 1 and try again
        verified = False
        while not verified and not self.backsub_depth < 0:

            try_string = f"Trying to verify with backsub depth: {self.backsub_depth}"
            print(try_string)
            print("-" * len(try_string) + "\n")

            for transformer in self.transformers:
                # print(f"Calculating: {transformer}")
                transformer.backsub_depth = self.backsub_depth
                transformer.calculate()

            verified = self.verify()
            if not backsub:
                break

            self.backsub_depth -= 1

        self.backsub_depth = 0

        if not verified:
            # then try optimizing slopes of relus using gradient descent
            print("Trying to optimize relu slopes")

            self.optimizer = torch.optim.Adam(
                [transformer.alphas for transformer in self.transformers if isinstance(transformer, ReLUTransformer)],
                lr=0.01)

            n_epochs = 1000

            for epoch in range(n_epochs):
                self.optimizer.zero_grad()
                self.sum_diff = torch.sum(torch.relu(-self.final_lb))
                self.sum_diff.backward(retain_graph=True)
                # print grads
                for transformer in self.transformers:
                    if isinstance(transformer, ReLUTransformer):
                        # print(f"Relu Slopes: {transformer.alphas}")
                        # print(f"Relu Grads: {transformer.alphas.grad}")
                        pass

                self.optimizer.step()

                for transformer in self.transformers:
                    if isinstance(transformer, ReLUTransformer):
                        transformer.alphas.data.clamp_(min=0, max=1)

                for transformer in self.transformers:
                    transformer.calculate()

                verified = self.verify()
                if verified:
                    break

        return verified
