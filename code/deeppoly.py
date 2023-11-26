from abstract_transformers import *
#from torchviz import make_dot


class DeepPoly:

    def __init__(self, net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int, print_debug=True):

        self.print_debug = print_debug

        self.net = net
        self.inputs = inputs
        self.eps = eps
        self.true_label = true_label

        # we have an abstract transformer for each layer
        self.transformers: List[AbstractTransformer] = []

        for name, layer in net.named_children():
            if self.print_debug:
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
                # "layer" is not used here, as ReLU is non-parameteric

            elif isinstance(layer, nn.LeakyReLU):
                self.transformers.append(LeakyReLUTransformer(layer, self.transformers[-1], len(self.transformers)))

            else:
                print(f"Layers of type {type(layer).__name__} are not yet supported")

        # add a final linear layer to be able to backsubstitue the difference between x_{true_label} - x_i for all i
        # != true_label
        out = self.net(self.inputs.unsqueeze(0))
        num_classes = out.shape[1]

        final_linear = nn.Linear(num_classes, num_classes)
        final_linear.bias = nn.Parameter(torch.zeros(num_classes))
        final_linear.bias.requires_grad = False
        weight_matrix = -1 * torch.eye(num_classes)
        weight_matrix[:, true_label] += 1
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
        if self.print_debug:
            print(f"True label: {self.true_label}")
            print(f"Lower bounds: {self.transformers[-1].lb}")

        verified = bool(torch.all(self.transformers[-1].lb >= 0))
        if self.print_debug:
            print(f"Verified: {verified}\n")
        return verified
    
    def first_run(self, backsub = True) -> bool:

        # try to verify - if we can't, backsub every layer by 1 and try again
        verified = False
        while not verified and not self.backsub_depth < 0:

            try_string = f"Trying to verify with backsub depth: {self.backsub_depth}"
            if self.print_debug:
                print(try_string)
                print("-" * len(try_string) + "\n")

            if self.backsub_depth > 0:
                with torch.autograd.no_grad():
                    for transformer in self.transformers:
                        transformer.backsub_depth = self.backsub_depth
                        transformer.calculate()
            else:
                for transformer in self.transformers:
                    transformer.backsub_depth = self.backsub_depth
                    transformer.calculate()

            verified = self.verify()
            if not backsub:
                break

            self.backsub_depth -= 1
        
        return verified
    
    def optimization_run(self, n_epochs=1000) -> bool:

        verified = False
        # each run should randomly initialize the slopes and optimize them
        layer_types_to_optimise = (ReLUTransformer, LeakyReLUTransformer)
        if not any(isinstance(layers, layer_types_to_optimise) for layers in self.transformers):
            return verified
        
        if self.print_debug:
            print("Trying to optimize (leaky) relu slopes")

        params = []
        for transformer in self.transformers:
            if isinstance(transformer, layer_types_to_optimise):
                transformer.reinitialize_alphas()
                params.append(transformer.alphas)

        self.optimizer = torch.optim.Adam(params, lr = 0.01)

        for epoch in range(n_epochs):
            self.sum_diff = torch.sum(torch.relu(-self.transformers[-1].lb))
            self.sum_diff.backward(retain_graph=True)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # TODO: check if it makes sense to try to verify after each step

            for transformer in self.transformers:
                if isinstance(transformer, layer_types_to_optimise):
                    transformer.clamp_alphas()

                transformer.calculate()

            verified = self.verify()
            if verified:
                break

        return verified

    def run(self, n_epochs=1000, backsub=True) -> bool:

        verified = self.first_run(backsub = backsub)

        if verified:
            return verified
        
        # if straightforwards backsubstitution doesn't work, try to optimize the slopes until we verify
        num_runs = 0
        while True:

            verified = self.optimization_run(n_epochs=n_epochs)
            if verified:
                break
            num_runs += 1

            # TODO: DELETE FOR THE FINAL SUBMISSION
            if num_runs > 0:
                break
        
        return verified

