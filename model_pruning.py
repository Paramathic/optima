from .n_m_sparsity import *
from types import MethodType
from src.utils import is_main_process


def dense_forward(module, input):
    output = Matmul.apply(input, module.weight)
    if not module.bias is None:
        output += module.bias
    return output

def static_prune_inputs_forward(module, input):
    output, module.mask = StaticPruneInputsMatmul.apply(input, module.weight, module.mask)
    if not module.bias is None:
        output += module.bias
    if module.add_lora:
        output += torch.matmul(torch.matmul(input[0], module.lora_left), module.lora_right) / module.lora_rank
    return output

def dynamic_prune_inputs_forward(module, input):
    output = DynamicPruneInputsMatmul.apply(input, module.weight)
    if not module.bias is None:
        output += module.bias
    if module.add_lora:
        output += torch.matmul(torch.matmul(input[0], module.lora_left), module.lora_right) / module.lora_rank
    return output


def static_prune_inputs_reduction_dim_forward(module, input):
    output, module.mask = ReductionDimStaticPruneInputsMatmul.apply(input, module.weight, module.mask)
    if not module.bias is None:
        output += module.bias
    if module.add_lora:
        output += torch.matmul(torch.matmul(input[0], module.lora_left), module.lora_right) / module.lora_rank
    return output


def dynamic_prune_inputs_reduction_dim_forward(module, input):
    output = ReductionDimDynamicPruneInputsMatmul.apply(input, module.weight)
    if not module.bias is None:
        output += module.bias
    if module.add_lora:
        output += torch.matmul(torch.matmul(input[0], module.lora_left), module.lora_right) / module.lora_rank
    return output


def dynamic_prune_weight_forward(module, input):
    output = DynamicPruneWeightMatmul.apply(input, module.weight)
    if not module.bias is None:
        output += module.bias
    if module.add_lora:
        output += torch.matmul(torch.matmul(input[0], module.lora_left), module.lora_right) / module.lora_rank
    return output


def dynamic_prune_weight_reduction_dim_forward(module, input):
    output = ReductionDimDynamicPruneWeightMatmul.apply(input, module.weight)
    if not module.bias is None:
        output += module.bias
    if module.add_lora:
        output += torch.matmul(torch.matmul(input[0], module.lora_left), module.lora_right) / module.lora_rank
    return output


def static_prune_weight_forward(module, input):
    output, module.mask = StaticPruneWeightMatmul.apply(input, module.weight, module.mask)
    if not module.bias is None:
        output += module.bias
    if module.add_lora:
        output += torch.matmul(torch.matmul(input[0], module.lora_left), module.lora_right) / module.lora_rank
    return output


def static_prune_weight_reduction_dim_forward(module, input):
    output, module.mask = ReductionDimStaticPruneWeightMatmul.apply(input, module.weight, module.mask, module.n, module.m)
    if not module.bias is None:
        output += module.bias
    if module.add_lora:
        output += torch.matmul(torch.matmul(input[0], module.lora_left), module.lora_right) / module.lora_rank
    return output


def dynamic_prune_output_grad_forward(module, input):
    output = DynamicPruneOutputGradMatmul.apply(input, module.weight)
    if not module.bias is None:
        output += module.bias
    if module.add_lora:
        output += torch.matmul(torch.matmul(input[0], module.lora_left), module.lora_right) / module.lora_rank
    return output


def dynamic_prune_output_grad_reduction_dim_forward(module, input):
    output = ReductionDimDynamicPruneOutputGradMatmul.apply(input, module.weight)
    if not module.bias is None:
        output += module.bias
    if module.add_lora:
        output += torch.matmul(torch.matmul(input[0], module.lora_left), module.lora_right) / module.lora_rank
    return output


def sparse_linear_activation_forward(module, input):
    if not module.bias is None:
        return module.biased_act_fn(module.bias, module.linear(input))
    else:
        return module.act_fn(module.linear(input))


def prune_model(model,
                skip_layers=[],
                pruned_matrix="static-weight",
                reduction_dim=True,
                add_lora=False,
                lora_rank=4):
    if is_main_process():
        print(f"Modifying model to prune {pruned_matrix}")
    known_modules = {"Linear", "LinearActivation"}
    if pruned_matrix in ["dynamic-input", "input-dynamic"]:
        if reduction_dim:
            pruning_layer = {"Linear": dynamic_prune_inputs_reduction_dim_forward,
                               "LinearActivation": sparse_linear_activation_forward}
            pruning_kernel = {"LinearActivation": pruning_layer["Linear"]}
        else:
            pruning_layer = {"Linear": dynamic_prune_inputs_forward,
                               "LinearActivation": sparse_linear_activation_forward}
            pruning_kernel = {"LinearActivation": pruning_layer["Linear"]}
    elif pruned_matrix in ["static-input", "input-static"]:
        if reduction_dim:
            pruning_layer = {"Linear": static_prune_inputs_reduction_dim_forward,
                               "LinearActivation": sparse_linear_activation_forward}
            pruning_kernel = {"LinearActivation": pruning_layer["Linear"]}
        else:
            pruning_layer = {"Linear": static_prune_inputs_forward}
    elif pruned_matrix in ["dynamic-weight", "weight-dynamic"]:
        if reduction_dim:
            pruning_layer = {"Linear": dynamic_prune_weight_reduction_dim_forward,
                               "LinearActivation": sparse_linear_activation_forward}
            pruning_kernel = {"LinearActivation": pruning_layer["Linear"]}
        else:
            pruning_layer = {"Linear": dynamic_prune_weight_forward,
                               "LinearActivation": sparse_linear_activation_forward}
            pruning_kernel = {"LinearActivation": pruning_layer["Linear"]}
    elif pruned_matrix in ["static-weight", "weight-static"]:
        if reduction_dim:
            pruning_layer = {"Linear": static_prune_weight_reduction_dim_forward,
                               "LinearActivation": sparse_linear_activation_forward}
            pruning_kernel = {"LinearActivation": pruning_layer["Linear"]}
        else:
            pruning_layer = {"Linear": static_prune_weight_forward,
                               "LinearActivation": sparse_linear_activation_forward}
            pruning_kernel = {"LinearActivation": pruning_layer["Linear"]}
    elif pruned_matrix in ["dynamic-grad", "grad-dynamic", "dynamic-output-grad", "output-grad-dynamic"]:
        if reduction_dim:
            pruning_layer = {"Linear": dynamic_prune_output_grad_reduction_dim_forward,
                               "LinearActivation": sparse_linear_activation_forward}
            pruning_kernel = {"LinearActivation": pruning_layer["Linear"]}
        else:
            pruning_layer = {"Linear": dynamic_prune_output_grad_forward,
                               "LinearActivation": sparse_linear_activation_forward}
            pruning_kernel = {"LinearActivation": pruning_layer["Linear"]}
    elif pruned_matrix in ["dense"]:
        pruning_layer = {"Linear": dense_forward,
                         "LinearActivation": dense_forward}
        pruning_kernel = {"LinearActivation": pruning_layer["Linear"]}
    else:
        raise NotImplementedError

    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type in known_modules:
            if module in skip_layers:
                if is_main_process():
                    print("Skipping Module: ", module)
                continue
            if module_type == "LinearActivation":
                module.linear = MethodType(pruning_kernel[module_type], module)
            module.forward = MethodType(pruning_layer[module_type], module)
            module.add_lora = add_lora
            if add_lora:
                module.lora_left = torch.nn.Parameter(torch.randn(module.weight.shape[1], lora_rank)).to(module.weight.device)
                module.lora_right = torch.nn.Parameter(torch.zeros(lora_rank, module.weight.shape[0])).to(module.weight.device)
                module.lora_rank = lora_rank
            module.n, module.m = torch.tensor(0), torch.tensor(0)
            if "static" in pruned_matrix:
                setattr(module.weight, "pruned", False)
                module.mask = None



def get_skip_layers(model, args):
    skip_layers = set()
    if args.skip_attention:
        for block in model.bert.encoder.layer:
            skip_layers.update({
                block.attention.self.query,
                block.attention.self.key,
                block.attention.self.value,
                block.attention.output,
            })
    if args.skip_first_block:
        first_block = model.bert.encoder.layer[0]
        skip_layers.update({
            first_block.attention.self.query,
            first_block.attention.self.key,
            first_block.attention.self.value,
            first_block.attention.output,
            first_block.intermediate.dense_act,
            first_block.output.dense
        })
    if args.skip_last_block:
        last_block = model.bert.encoder.layer[-1]
        skip_layers.update({
            last_block.attention.self.query,
            last_block.attention.self.key,
            last_block.attention.self.value,
            last_block.attention.output,
            last_block.intermediate.dense_act,
            last_block.output.dense
        })
    skip_layers.update({
                            model.bert.encoder.layer[0].attention.self.query,
                            model.bert.encoder.layer[0].attention.self.key,
                            model.bert.encoder.layer[0].attention.self.value,
    })
    try:
        skip_layers.update({
                                model.cls.predictions.decoder,
                                model.cls.seq_relationship
        })
    except:
        pass
    return skip_layers


def set_n_m(model, sparsity_increment=[]):
    if sparsity_increment == "" or sparsity_increment == "-1":
        return
    else:
        sparsity_increment = sparsity_increment.split(",")
        sparsity_increment = [int(x) for x in sparsity_increment]
        if sparsity_increment[-1] == -1:
            sparsity_increment = sparsity_increment[:-1]
        else:
            sparsity_increment = sparsity_increment + [len(model.bert.encoder.layer)]
    n, m = torch.tensor(2), torch.tensor(8)
    for i in range(len(sparsity_increment) - 1):
        for layer_number in range(sparsity_increment[i], sparsity_increment[i + 1]):
            if is_main_process():
                print(f"Setting {n}:{m} sparsity for layers {layer_number}")
            layer = model.bert.encoder.layer[layer_number]
            layer.attention.self.query.m = m.clone().detach()
            layer.attention.self.key.m = m.clone().detach()
            layer.attention.self.value.m = m.clone().detach()
            layer.attention.output.m = m.clone().detach()
            layer.intermediate.dense_act.m = m.clone().detach()
            layer.output.dense.m = m.clone().detach()
            layer.attention.self.query.n = n.clone().detach()
            layer.attention.self.key.n = n.clone().detach()
            layer.attention.self.value.n = n.clone().detach()
            layer.attention.output.n = n.clone().detach()
            layer.intermediate.dense_act.n = n.clone().detach()
            layer.output.dense.n = n.clone().detach()
        m *= 2

