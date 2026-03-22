import torch.nn as nn
import torch.nn.utils.prune as prune

def apply_global_pruning(model, amount):
    parameters_to_prune = []

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    return model

def remove_pruning_reparam(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
    return model

def measure_sparsity(model):
    zero_params = 0
    total_params = 0

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.detach()
            zero_params += (w == 0).sum().item()
            total_params += w.numel()

    return zero_params / total_params if total_params > 0 else 0.0
