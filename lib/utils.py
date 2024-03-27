import torch


def get_layers_list(model):
    if hasattr(model, "model"):
        layers = model.model.decoder.layers
    elif hasattr(model, "transformer"):
        layers = model.transformer.h
    else:
        raise NotImplementedError
    return layers


def shift_zeros(x):
    min_positive = x.clone().detach()
    min_positive[min_positive == 0] = 1
    min_positive = min_positive.min()
    return x + min_positive

def add_lora(module, W_mask, rank_ratio=0.01, use_wanda=False, activations=None, use_randomized_svd=True, quantizer=None, bitwidth=8):
    if use_wanda and not any (activations.scaler_row == 0):
        if quantizer is None:
            W_metric = module.weight.data * (torch.sqrt(activations.scaler_row.reshape((1,-1))))
            new_weight = W_metric.clone().detach()
            new_weight[W_mask] = 0
            error_mat = W_metric - new_weight
        else:
            W_metric = module.weight.data
            new_weight = W_metric.clone().detach()
            new_weight[W_mask] = 0
            new_weight = quantizer.quantize_weight(new_weight, bitwidth)
            new_weight = quantizer.dequantize_absmax(new_weight)
            error_mat = (W_metric - new_weight)* (torch.sqrt(activations.scaler_row.reshape((1,-1))))
        # Use SVD on the error matrix to find the best low-rank approximation
        if use_randomized_svd:
            U, S, V = randomized_svd(error_mat.float(), rank_ratio)
        else:
            U, S, V = torch.svd(error_mat.float())
        rank = int(rank_ratio * min(error_mat.shape))
        module.weight.data = ((new_weight + U[:, :rank].half() @ torch.diag_embed(S[:rank]).half() @ V[:, :rank].half().T) / (torch.sqrt(activations.scaler_row.reshape((1,-1))))).half()
    else:
        new_weight = module.weight.data.clone().detach()
        new_weight[W_mask] = 0
        if quantizer is not None:
            new_weight = quantizer.quantize_weight(new_weight, bitwidth)
            new_weight = quantizer.dequantize_absmax(new_weight)
        error_mat = module.weight.data - new_weight
        # Use SVD on the error matrix to find the best low-rank approximation
        if use_randomized_svd:
            U, S, V = randomized_svd(error_mat.float(), rank_ratio)
        else:
            U, S, V = torch.svd(error_mat.float())
        rank = int(rank_ratio * min(error_mat.shape))
        module.weight.data = new_weight + U[:, :rank].half() @ torch.diag_embed(S[:rank]).half() @ V[:, :rank].half().T


def randomized_svd(B, rank, redundancy=2):
    if rank < 1:
        output_rank = int(rank * min(B.size()))
        rank = int(min(rank * redundancy, 0.5) * min(B.size()))
    m, n = B.size()
    rand_matrix = torch.randn((n, rank)).to(B.device)  # short side by k
    Q, _ = torch.linalg.qr(B @ rand_matrix)  # long side by k
    smaller_matrix = (Q.transpose(0, 1) @ B)  # k by short side
    U_hat, S, V = torch.svd(smaller_matrix, False)
    U = (Q @ U_hat)
    return U[:, :output_rank], S[0:output_rank], V[:, :output_rank]


if __name__ == "__main__":
    # Unit Test for randomized_svd
    A = torch.randn(1000, 1000)
    rank_ratio = 0.1
    U, S, V = randomized_svd(A, rank_ratio)
    error = A - U @ torch.diag_embed(S) @ V.T
    print(error.norm() / A.norm())
    U, S, V = torch.svd(A)
    rank = int(rank_ratio * min(A.shape))
    error = A - U[:, :rank] @ torch.diag_embed(S[:rank]) @ V[:, :rank].T
    print(error.norm() / A.norm())