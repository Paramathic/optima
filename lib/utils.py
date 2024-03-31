import torch


def density_ratio(x):
    return (x != 0).sum().float() / x.numel()


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

def optimize_pruned_l(L, R, error_mat):
    sparse_L, mask_L = prune_row_wise(L, 2, 4)
    sparse_L = sparse_L.half()
    criterion = torch.nn.MSELoss()
    optimizer_L = optim.SGD(params=[sparse_L], lr=0.01, momentum=0.9)
    optimizer_R = optim.SGD(params=[R], lr=0.01, momentum=0.9)
    sparse_L.requires_grad = True
    R.requires_grad = True
    convergence_threshold = 5e-8
    e2e_iteration = 0
    max_iterations = 50
    e2e_convergence_threshold = 1e-8
    e2e_prev_loss = float('inf')

    while True:
        prev_loss = float('inf')
        iteration = 0
        while True:
            optimizer_L.zero_grad()
            error_mat_hat = sparse_L @ R
            loss = criterion(error_mat_hat, error_mat.half())
            loss.backward()
            optimizer_L.step()
            sparse_L_detached = sparse_L.detach()
            sparse_L_detached[mask_L] = 0.
            sparse_L = sparse_L_detached.requires_grad_()

            if abs(loss.item() - prev_loss) < convergence_threshold:
                print("L Converged in "+ str(iteration))
                break
            prev_loss = loss.item()
            iteration += 1

            if iteration >= max_iterations:
                print("Maximum iterations reached.")
                break

        iteration = 0
        prev_loss = float('inf')

        while True:
            optimizer_R.zero_grad()
            error_mat_hat = sparse_L @ R
            loss = criterion(error_mat_hat, error_mat.half())
            loss.backward()
            optimizer_R.step()
            sparse_L_detached = sparse_L.detach()
            sparse_L_detached[mask_L] = 0.
            sparse_L = sparse_L_detached.requires_grad_()

            if abs(loss.item() - prev_loss) < convergence_threshold:
                print("R Converged in "+ str(iteration))
                break
            prev_loss = loss.item()
            iteration += 1

            if iteration >= max_iterations:
                print("Maximum iterations reached.")
                break

        if abs(loss.item() - e2e_prev_loss) < e2e_convergence_threshold:
                print("e2e algo Converged in "+ str(e2e_iteration))
                break
        e2e_prev_loss = loss.item()
        e2e_iteration += 1

        if e2e_iteration >= max_iterations:
            print("E2E Maximum iterations reached.")
            break

    sparse_L.requires_grad = False
    R.requires_grad = False

    return sparse_L, R

def add_lora(module, 
             W_mask, 
             rank_ratio=0.01, 
             use_wanda=False, 
             activations=None, 
             use_randomized_svd=True, 
             quantizer=None, 
             bitwidth=8,
             use_std=False,
             max_bitwidth=8,
             pruned_L = False):

    if use_wanda and not any (activations.scaler_row == 0):
        if quantizer is None:
            W_metric = module.weight.data * (torch.sqrt(activations.scaler_row.reshape((1,-1))))
            new_weight = W_metric.clone().detach()
            new_weight[W_mask] = 0
            error_mat = W_metric - new_weight
        else:
            W_metric = module.weight.data * (torch.sqrt(activations.scaler_row.reshape((1,-1))))
            new_weight = module.weight.data
            new_weight[W_mask] = 0
            new_weight = quantizer.quantize_weight(new_weight, bitwidth, use_std=use_std, max_bitwidth=max_bitwidth)
            new_weight = quantizer.dequantize_absmax(new_weight) * (torch.sqrt(activations.scaler_row.reshape((1,-1))))
            error_mat = (W_metric - new_weight)
    else:
        new_weight = module.weight.data.clone().detach()
        new_weight[W_mask] = 0
        if quantizer is not None:
            new_weight = quantizer.quantize_weight(new_weight, bitwidth, use_std=use_std, max_bitwidth=max_bitwidth)
            new_weight = quantizer.dequantize_absmax(new_weight)
        error_mat = module.weight.data - new_weight
    # Use SVD on the error matrix to find the best low-rank approximation
    if use_randomized_svd:
        U, S, V = randomized_svd(error_mat.float(), rank_ratio)
    else:
        U, S, V = torch.svd(error_mat.float())
    rank = int(rank_ratio * min(error_mat.shape))

    if pruned_L:
        L = U[:, :rank].half()
        R = torch.diag_embed(S[:rank]).half() @ V[:, :rank].half().T
        sparse_L, R = optimize_pruned_l(L, R, error_mat)
        low_rank_weight = sparse_L.half() @ R

    else:
        low_rank_weight = U[:, :rank].half() @ torch.diag_embed(S[:rank]).half() @ V[:, :rank].half().T
    if use_wanda and not any (activations.scaler_row == 0):
        low_rank_weight = low_rank_weight / (torch.sqrt(activations.scaler_row.reshape((1,-1)))).half()
    new_weight = module.weight.data - low_rank_weight
    new_weight[W_mask] = 0
    if quantizer is not None:
        new_weight = quantizer.quantize_weight(new_weight, bitwidth, use_std=use_std, max_bitwidth=max_bitwidth)
        new_weight = quantizer.dequantize_absmax(new_weight)
    module.weight.data = (new_weight + low_rank_weight).half()


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