
import numpy as np
import gc
import jax
import jax.numpy as jnp
from mpax import create_qp, raPDHG
import numpy as np
from mpax.utils import TerminationStatus
import torch
from .mask_optim import block_wise_optimize_mask
import tqdm
from jax.dlpack import from_dlpack, to_dlpack


def torch_to_jax(tensor: torch.Tensor) -> jnp.ndarray:
    """
    Convert a PyTorch tensor to a JAX tensor, inferring the dtype (bf16, float32, or float64).

    Args:
        tensor (torch.Tensor): Input PyTorch tensor.

    Returns:
        jnp.ndarray: JAX tensor with the same dtype as the input (bf16, float32, or float64).

    Raises:
        ValueError: If the input tensor's dtype is not supported.
    """
    if tensor.dtype == torch.bfloat16:
        # JAX does not natively support direct conversion of torch.bfloat16 tensors.
        # We perform a conversion via DLPack and then explicitly cast the JAX array
        # to jnp.bfloat16 to preserve the data type.
        # Note: This might involve a data copy.
        temp_jax_array = jnp.array(tensor.to(torch.float32))
        return temp_jax_array.astype(jnp.bfloat16)
    else:
        # For other supported dtypes (e.g., fp32, fp64), direct conversion works
        # and automatically respects the device.
        return jnp.array(tensor)


def update_weights_qp(layer, W_mask, double_precision=False, half_precision=False):
    assert not (half_precision and double_precision), "Cannot use both half and double precision"
    with torch.no_grad():
        device = "cpu"
        input_cov = torch.zeros((layer.inputs[0].shape[-1], layer.inputs[0].shape[-1]), device="cuda")
        for x in layer.inputs:
            x = x.view(-1, x.shape[-1]).cuda()
            input_cov += torch.matmul(x.t(), x) / len(layer.inputs) / x.shape[0]
            x = x.cpu()
        def single_optimize(c_vector, G_matrix, h_vector, l_vector, u_vector, Q_matrix, A_matrix, b_vector, eps_abs=1e-2):
            """Optimize a single QP problem."""
            # print("Q:", Q_matrix)
            # print("c:", c_vector)
            # print("A:", A_matrix)
            # print("b:", b_vector)
            # print("G:", G_matrix)
            # print("h:", h_vector)
            # print("l:", l_vector)
            # print("u:", u_vector)
            # exit()
            qp = create_qp(Q_matrix, c_vector, A_matrix, b_vector, G_matrix, h_vector, l_vector, u_vector, use_sparse_matrix=False)
            solver = raPDHG(eps_abs=eps_abs, eps_rel=1e-2, verbose=False, iteration_limit=100_000)  # Set verbose=False for batch processing
            result = solver.optimize(qp)

            # Calculate objective value: 1/2 x' Q x + c' x
            obj = 0.5 * jnp.dot(result.primal_solution, jnp.dot(Q_matrix, result.primal_solution)) + jnp.dot(c_vector, result.primal_solution)
            return result.primal_solution, obj, result.termination_status
        if half_precision:
            dtype = torch.float16
        elif double_precision:
            dtype = torch.float64
        else:
            dtype = torch.float32
        jax.config.update("jax_enable_x64", double_precision)
        weight = layer.original_weight.clone().to(dtype).to(device)
        # For each row of weight (w), minimize
        # w @ input_cov @ w + 0
        # s.t. w[w_mask] = 0
        Q_torch = input_cov.to(device).to(dtype)
        Q = torch_to_jax(Q_torch)  # Shared Q matrix
        n_params = Q.shape[0]
        
        c_torch = torch.zeros(n_params, device=device, dtype=dtype)
        c = torch_to_jax(c_torch)

        A_torch = torch.zeros(1, n_params, device=device, dtype=dtype)
        b_torch = torch.zeros(1, device=device, dtype=dtype)
        A = torch_to_jax(A_torch)  # Zero A matrix
        b = torch_to_jax(b_torch)  # Zero b vector

        G_torch_mini = torch.ones((1, n_params), device=device, dtype=dtype)
        h_torch_mini = -torch.ones((1), device=device, dtype=dtype) * 1e5
        G = torch_to_jax(G_torch_mini)
        h = torch_to_jax(h_torch_mini)
                

        batch_optimize = jax.vmap(single_optimize, in_axes=(None, None, None, 0, 0, None, None, None, None))

        # Process in mini-batches
        all_solutions = []
        all_objectives = []

        batch_size = weight.shape[0]
        eps_abs = 1e-2
        mini_batch_size = min(batch_size, max(1, 128 * 12 * 1024 * 1024 // (weight.shape[1] ** 2)))
        tuning_progress_bar = tqdm.tqdm(range(0, batch_size, mini_batch_size), desc="Tuning Progress")
        skip_layer = False

        for start_idx in tuning_progress_bar:
            end_idx = min(start_idx + mini_batch_size, batch_size)
            current_mini_batch_size = end_idx - start_idx
            
            l_torch_mini = torch.full((current_mini_batch_size, n_params), -float('inf'), device=device, dtype=dtype)
            u_torch_mini = torch.full((current_mini_batch_size, n_params), float('inf'), device=device, dtype=dtype)
            
            # For each sample in mini-batch, randomly select variables to have equality constraints
            for row in range(current_mini_batch_size):
                eq_indices = W_mask[start_idx + row, :].to(device)
                l_torch_mini[row, eq_indices] = -weight[start_idx+row, :][eq_indices].clone()
                u_torch_mini[row, eq_indices] = -weight[start_idx+row, :][eq_indices].clone() + 1e-4

            # Convert mini-batch tensors to JAX arrays
            l_mini = torch_to_jax(l_torch_mini)
            u_mini = torch_to_jax(u_torch_mini)

            torch.cuda.empty_cache()  # Clear GPU memory if using CUDA

            # Solve mini-batch of QP problems
            tuning_progress_bar.set_description(f"Solving mini-batch {start_idx // mini_batch_size + 1}/{(batch_size + mini_batch_size - 1) // mini_batch_size}")
            tuning_progress_bar.set_postfix({"eps_abs": eps_abs})
            solutions_mini, objectives_mini, termination_status_mini = batch_optimize(c, G, h, l_mini, u_mini, Q, A, b, eps_abs)
            converged_vals = [status == TerminationStatus.OPTIMAL for status in termination_status_mini]
            if sum(converged_vals) / len(converged_vals) < 0.95:
                print("QP solver did not converge")
                print("Termination status:", termination_status_mini)
                print("Skipping Layer")
                skip_layer = True
                break

            solutions_mini = np.asarray(solutions_mini)
            solutions_mini = torch.from_numpy(solutions_mini).to(dtype=torch.bfloat16).to(device)
            # print("R", solutions_mini)
            # print("W", weight[start_idx, :])
            # print("W + R", weight[start_idx, :] + solutions_mini)
            all_solutions.append(solutions_mini)
            all_objectives.append(objectives_mini)
        if not skip_layer:
            all_solutions = torch.cat(all_solutions, dim=0)
            best_weight = weight.to(torch.bfloat16) + all_solutions
            best_weight[W_mask] = 0  ## set weights to zero
        else:
            best_weight = weight.to(torch.bfloat16)
            best_weight[W_mask] = 0  ## set weights to zero
        best_weight = best_weight.cuda()

        #Delete all the variables
        del G_torch_mini, h_torch_mini, l_torch_mini, u_torch_mini
        del G, h, l_mini, u_mini
        del c_torch, c, A_torch, b_torch, A, b
        del Q_torch, Q
        del weight, solutions_mini, all_solutions, all_objectives
        torch.cuda.empty_cache()
        gc.collect()
        return best_weight


def update_weights_learnable(layer, trainable_weight, W_mask, num_steps=20):
    """
    Update the weights of the layer using learnable parameters.
    """
    best_loss = torch.inf
    best_weight = None
    with torch.set_grad_enabled(True):
        for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
            weight = torch.nn.Parameter(trainable_weight.data.clone().cuda())
            optimizer = torch.optim.Adam([weight], lr=lr)
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_steps)
            tuning_progress_bar = tqdm.tqdm(range(num_steps), desc="Tuning Progress")
            init_loss = None
            for step in tuning_progress_bar:
                avg_loss = 0
                for (x, y) in zip(layer.inputs, layer.outputs):
                    output = x.cuda() @ weight.t()
                    loss = torch.nn.functional.mse_loss(output, y.cuda())
                    loss.backward()
                    avg_loss += loss.item()
                avg_loss /= len(layer.inputs)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                weight.data[W_mask] = 0  ## set weights to zero
                if init_loss is None:
                    init_loss = avg_loss
                tuning_progress_bar.set_postfix({"init_loss": init_loss, "loss": avg_loss, "lr": lr})
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_weight = weight.data.clone().detach()
    return best_weight


def optimize_weights(layer, compressed_layer, use_qp_solver, double_precision, update_mask, W_mask):
    def compute_error(weight):
        with torch.no_grad():
            errors = []
            for (x, y) in zip(layer.inputs, layer.outputs):
                y_hat = torch.matmul(x.cuda(), weight.t())
                errors.append(torch.nn.functional.mse_loss(y_hat, y.cuda()).item())
        return np.mean(errors)
    init_loss = compute_error(compressed_layer.weight.cuda())
    

    if use_qp_solver:
        best_weight = update_weights_qp(
            layer, 
            W_mask,
            double_precision,
        )
    else:
        if update_mask:
            tunable_layer = torch.nn.Linear(layer.original_weight.data.shape[1], layer.original_weight.data.shape[0], bias=False)
            tunable_layer.weight.data = layer.original_weight.data.clone().detach().cuda()
            tunable_layer.init_mask = (compressed_layer.weight.data != 0).to(torch.bfloat16).cuda()

            init_loss_, final_loss_ = block_wise_optimize_mask(
                tunable_layer,
                {},
                layer.inputs,
                layer.outputs,
                num_epochs=4,
                optimizer="adam",
                verbose=False,
            )

            best_weight = tunable_layer.weight.data.clone().detach()
            trainable_weight = best_weight

            print("Average Mask Similarity: ", ((best_weight == 0) == W_mask).float().mean())
        else:
            trainable_weight = compressed_layer.weight

        best_weight = update_weights_learnable(layer, trainable_weight, W_mask=W_mask)

    final_loss = compute_error(best_weight)
    norm = compute_error(torch.zeros_like(best_weight))
    print(f"Init Loss: {init_loss / norm}, Final Loss: {final_loss / norm}")
    if final_loss < init_loss:
        compressed_layer.weight.data = best_weight.to(compressed_layer.weight.dtype)
    else:
        print("No improvement. Skipping update.")

    layer.inputs = []
    layer.outputs = []
    del layer.original_weight
    gc.collect()