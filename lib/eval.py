# Import necessary modules
import time
import torch
import torch.nn as nn

# Import get_loaders function from data module within the same directory
from .data import get_loaders 

from collections import defaultdict
import fnmatch
from lib.utils import remove_outlier
import numpy as np


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0"), single_gpu = False, num_partition=8):
    # Set dataset
    dataset = args.eval_dataset

    # Print status
    print(f"evaluating on {dataset}")

    # Get the test loader
    _, testloader = get_loaders(
        dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer 
    )

    # Evaluate ppl in no grad context to avoid updating the model
    with torch.no_grad():
        if single_gpu:
            ppl_test = eval_ppl_single_gpu_wikitext(model, testloader, args.eval_batch_size, device)
        else:
            ppl_test = eval_ppl_wikitext(model, testloader, args.eval_batch_size, device)
    return ppl_test 

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext_train(model, trainloader, bs=1, device=None):
    # Get input IDs
    # testenc = testenc.input_ids

    # Calculate number of samples
    # nsamples = testenc.numel() // model.seqlen
    nsamples = len(trainloader)

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        # inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = trainloader[i][0].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_wikitext(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []

    # Loop through each batch
    for i in range(0,nsamples,bs):
        if i % 50 == 0 and i > 0:
            print(f"sample {i}, Perplexity {torch.exp(torch.stack(nlls).sum() / (i * model.seqlen))}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
        inputs = inputs.reshape(j-i, model.seqlen)

        # Forward pass through the model
        # if inputs.shape[0] != bs:
        #     continue
        start.record()
        lm_logits = model(inputs).logits
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * model.seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    times = remove_outlier(times)
    print(f"Inference Time: ", np.mean(times), "+-", np.std(times))

    # # Empty CUDA cache to save memory
    # torch.cuda.empty_cache()

    return ppl.item()


def eval_ppl_single_gpu_wikitext(model, testenc, bs=1, dev=torch.device(0), num_partitions=8):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    batch_size = nsamples // num_partitions
    outs = torch.zeros_like(inps[:batch_size, :, :])
    print("inps shape: " + str(inps.shape))
    print("outs shape: " + str(outs.shape))

    for batch in range(num_partitions + 1):
        if batch == num_partitions:
            old_batch_size = batch_size
            batch_size = nsamples - (num_partitions * batch_size)
        attention_mask = cache['attention_mask']

        for i in range(len(layers)):
            layer = layers[i].to(dev)

            # if args.gmp:
            #   subset = find_layers(layer)
            #  for name in subset:
            #     W = subset[name].weight.data
            #    thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
            #   W.data[torch.abs(W.data) <= thresh] = 0

            for j in range(batch_size):
                 #    if j % 50 == 0 and j > 0:
                #     print(f"sample {j}, Perplexity {torch.exp(torch.stack(nlls).sum() / (j * model.seqlen))}")
                if batch == 8:
                    outs[j] = layer(inps[(batch * old_batch_size)+j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[(batch * batch_size)+j].unsqueeze(0), attention_mask=attention_mask)[0]

            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            if batch == num_partitions:
                # print("begin "+ str(batch* old_batch_size))
                temp = inps[(batch * old_batch_size):, :, :].clone()
                inps[(batch * old_batch_size):, :, :] = outs[:batch_size, :, :].clone()
                outs[:batch_size, :, :] = temp
                # inps[(batch* old_batch_size):,:,:], outs[:batch_size,:,:] = outs[:batch_size,:,:], inps[(batch* old_batch_size):,:,:]
                del temp
                torch.cuda.empty_cache()
            else:
                # print("begin " + str(batch* batch_size))
                # print("end "+ str((batch+1) * batch_size))
                # print("before")
                # print(inps[(batch* batch_size):((batch+1) * batch_size),:,:])
                # print(outs)
                temp = inps[(batch * batch_size):((batch + 1) * batch_size), :, :].clone()
                inps[(batch * batch_size):((batch + 1) * batch_size), :, :] = outs.clone()
                outs = temp
                del temp
                torch.cuda.empty_cache()
                #inps[(batch* batch_size):((batch+1) * batch_size),:,:], outs = outs, inps[(batch* batch_size):((batch+1) * batch_size),:,:]
                #print("after")
                #print(inps[(batch* batch_size):((batch+1) * batch_size),:,:])
                #print(outs)
    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []

    print(f"nsamples {nsamples}")

    for i in range(nsamples):
        if i % 50 == 0 and i > 0:
            print(f"sample {i}, Perplexity {torch.exp(torch.stack(nlls).sum() / (i * model.seqlen))}")
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
                       :, (i * model.seqlen):((i + 1) * model.seqlen)
                       ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    # print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache
    return ppl.item()


def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"],
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    from lm_eval import tasks, evaluator 
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer, 
        add_special_tokens=add_special_tokens
    )

    return results 