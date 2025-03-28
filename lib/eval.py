# Import necessary modules
import time
import torch
import torch.nn as nn
import tqdm
# Import get_loaders function from data module within the same directory
from .data import get_loaders 
from collections import defaultdict
import fnmatch


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(args, model, tokenizer, device=torch.device("cuda:0")):
    # Set dataset
    ppls = {}
    for dataset in args.ppl_datasets:

        # Print status
        print(f"evaluating on {dataset}")

        # Get the test loader
        testloader = get_loaders(
            dataset, seed=0, seqlen=model.seqlen, tokenizer=tokenizer,
            train_test="test"
        )

        # Evaluate ppl in no grad context to avoid updating the model
        with torch.no_grad():
            ppl_test = ppl_eval_single_dataset(model, testloader, 1, device)
        ppls[dataset] = ppl_test
    return ppls


# Function to evaluate perplexity (ppl) specifically on a single dataset
def ppl_eval_single_dataset(model, testenc, bs=1, device=None):
    # Get input IDs
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // model.seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"nsamples {nsamples}")

    # Loop through each batch
    for i in tqdm.tqdm(range(0,nsamples,bs)):
        if i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * model.seqlen):(j * model.seqlen)].to(device)
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


def zero_shot(base_model, model, 
              batch_size = 1,
              tasks:list[str] = ["winogrande", "piqa", "hellaswag", "arc_easy", "arc_challenge"],
              num_fewshot:int = 0):
    from transformers import AutoTokenizer
    from .lm_eval_adaptor import LMEvalAdaptor
    from lm_eval import evaluator
    
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    lm_eval_model = LMEvalAdaptor(
        base_model,
        model, tokenizer, batch_size=batch_size)
    
    results = evaluator.simple_evaluate(
        model = lm_eval_model,
        tasks = tasks,
        batch_size = batch_size,
        no_cache = True,
        num_fewshot = num_fewshot,
    )
    print(evaluator.make_table(results))
    return results["results"]

