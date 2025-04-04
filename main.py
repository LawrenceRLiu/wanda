import argparse
import os 
import numpy as np
import torch
import json 
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, zero_shot

print("pid: ", os.getpid())
print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="balanced"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument("--calibration_dataset", type=str, default="c4", choices=["wikitext2", "c4", "ptb","pajama"])
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, default="unstructured")
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default=None, type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument("--eval_zero_shot_tasks", nargs="+", 
                        default=["winogrande", "rte", "piqa", "arc_easy", "arc_challenge"],
                        help="The zero shot tasks to evaluate on, if None then no zero shot evaluation will be performed.")
    parser.add_argument('--ppl_datasets', type=str, nargs="+",
                        choices=["wikitext2", "c4", "ptb"],
                        help="The datasets to evaluate on.",
                        default=["wikitext2","c4"])
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    # if "30b" in args.model or "70b" in args.model.lower(): # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
    #     device = model.hf_device_map["lm_head"]
    print("use device ", device)

    with torch.no_grad():
        if args.sparsity_ratio > 0:
            # Handling n:m sparsity
            prune_n, prune_m = 0, 0
            if args.sparsity_type != "unstructured":
                assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
                prune_n, prune_m = map(int, args.sparsity_type.split(":"))

            model_name = args.model.split("/")[-1]
            print(f"loading llm model {args.model}")

            if args.sparsity_ratio != 0:
                print("pruning starts")
                if args.prune_method == "wanda":
                    prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
                elif args.prune_method == "magnitude":
                    prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
                elif args.prune_method == "sparsegpt":
                    prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
                elif "ablate" in args.prune_method:
                    prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

            ################################################################
            print("*"*30)
            sparsity_ratio = check_sparsity(model)
            print(f"sparsity sanity check {sparsity_ratio:.4f}")
            print("*"*30)

            #save first then eval
            if args.save_model:
                model.save_pretrained(args.save_model)
                tokenizer.save_pretrained(args.save_model)
            ################################################################
    
        #evaluation script 

        evals = {}

        ppl_test = eval_ppl(args, model, tokenizer, device)
        evals["ppl"] = ppl_test
        for dataset_name in ppl_test:
            print("dataset: ", dataset_name, "ppl: ", ppl_test[dataset_name])

    #save as a yaml
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        with open(os.path.join(args.save, "evals.yaml")
                  , "w") as f:
            yaml.dump(evals, f)

    #zero shot eval
    if "None" not in args.eval_zero_shot_tasks:

        num_shot = 0
        results = zero_shot(args.model,
                            model,
                            batch_size=1,
                            tasks = args.eval_zero_shot_tasks,
                            num_fewshot=num_shot)
                            
        
        #print the results
        print("results to add to a table:")
        avg_acc = 0

        for task in args.eval_zero_shot_tasks:
            print(round(results[task]["acc"] * 100, 2), end=" & ")
            avg_acc += results[task]["acc"]
        print()
        print("avg acc:", round(avg_acc / len(args.eval_zero_shot_tasks,) * 100, 2))

        #save the results
        if args.save:
            evals["zero_shot"] = results
            
            #overwrite the yaml file
            
            with open(
                os.path.join(args.save, "evals.yaml")
                , "w") as f:
                yaml.dump(evals, f)
            
    




if __name__ == '__main__':
    main()