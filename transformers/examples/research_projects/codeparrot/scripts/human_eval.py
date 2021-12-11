import json
import multiprocessing
import os
import re

from datasets import load_dataset, load_metric
from tqdm import tqdm

import transformers
from arguments import HumanEvalArguments
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline, set_seed


def first_block(string):
    """Split off first block of code by scanning for class, def etc. on newlines."""
    return re.split("\nclass|\ndef|\n#|\n@|\nprint|\nif", string)[0].rstrip()


def complete_code(pipe, prompt, num_completions=1, **gen_kwargs):
    """Complete prompt with text generation pipeline and return num_completions."""
    prompt = pipe.tokenizer.eos_token + prompt
    code_gens = pipe(prompt, num_return_sequences=num_completions, **gen_kwargs)
    return [first_block(code_gen["generated_text"][len(prompt) :]) for code_gen in code_gens]


def main():
    # Setup configuration
    parser = HfArgumentParser(HumanEvalArguments)
    args = parser.parse_args()

    transformers.logging.set_verbosity_error()
    # enables code execution in code_eval metric
    os.environ["HF_ALLOW_CODE_EVAL"] = args.HF_ALLOW_CODE_EVAL
    # make sure tokenizer plays nice with multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.num_workers is None:
        args.num_workers = multiprocessing.cpu_count()

    set_seed(args.seed)

    # Generation settings
    gen_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.model_ckpt)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

    # Load evaluation dataset and metric
    human_eval = load_dataset("openai_humaneval")
    code_eval_metric = load_metric("code_eval")

    # Generate completions for evaluation set
    n_tasks = 4  # len(human_eval["test"])
    generations, references = [], []
    for task in tqdm(range(n_tasks)):
        task_generations = []
        prompt = human_eval["test"][task]["prompt"].strip()
        for batch in range(args.n_samples // args.batch_size):
            task_generations.extend(complete_code(pipe, prompt, num_completions=args.batch_size, **gen_kwargs))
        generations.append([prompt + gen for gen in task_generations])
        test_func = human_eval["test"][task]["test"]
        entry_point = f"check({human_eval['test'][task]['entry_point']})"
        references.append("\n" + test_func + "\n" + entry_point)

    # Evaluate completions with "code_eval" metric
    pass_at_k, _ = code_eval_metric.compute(
        references=references, predictions=generations, num_workers=args.num_workers
    )
    print(f"Results: {pass_at_k}")

    # Save results to json file
    with open(args.output_file, "w") as fp:
        json.dump(pass_at_k, fp)


# For some reason the folliwng seems to be necessary sometimes for code_eval to work nice with multiprocessing
# https://stackoverflow.com/questions/60804599/python-multiprocessing-keeps-spawning-the-whole-script
if __name__ == "__main__":
    main()
