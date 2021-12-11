from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    """
    Configuration for training model.
    """

    model_ckpt: Optional[str] = field(
        default="lvwerra/codeparrot",
        metadata={"help": "Model name or path of model to be trained."},
    )
    save_dir: Optional[str] = field(
        default="./",
        metadata={"help": "Save dir where model repo is cloned and models updates are saved to."},
    )
    dataset_name_train: Optional[str] = field(
        default="lvwerra/codeparrot-clean-train", metadata={"help": "Name or path of training dataset."}
    )
    dataset_name_valid: Optional[str] = field(
        default="lvwerra/codeparrot-clean-valid", metadata={"help": "Name or path of validation dataset."}
    )
    train_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size for training."})
    valid_batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size for evaluation."})
    weight_decay: Optional[float] = field(default=0.1, metadata={"help": "Value of weight decay."})
    shuffle_buffer: Optional[int] = field(
        default=1000, metadata={"help": "Size of buffer used to shuffle streaming dataset."}
    )
    learning_rate: Optional[float] = field(default=2e-4, metadata={"help": "Learning rate fo training."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "Learning rate."})
    num_warmup_steps: Optional[int] = field(
        default=750, metadata={"help": "Number of warmup steps in the learning rate schedule."}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "Number of gradient accumulation steps."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Use gradient checkpointing to reduce memory footprint."}
    )
    max_train_steps: Optional[int] = field(default=50_000, metadata={"help": "Maximum number of training steps."})
    max_eval_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    )
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Sequence lengths used for training."})
    seed: Optional[int] = field(default=1, metadata={"help": "Training seed."})
    save_checkpoint_steps: Optional[int] = field(
        default=1024,
        metadata={"help": "Interval to save checkpoints. Measured as number of forward passes not training steps."},
    )


@dataclass
class EvaluationArguments:
    """
    Configuration for evaluating model.
    """

    model_ckpt: Optional[str] = field(
        default="lvwerra/codeparrot",
        metadata={"help": "Model name or path of model to be evaluated."},
    )
    dataset_name: Optional[str] = field(
        default="lvwerra/codeparrot-clean-valid", metadata={"help": "Name or path of validation dataset."}
    )
    batch_size: Optional[int] = field(default=2, metadata={"help": "Batch size used for evaluation."})
    max_eval_steps: Optional[int] = field(
        default=-1, metadata={"help": "Maximum number of evaluation steps. If -1 the full dataset is evaluated."}
    )
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Length of sequences to be evaluated."})
    seed: Optional[int] = field(default=1, metadata={"help": "Random seed used for evaluation."})


@dataclass
class HumanEvalArguments:
    """
    Configuration for running evaluation on HumanEval dataset.
    """

    model_ckpt: Optional[str] = field(
        default="lvwerra/codeparrot",
        metadata={"help": "Model name or path of model to be evaluated."},
    )
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers used for code evaluation."})
    do_sample: Optional[bool] = field(
        default=True, metadata={"help": "Sample from the language model's output distribution."}
    )
    temperature: Optional[float] = field(default=0.2, metadata={"help": "Sampling temperature used for generation."})
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "Maximum number of newly generated tokens."})
    top_k: Optional[int] = field(default=0, metadata={"help": "Top-k parameter used for generation."})
    top_p: Optional[float] = field(default=0.95, metadata={"help": "Top-p parameter used for nucleus sampling."})
    batch_size: Optional[int] = field(default=10, metadata={"help": "Number of generations to run in parallel."})
    n_samples: Optional[int] = field(
        default=200, metadata={"help": "Number of completions to generate for each sample."}
    )
    seed: Optional[int] = field(default=1, metadata={"help": "Random seed used for evaluation."})
    output_file: Optional[str] = field(
        default="eval_results.json", metadata={"help": "Random seed used for evaluation."}
    )
    HF_ALLOW_CODE_EVAL: Optional[str] = field(
        default="0", metadata={"help": "Allow `code_eval` to execute Python code on machine"}
    )


@dataclass
class PreprocessingArguments:
    """
    Configuration for preprocessing data.
    """

    num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of CPU cores to use for parallel preprocessing. Default uses the maximum available."
        },
    )
    dataset_name: Optional[str] = field(
        default="codeparrot", metadata={"help": "Folder or name of dataset to process."}
    )
    output_dir: Optional[str] = field(
        default="codeparrot-clean", metadata={"help": "Folder to save processed processed dataset."}
    )
    samples_per_file: Optional[int] = field(
        default=100_000, metadata={"help": "Number of files to save per JSON output file."}
    )
    text_column: Optional[str] = field(default="content", metadata={"help": "Column containing text data to process."})
    line_max: Optional[float] = field(
        default=1000, metadata={"help": "Maximum line length in file, otherwise file is filtered."}
    )
    line_mean: Optional[float] = field(
        default=100, metadata={"help": "Maximum mean line length in file, otherwise file is filtered."}
    )
    alpha_frac: Optional[float] = field(
        default=0.25, metadata={"help": "Maximum fraction of non-alphanumeric characters, otherwise file is filtered."}
    )


@dataclass
class TokenizerTrainingArguments:
    """
    Configuration for tokenizer training.
    """

    base_tokenizer: Optional[str] = field(
        default="gpt2",
        metadata={"help": "Base tokenizer to build new tokenizer from."},
    )
    dataset_name: Optional[str] = field(
        default="transformersbook/codeparrot-train", metadata={"help": "Dataset to train tokenizer on."}
    )
    text_column: Optional[str] = field(default="content", metadata={"help": "Column containing text data to process."})
    vocab_size: Optional[int] = field(default=200000, metadata={"help": "Number of examples to train tokenizer on."})
    n_examples: Optional[int] = field(
        default=32768, metadata={"help": "Number of examples to train the tokenizer on."}
    )
    tokenizer_name: Optional[str] = field(default="codeparrot", metadata={"help": "Name of new tokenizer."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenizer to the hub."})


@dataclass
class InitializationArguments:
    """
    Configuration for initializing new model.
    """

    config_name: Optional[str] = field(
        default="gpt2-large",
        metadata={"help": "Configuration to use for model initialization."},
    )
    tokenizer_name: Optional[str] = field(
        default="lvwerra/codeparrot", metadata={"help": "Tokenizer attached to model."}
    )
    model_name: Optional[str] = field(default="codeparrot", metadata={"help": "Name of the created model."})
    push_to_hub: Optional[bool] = field(default=True, metadata={"help": "Push saved tokenizer to the hub."})
