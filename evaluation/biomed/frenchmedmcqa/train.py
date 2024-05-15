import os
import sys
import json
import logging
from argparse import ArgumentParser
from functools import partial
from shutil import rmtree
from datetime import datetime
from warnings import simplefilter

import numpy as np
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics._classification import UndefinedMetricWarning
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    set_seed
)
from torch import manual_seed
from torch.optim import AdamW

HERE = os.path.dirname(__file__)
sys.path.append(os.path.normpath(os.path.join(HERE, os.pardir)))
from finetuning_cl_args_common import add_common_training_arguments

logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
LOGFMT = "%(asctime)s - %(levelname)s - \t%(message)s"
logging.basicConfig(format=LOGFMT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)
simplefilter(action="ignore", category=UndefinedMetricWarning)


OUTPUT_DIR_HELP = """Where to put the output directory"""
SEED_HELP = """Sets the base random state for the script, including the generation
of seeds for multiple runs"""
AUTH_HELP = """Authorisation token for loading a model from the HuggingFace hub, if
required"""
TRC_HELP = """Set this flag when the model being loaded requires custom code to be
run (as is the case for the Jargon models); if not set, transformers will show a
command prompt asking for confirmation"""


def parse_arguments():
    default_data_script_path = os.path.normpath(
        os.path.join(
            HERE, "frenchmedmcqa.py"
        )
    )
    default_output_dir = None
    parser = add_common_training_arguments(ArgumentParser())
    parser.add_argument("model", type=str)
    parser.add_argument("--dataloader", type=str, default=default_data_script_path)
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help=OUTPUT_DIR_HELP)
    parser.add_argument("--seed", type=int, default=42, help=SEED_HELP)
    parser.add_argument("--auth", type=str, help=AUTH_HELP)
    parser.add_argument("--trust_remote_code", action="store_true", help=TRC_HELP)
    return parser.parse_args()


def tokenize_fn(example, tokenizer, max_length):
    enc = tokenizer(
        example["bert_text_no_ctx"],
        truncation=True,
        max_length=max_length
    )
    return enc


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {"f1_w": f1, "precision_w": precision, "recall_w": recall}


def main(args):
    logger.info("Loading data...")
    set_seed(args.seed)
    dataset = load_dataset(args.dataloader)
    train_data = dataset.get("train")
    eval_data = dataset.get("validation")
    
    logger.info("Model setup: %s", args.model)
    # trust_remote_code = args.model.startswith("TCMVince")
    # auth_token = "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" if trust_remote_code else None
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code, use_auth_token=args.auth
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=31, trust_remote_code=args.trust_remote_code, use_auth_token=args.auth
    )
    optimizer = AdamW(tuple(model.parameters()), lr=args.lr)
    scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup) \
        if args.warmup else get_constant_schedule(optimizer)
    
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenize_map_fn = partial(tokenize_fn, tokenizer=tokenizer, max_length=model.config.max_position_embeddings)
    train_dataset = train_data.map(tokenize_map_fn, batched=True)
    eval_dataset = eval_data.map(tokenize_map_fn, batched=True)

    if "/" in args.model:
        _, output_name_model = os.path.split(args.model)
    else:
        output_name_model = args.model
    now = datetime.now()
    output_subdir = f"{output_name_model}_fmcqa_{now.day}-{now.month}_{now.hour}-{now.minute}"
    if not os.path.isdir(args.output_dir) and args.save != "no":
        os.mkdir(args.output_dir)
    output_dir = os.path.join(args.output_dir, output_subdir)
    if os.path.isdir(output_dir):
        rmtree(output_dir)
    os.mkdir(output_dir)
    with open(os.path.join(output_dir, "cl_args.json"), "w") as f:
        json.dump(vars(args), f)
    rng = np.random.default_rng(seed=args.seed)
    seeds = rng.integers(0, np.iinfo(np.int16).max, size=args.runs, dtype=np.int16)
    for run, seed in enumerate(seeds):
        manual_seed(seed)
        run_output_dir = os.path.join(output_dir, "run" + str(run + 1))
        training_arguments = TrainingArguments(
            output_dir=run_output_dir,
            weight_decay=args.weight_decay,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_acc,
            eval_accumulation_steps=args.eval_acc,  # n. batches to send to the CPU at a time
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            max_steps=args.steps,
            evaluation_strategy="steps" if args.steps > 0 else "epoch",
            save_strategy=args.save,
            save_steps=args.steps,
            fp16=True,
            seed=int(seed)  # np.int16 is not JSON serializable
        )
        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            optimizers=(optimizer, scheduler),
            compute_metrics=compute_metrics
        )
        logger.info("Launching training run %d...", run + 1)
        trainer.train()
    logger.info("All done; outputs @ %s\n%s\n", output_dir, 50 * '=')
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    