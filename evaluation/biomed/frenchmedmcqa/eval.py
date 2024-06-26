import os
import sys
import json
import logging
from argparse import ArgumentParser
from itertools import chain, combinations
from datetime import datetime

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline
)
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
logger = logging.getLogger(__name__)
LOGFMT = "%(asctime)s - %(levelname)s - \t%(message)s"
logging.basicConfig(format=LOGFMT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)

MODEL_DIR_HELP = """Path to a directory containing fine-tuning runs for the FrenchMedMCQA task,
as generated by train.py"""
DATALOADER_HELP = """Dataset loader script defining the `FrenchMedMCQA` dataset
builder object, allowing for compatibility with the `datasets` library"""
OUTPUT_DIR_HELP = """Where to put the output. If not provided, will create a `results` directory
in the same place as this script"""
HERE = os.path.dirname(__file__)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("model_dir", type=str, help=MODEL_DIR_HELP)
    default_data_script_path = os.path.normpath(
        os.path.join(
            HERE, "frenchmedmcqa.py"
        )
    )
    parser.add_argument("--dataloader", type=str, default=default_data_script_path)
    parser.add_argument("--output_dir", type=str)
    return parser.parse_args()


def get_label_ref(data_script_path):
    script_dir, script_name = os.path.split(data_script_path)
    sys.path.append(script_dir)
    module_name = script_name.replace(".py", "")
    exec("import " + module_name)
    builder_class = getattr(eval(module_name), "FrenchMedMCQA")
    return builder_class.generate_label_names()


def load_checkpoint(cp_dir, num_labels):
    kwargs = {"num_labels": num_labels}
    model = AutoModelForSequenceClassification.from_pretrained(cp_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cp_dir, trust_remote_code=True)
    return model, tokenizer


def compute_exact_match_and_hamming(preds, refs):
    exact_score, overlap_score = [], []
    for p, r in zip(preds, refs):
        exact_score.append(p == r)
        n_correct = sum(letter in r for letter in p)
        n_total = len(set(p + r))
        overlap_score.append(n_correct / n_total)
    emr = sum(exact_score) / len(exact_score)
    hamming = sum(overlap_score) / len(overlap_score)
    return emr, hamming


def main(args):
    logger.info("Loading test dataset using script @ %s", args.dataloader)
    test_dataset = load_dataset(args.dataloader).get("test")
    label_ref = get_label_ref(args.dataloader)
    num_labels = len(label_ref)
    text, labels = map(lambda x: [ex.get(x) for ex in test_dataset], ("bert_text_no_ctx", "label"))
    y_true = [label_ref[label] for label in labels]
    all_results = {}
    for i, model_run_dir in enumerate(os.listdir(args.model_dir)):
        model_run_path = os.path.join(args.model_dir, model_run_dir)
        emr_list, hamming_list = [], []
        for exp_element in os.listdir(model_run_path):
            exp_element_path = os.path.join(model_run_path, exp_element)
            if os.path.isdir(exp_element_path) and exp_element.startswith("run"):
                run_elements = os.listdir(exp_element_path)
                checkpoint_to_use = max(int(elem.replace("checkpoint-", "")) for elem in run_elements \
                    if elem.startswith("checkpoint"))  # use latest checkpoint
                checkpoint_path = os.path.join(exp_element_path, f"checkpoint-{checkpoint_to_use}")
                with open(os.path.join(checkpoint_path, "config.json")) as f:
                    model_name = json.load(f).get("_name_or_path", "model")
                logger.info("Loading model from %s...", checkpoint_path)
                model, tokenizer = load_checkpoint(checkpoint_path, num_labels=num_labels)
                pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
                outputs = pipeline(text, truncation=True, max_length=model.config.max_position_embeddings)
                y_pred = [label_ref[int(output.get("label", "0").split("_")[-1])] for output in outputs]
                emr, hamming = compute_exact_match_and_hamming(y_pred, y_true)
                emr_list.append(emr)
                hamming_list.append(hamming)
        emr_arr = 100 * np.array(emr_list)
        emr, emr_std = emr_arr.mean(), emr_arr.std()
        hamming_arr = 100 * np.array(hamming_list)
        hamming, hamming_std = hamming_arr.mean(), hamming_arr.std()
        logger.info("EMR = %.3f +/- %.3f, Hamming = %.2f +/- %.3f", emr, emr_std, hamming, hamming_std)
        all_results[model_run_dir] = {
            "emr": {"mean": emr, "std": emr_std},
            "hamming": {"mean": hamming, "std": hamming_std}
        }
    now = datetime.now()
    output_filename = f"eval-results_{now.day}-{now.month}_{now.hour}-{now.minute}.json"
    if args.output_dir is None:
        args.output_dir = os.path.normpath(os.path.join(HERE, "results"))
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    output_fp = os.path.join(args.output_dir, output_filename)
    with open(output_fp, "w") as f:
        json.dump(all_results, f)
    logger.info("%s Finished, output file @ %s %s", "=" * 5, output_fp, "=" * 5)
    

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
