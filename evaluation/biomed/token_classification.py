import os
import json
import logging
from pathlib import Path
from argparse import ArgumentParser
from warnings import filterwarnings
from functools import partial
from itertools import chain
from collections import defaultdict
from datetime import datetime
from shutil import rmtree

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    Trainer,
    TrainingArguments,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    set_seed
)
from datasets import load_dataset

from finetuning_cl_args_common import add_common_training_arguments


logger = logging.getLogger(__name__)
LOGFMT = "%(asctime)s - %(levelname)s - \t%(message)s"
logging.basicConfig(format=LOGFMT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)
filterwarnings(action="ignore", category=UserWarning)

DATA_FP_HELP = """Data file path. Expects a folder containing train.json, dev.json and
test.json files, or a valid argument to pass to datasets.load_dataset, in the case
where the hfload option is activated."""
MODEL_HELP = """Path to a pre-trained PyTorch BERT-style model compatible with the
transformers token classification model."""
LABEL_NAME_HELP = """The target class variable name in the dataset specified by data_path
"""
TEXT_NAME_HELP = """The name of the text component in the input data file"""
OUTPUT_DIR_HELP = """Where to put the output directory"""
DATA_OUT_NAME_HELP = """A name to use for the dataset in the output directory; by default,
the last name in the file path `data_path` will be used"""
SEED_HELP = """Sets the base random state for the script, including the generation
of seeds for multiple runs"""
EVAL_SPLIT_NAME_HELP = """The name of the evaluation split in the input dataset
(this can vary - 'val', 'validation', 'dev', etc. - depending on the dataset)"""
GPU_HELP = """GPU device index (optional, defaults to 0)"""
NOEVAL_HELP = """Include this when the dataset only has a train/test split, without
a validation set"""
SKIP_TEST_HELP = """Only run the training loop on the train & dev sets; for
debugging etc."""
TRC_HELP = """Set this flag when the model being loaded requires custom code to be
run (as is the case for the Jargon models); if not set, transformers will show a
command prompt asking for confirmation"""
HFLOAD_HELP = """Use the Huggingface datasets library to load the input dataset"""
BIO_HELP = """Add BIO schema formatting to the target labels"""
ADD_NONE_HELP = """Add a placeholder 'none' class to the target labels, which will
become class 0 in the collated input labels - for when not all of the tokens in the
input text have been labelled"""
AUTH_HELP = """Authorisation token for loading a model from the HuggingFace hub, if
required"""
METRIC_AVG = "micro", "macro", "weighted"
METRICS = "_precision", "_recall", "_f1"


class BatchEncodingDataset(Dataset):

    def __init__(self, data_, return_tensors=True):
        super().__init__()
        if isinstance(data_, BatchEncoding):
            self._init_sequences(data_)
        elif isinstance(data_, (list, tuple)):
            for encoding in data_:
                if isinstance(encoding, BatchEncoding):
                    if hasattr(self, "_keys"):
                        for k in self._keys:
                            self.__dict__[k] += encoding[k]
                    else:
                        self._init_sequences(encoding)
        self.return_tensors = return_tensors

    def __len__(self):
        return len(getattr(self, self._keys[0]))

    def __getitem__(self, item):
        if self.return_tensors:
            return {key: torch.as_tensor(getattr(self, key)[item]) for key in self._keys}
        return {key: getattr(self, key)[item] for key in self._keys}

    def _init_sequences(self, batch_encoding):
        self._keys = tuple(batch_encoding.keys())
        sequences = dict(batch_encoding.items())
        self.__dict__.update(sequences)


def parse_arguments():
    parser = add_common_training_arguments(ArgumentParser())
    parser.add_argument("--data_path", type=str, help=DATA_FP_HELP)
    parser.add_argument("--label_name", type=str, default="tags", help=LABEL_NAME_HELP)
    parser.add_argument("--text_name", type=str, default="text", help=TEXT_NAME_HELP)
    parser.add_argument("--word2token_overflow", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int)
    default_output_dir = os.path.join(os.getenv("HOME"), "token-clf-eval")
    parser.add_argument("--output_dir", type=str, default=default_output_dir, help=OUTPUT_DIR_HELP)
    parser.add_argument("--dataset_output_name", type=str)
    parser.add_argument("--seed", type=int, default=42, help=SEED_HELP)
    parser.add_argument("--subset_name", type=str)
    parser.add_argument("--eval_split_name", type=str, default="validation",
        help=EVAL_SPLIT_NAME_HELP)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--set_model_pad_token_id", action="store_true", help=NOEVAL_HELP)
    parser.add_argument("--no_eval", action="store_true", help=NOEVAL_HELP)
    parser.add_argument("--skip_test", action="store_true", help=SKIP_TEST_HELP)
    parser.add_argument("--trust_remote_code", action="store_true", help=TRC_HELP)
    parser.add_argument("--hfload", action="store_true", help=HFLOAD_HELP)
    parser.add_argument("--bio", action="store_true", help=BIO_HELP)
    parser.add_argument("--add_none", action="store_true", help=ADD_NONE_HELP)
    parser.add_argument("--auth", type=str, help=AUTH_HELP)
    return parser.parse_args()


def labels2bio(labels):
    res = []
    for label_list in labels:
        prev_label = None
        bio_labels = []
        for label in label_list:
            if label == "none":
                bio_labels.append("O")
            elif label == prev_label:
                bio_labels.append("I-" + label)
            else:
                bio_labels.append("B-" + label)
            prev_label = label
        res.append(bio_labels)
    return res


def load_ner_data(fp, label_name, text_name):
    if not os.path.isfile(fp):
        text = labels = None
    else:
        with open(fp, "r") as f:
            dataset = json.load(f)
        text, labels = [], []
        for doc in dataset.values():
            text.append(doc[text_name].split() if isinstance(doc, str) else doc[text_name])
            doc_labels = doc[label_name]
            if isinstance(doc_labels, list):
                labels.append(doc_labels)
            elif isinstance(doc_labels, str):
                labels.append(doc_labels.split())
    return text, labels


def make_word_ids(input_ids, attn_mask, tokenizer, text_):
    word_ids = [None]  # sequence will always start with cls
    current_id = 0
    for ii, attn in zip(input_ids[1:], attn_mask[1:]):
        if not attn:
            word_ids.append(None)
            continue
        try:
            next_word = text_[current_id + 1]
        except IndexError:
            word_ids.append(None)
            continue
        if next_word.startswith(tokenizer.decode(ii)):
            # start of the next word
            current_id += 1
        word_ids.append(current_id)
    return word_ids


def tokenize_and_align(text, labels, tokenizer, label2id, split_into_words):
    """Runs tokenisation on the given text and aligns the target labels with the resulting
    subword identifiers"""
    input_encoding = None
    if text and labels:
        if split_into_words:
            text = [doc.split() for doc in text]
        input_encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            is_split_into_words=True,
            return_overflowing_tokens=True
        )
        aligned_labels = []
        text_idx = []
        if hasattr(input_encoding, "overflowing_tokens"):
            overflowing_tokens = input_encoding["overflowing_tokens"]
            if isinstance(overflowing_tokens[0], int):
                tmp = []
                x = len(overflowing_tokens) / tokenizer.model_max_length
                n_full_overflow_seq = int(x)
                for i in range(n_full_overflow_seq):
                    seq = overflowing_tokens[i * tokenizer.model_max_length:(i + 1) * tokenizer.model_max_length]
                    tmp.append(seq)
                if x > n_full_overflow_seq:
                    tmp.append(overflowing_tokens[(i + 1) * tokenizer.model_max_length:])
                overflowing_tokens = tmp
            if not tokenizer.is_fast and any(overflowing_tokens):
                n_overflows_added = 0
                for i, overflow in enumerate(overflowing_tokens):
                    text_idx.append(i)
                    if not overflow:
                        continue
                    overflow_idx = 0
                    while overflow_idx < len(overflow):
                        overflow_ = overflow[overflow_idx:]
                        overflow_.insert(0, tokenizer.bos_token_id)
                        overflow_.insert(tokenizer.model_max_length, tokenizer.sep_token_id)
                        if len(overflow_) < tokenizer.model_max_length:
                            attn_mask_ = np.ones(len(overflow_)).tolist()
                            while len(overflow_) < tokenizer.model_max_length:
                                overflow_.append(tokenizer.pad_token_id)
                            while len(attn_mask_) < tokenizer.model_max_length:
                                attn_mask_.append(0)
                        else:
                            attn_mask_ = np.ones(tokenizer.model_max_length).tolist()
                        insertion_idx = i + n_overflows_added + 1
                        input_encoding["input_ids"].insert(insertion_idx, overflow_[:tokenizer.model_max_length])
                        input_encoding["attention_mask"].insert(insertion_idx, attn_mask_)
                        n_overflows_added += 1
                        text_idx.append(i)
                        overflow_idx += tokenizer.model_max_length - 2
                for del_key in ("overflowing_tokens", "num_truncated_tokens", "token_type_ids"):
                    del input_encoding[del_key]
        n_sequences = len(input_encoding["input_ids"])
        original_idx = -1
        for i in range(n_sequences):
            if text_idx:
                original_idx = text_idx[i]
            if not tokenizer.is_fast:
                input_ids, attn_mask = input_encoding["input_ids"][i], \
                    input_encoding["attention_mask"][i]
                text_ = text[original_idx] if isinstance(text, list) else text[original_idx].split()
                word_ids = make_word_ids(input_ids, attn_mask, tokenizer, text_)
            else:
                word_ids = input_encoding.word_ids(batch_index=i)
            if not text_idx and word_ids[1] == 0:
                original_idx += 1
            prev_word_id = None
            label_ids = []
            for word_id in word_ids:
                if word_id is None:
                    # special tokens
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    # first token of the word
                    try:
                        k = labels[original_idx][word_id]
                        if not isinstance(k, str):
                            k = str(k)
                        id_ = label2id[k]
                    except KeyError:
                        # label in the dev set that's not in the training set; ignore
                        id_ = -100
                    label_ids.append(id_)
                else:
                    # subsequent subword tokens
                    label_ids.append(-100)
                prev_word_id = word_id
            aligned_labels.append(label_ids)
        input_encoding["labels"] = aligned_labels
    return input_encoding


def main(args):
    logger.info("Loading dataset from %s", args.data_path)
    if args.hfload:
        train_data = load_dataset(
            args.data_path,
            split="train",
            name=args.subset_name,
            trust_remote_code=args.trust_remote_code
        )
        dev_data = load_dataset(
            args.data_path,
            split=args.eval_split_name,
            name=args.subset_name,
            trust_remote_code=args.trust_remote_code
        ) if not args.no_eval else None
        train_text, train_labels = train_data[args.text_name], train_data[args.label_name]
        if not args.no_eval:
            dev_text, dev_labels = dev_data[args.text_name], dev_data[args.label_name]
        else:
            dev_text = dev_labels = None
    else:
        train_data, dev_data = map(
            lambda x: load_ner_data(
                os.path.join(args.data_path, x + ".json"),
                label_name=args.label_name, text_name=args.text_name
            ), ("train", "dev")
        )
        train_text, train_labels = train_data
        dev_text, dev_labels = dev_data

    label_values = []
    for label_list in train_labels:
        labelset = set(label_list)
        for label in labelset:
            if label not in label_values and label != "none":
                label_values.append(label)
    label_values.sort()
    labels_are_strings = any(isinstance(val, str) for val in label_values)
    if labels_are_strings:
        if args.bio:
            id_ = 0
            label2id = {"O": id_}
            for label_value in label_values:
                id_ += 1
                label2id["B-" + label_value] = id_
                id_ += 1
                label2id["I-" + label_value] = id_
            train_labels, dev_labels = map(labels2bio, (train_labels, dev_labels))
        else:
            label2id = {
                label_value: i for i, label_value in enumerate(
                    ["none", *label_values] if args.add_none else label_values
                )
            }
    else:
        label2id = {str(i): i for i in label_values}
    id2label = {v: k for k, v in label2id.items()}
    model_init_kwargs = {"num_labels": len(label2id), "id2label": id2label, "label2id": label2id}
    if args.gpu:
        if args.gpu < torch.cuda.device_count():
            torch.cuda.set_device(args.gpu)
            logger.info("Model setup on GPU %d...", torch.cuda.current_device())
        else:
            logger.warning("GPU %d not available; using %d...", args.gpu, torch.cuda.current_device())
    else:
        logger.info("Model setup")
    if args.auth is not None:
        hub_kwargs = {"use_auth_token": args.auth, "trust_remote_code": True}
    else:
        hub_kwargs = {"trust_remote_code": args.trust_remote_code}
    model_init_kwargs.update(hub_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **hub_kwargs)
    if args.set_model_pad_token_id:
        model_init_kwargs["pad_token_id"] = tokenizer.pad_token_id
    model = AutoModelForTokenClassification.from_pretrained(args.model, **model_init_kwargs)
    optimizer = AdamW(tuple(model.parameters()), lr=args.lr)
    scheduler = get_constant_schedule_with_warmup(optimizer, args.warmup) \
        if args.warmup else get_constant_schedule(optimizer)

    logger.info("Tokenizing text & constructing encoded datasets...")
    set_seed(args.seed)
    if args.seq_len < 1:
        # calculate the maximal sequence length from the data
        len_fn = len if isinstance(train_text[0], list) else lambda x: len(x.replace('\n', ' ').split())
        text_list = train_text + dev_text if dev_text else train_text
        max_len = max(len_fn(t) for t in text_list) + args.word2token_overflow
        if hasattr(model, "roberta"):
            model_max_embedding_len = model.roberta.embeddings.position_embeddings.num_embeddings - 2
            max_len = min(max_len, model_max_embedding_len)
        args.seq_len = max_len
    if hasattr(tokenizer, "max_length"):
        tokenizer.max_length = args.seq_len
    else:
        tokenizer.model_max_length = args.seq_len
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    input_encoding_train, input_encoding_dev = map(
        partial(
            tokenize_and_align,
            tokenizer=tokenizer,
            label2id=label2id,
            split_into_words=False
        ),
        (train_text, dev_text), (train_labels, dev_labels)
    )
    train_dataset = BatchEncodingDataset(input_encoding_train)
    if input_encoding_dev:
        dev_dataset = BatchEncodingDataset(input_encoding_dev)
        evaluation_strategy = "epoch"
    else:
        dev_dataset = None
        evaluation_strategy = "no"

    logger.info("Training setup...")
    def compute_metrics(input_):
        clf_output_predictions, labels = input_
        class_predictions = np.argmax(clf_output_predictions, axis=2)
        per_sequence_metrics = defaultdict(list)
        for prediction, label in zip(class_predictions, labels):
            preds, refs = [], []
            for pred, lbl in zip(prediction, label):
                if lbl != -100:
                    preds.append(id2label[pred])
                    refs.append(id2label[lbl])
            for average in METRIC_AVG:
                metric_values = precision_recall_fscore_support(
                    refs, preds, average=average
                )
                for name, val in zip(METRICS, metric_values):
                    per_sequence_metrics[average + name].append(val)
        return {k: sum(v) / len(v) for k, v in per_sequence_metrics.items()}
    
    output_name_model = Path(args.model).name
    dataset_output_name = args.dataset_output_name if args.dataset_output_name \
        else Path(args.data_path).name
    # _, dataset_output_name = os.path.split(args.data_path if args.data_path[-1] != "/" else args.data_path[:-1])
    if args.subset_name:
        dataset_output_name += "_" + args.subset_name
    now = datetime.now()
    output_subdir = f"{output_name_model}_{dataset_output_name}_{now.day}-{now.month}_{now.hour}-{now.minute}"
    if not os.path.isdir(args.output_dir) and args.save != "no":
        os.mkdir(args.output_dir)
    output_dir = os.path.join(args.output_dir, output_subdir)
    if os.path.isdir(output_dir):
        rmtree(output_dir)
    os.mkdir(output_dir)
    with open(os.path.join(output_dir, "cl_args.json"), "w") as f:
        json.dump(vars(args), f)
    per_run_results = defaultdict(list)
    rng = np.random.default_rng(seed=args.seed)
    seeds = rng.integers(0, np.iinfo(np.int16).max, size=args.runs, dtype=np.int16)
    metric_names = {avg + met for met in METRICS for avg in METRIC_AVG}
    eval_batch_size = args.eval_batch_size if args.eval_batch_size else args.batch_size
    eval_acc = args.eval_acc if args.eval_acc else args.grad_acc
    for run, seed in enumerate(seeds):
        torch.manual_seed(seed)
        run_output_dir = os.path.join(output_dir, "run" + str(run + 1))
        train_args = TrainingArguments(
            output_dir=run_output_dir,
            weight_decay=args.weight_decay,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=args.grad_acc,
            eval_accumulation_steps=eval_acc,  # n. batches to send to the CPU at a time
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            max_steps=args.steps,
            evaluation_strategy=evaluation_strategy,
            save_strategy=args.save,
            save_steps=args.steps,  # just save the final model (no intermediate checkpoints) if args.save=='steps'
            fp16=True,
            seed=int(seed)  # np.int16 is not JSON serializable
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer,
            optimizers=(optimizer, scheduler),
            compute_metrics=compute_metrics
        )
        logger.info("Launching training run %d...", run + 1)
        trainer.train()

        if not args.skip_test:
            logger.info("Loading & encoding test dataset...")
            if args.hfload:
                test_data = load_dataset(
                    args.data_path,
                    split="test",
                    name=args.subset_name,
                    trust_remote_code=args.trust_remote_code
                )
                test_text, test_labels = test_data[args.text_name], test_data[args.label_name]
            else:
                test_text, test_labels = load_ner_data(
                    os.path.join(args.data_path, "test.json"),
                    args.label_name, args.text_name
                )
            test_encoding = tokenize_and_align(
                test_text, test_labels, tokenizer, label2id,
                split_into_words=False
            )
            logger.info("Running model on test dataset...")
            test_dataset = BatchEncodingDataset(test_encoding)
            test_results = trainer.evaluate(test_dataset)
            with open(os.path.join(run_output_dir, "results.json"), "w") as f:
                json.dump(test_results, f)
            for k, v in test_results.items():
                metric_name = k.replace("eval_", "")
                if metric_name in metric_names:
                    per_run_results[metric_name].append(v)
    averaged_results = {}
    if per_run_results:
        for k, v in per_run_results.items():
            arr = np.array(v)
            averaged_results[k] = {"mean": arr.mean(), "std": arr.std()}
        logger.info(
            "-- Test Set Results: averaged over %d runs -- \
            \nMicro: P=%.5f +/- %.5f, R=%.5f +/- %.5f, F=%.5f +/- %.5f \
            \nMacro: P=%.5f +/- %.5f, R=%.5f +/- %.5f, F=%.5f +/- %.5f \
            \nWeighted: P=%.5f +/- %.5f, R=%.5f +/- %.5f, F=%.5f +/- %.5f", args.runs,
            *list(chain(*[v.values() for v in averaged_results.values()]))
        )
    if args.save != "no":
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(averaged_results, f)
        with open(os.path.join(output_dir, "run_seeds.txt"), "w") as f:
            f.write("\n".join(seeds.astype(str).tolist()))
        logger.info("Done! Files in %s", output_dir)
    else:
        logger.info("Done!")
    print(f"\n{50 * '='}\n")


if __name__ == "__main__":
    main(parse_arguments())
