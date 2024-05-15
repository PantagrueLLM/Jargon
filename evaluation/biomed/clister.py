import os
import json
import logging
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import pandas as pd
from torch import manual_seed
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.util import cos_sim

from finetuning_cl_args_common import add_common_training_arguments

logger = logging.getLogger(__name__)
LOGFMT = "%(asctime)s - %(levelname)s - \t%(message)s"
logging.basicConfig(format=LOGFMT, datefmt="%d/%m/%Y %H:%M:%S", level=logging.INFO)

DATAPATH_HELP = """Path to the directory containing the files `clister_train.csv`
and `clister_test.csv"""
OUTPUT_DIR_HELP = """Where to put the results. If not provided, will create a `clister-results`
directory in the same place as this script"""
NOTQDM_HELP = """Flag to disable the progress bar"""
SEED_HELP = """Random seed"""
AUTH_HELP = """Authorisation token for loading a model from the HuggingFace hub, if
required"""
TRC_HELP = """Set this flag when the model being loaded requires custom code to be
run (as is the case for the Jargon models); if not set, transformers will show a
command prompt asking for confirmation"""


def parse_arguments():
    parser = add_common_training_arguments(ArgumentParser())
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str, help=OUTPUT_DIR_HELP)
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help=SEED_HELP)
    parser.add_argument("--auth", type=str, help=AUTH_HELP)
    parser.add_argument("--trust_remote_code", action="store_true", help=TRC_HELP)
    return parser.parse_args()


def main(args):
    logger.info("Setup...")
    train_df = pd.read_csv(os.path.join(args.data_path, "clister_train.tsv"), sep="\t")
    model_args = {
        "trust_remote_code": args.trust_remote_code,
        "use_auth_token": args.auth
    }
    transformer = Transformer(args.model, max_seq_length=args.seq_len, model_args=model_args)
    transformer.tokenizer.model_input_names = ["input_ids", "attention_mask"]
    pooling_layer = Pooling(transformer.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
    model = SentenceTransformer(modules=(transformer, pooling_layer))
    loss_function = CosineSimilarityLoss(model=model)
    optimizer_params = {"lr": args.lr}
    scheduler_type = "warmupconstant" if args.warmup else "constantlr"
    if args.output_dir is None:
        args.output_dir = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), "clister-results"
            )
        )
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    now = datetime.now()
    if "/" in args.model:
        _, model_name = os.path.split(args.model)
    else:
        model_name = args.model
    output_subdir = f"{model_name}_{now.day}-{now.month}_{now.hour}-{now.minute}"
    output_path = os.path.join(args.output_dir, output_subdir)
    manual_seed(args.seed)
    train_dataset = list(InputExample(texts=[t.id_1, t.id_2], label=(t.sim / 5)) for t in train_df.itertuples())
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model.fit(
        train_objectives=[(dataloader, loss_function)],
        epochs=args.epochs,
        scheduler=scheduler_type,
        warmup_steps=args.warmup,
        optimizer_params=optimizer_params,
        weight_decay=args.weight_decay,
        # output_path=output_path,
        show_progress_bar=not args.no_tqdm,
        # safe_serialization=False
    )
    logger.info("Running evaluation on test set...")
    test_dataset = pd.read_csv(os.path.join(args.data_path, "clister_test.tsv"), sep="\t")
    s1_embeddings, s2_embeddings = model.encode(test_dataset.id_1), model.encode(test_dataset.id_2)
    cosine_similarities = np.fromiter(
        (cos_sim(*embeddings).item() for embeddings in zip(s1_embeddings, s2_embeddings)),
        dtype=np.float64
    )
    result = spearmanr(test_dataset.sim, cosine_similarities)
    logger.info("Spearman correlation for test data: %.5f @ p=%.9f", result.statistic, result.pvalue)
    if output_path:
        with open(os.path.join(output_path, "cl_args.json"), "w") as f:
            json.dump(vars(args), f)
        with open(os.path.join(output_path, "results.json"), "w") as f:
            json.dump({attr: getattr(result, attr) for attr in ("statistic", "pvalue")}, f)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
