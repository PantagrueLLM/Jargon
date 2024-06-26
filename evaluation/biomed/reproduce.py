import os
import sys
import json
from importlib import import_module
from argparse import ArgumentParser

TASKS = [
    "fmcqa",
    # "mqc",
    "caspos",
    "essaipos",
    "cassg",
    "medline",
    "emea",
    "e3c",
    "clister"
]
MODELS = [
    "PantagrueLLM/jargon-general-base",
    "PantagrueLLM/jargon-general-biomed",
    "PantagrueLLM/jargon-multidomain-base",
    "PantagrueLLM/jargon-biomed",
    "PantagrueLLM/jargon-biomed-4096",
    "PantagrueLLM/jargon-NACHOS",
    "PantagrueLLM/jargon-NACHOS-4096",
    "emilyalsentzer/Bio_ClinicalBERT",
    "dmis-lab/biobert-v1.1",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    "almanach/camembert-base",
    "almanach/camembert-base-ccnet-4gb",
    "almanach/camembert-base-oscar-4gb",
    "almanach/camembert-bio-base",
    "medicalai/ClinicalBERT",
    "Dr-BERT/DrBERT-4GB",
    "Dr-BERT/DrBERT-7GB",
    "flaubert/flaubert_base_cased",
    "flaubert/flaubert_large_cased",
    "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
]
HERE = os.path.dirname(__file__)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("task", type=str, choices=TASKS)
    parser.add_argument("model", type=str, choices=MODELS)
    return parser.parse_args()


def main(args):
    if args.task == "fmcqa":
        sys.path.append(os.path.join(HERE, "frenchmedmcqa"))
        script = "train"
    elif args.task == "clister":
        script = "clister"
    else:
        script = "token_classification"
    xpmodule = import_module(script)
    xpfunc = xpmodule.main
    xpargparser = xpmodule.parse_arguments
    with open(os.path.join(HERE, "experiment_params", args.task + ".json")) as f:
        xpargs = json.load(f)
    args = xpargparser()  # defaults
    args.__dict__.update(xpargs)
    xpfunc(args)


if __name__ == "__main__":
    main(parse_arguments())
