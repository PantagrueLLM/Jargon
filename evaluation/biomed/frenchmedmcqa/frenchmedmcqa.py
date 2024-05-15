# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FrenchMedMCQA : A French Multiple-Choice Question Answering Corpus for Medical domain"""

import os
import json
from itertools import chain, combinations

import pandas as pd

import datasets

_DESCRIPTION = """\
FrenchMedMCQA
"""

_HOMEPAGE = "https://frenchmedmcqa.github.io"

_LICENSE = "Apache License 2.0"

_URL = "https://huggingface.co/datasets/DEFT-2023/DEFT2023/resolve/main/DEFT-2023-FULL.zip"

_CITATION = """\
@InProceedings{FrenchMedMCQA,
  title     = {FrenchMedMCQA : A French Multiple-Choice Question Answering Dataset for Medical domain},
  author    = {Yanis LABRAK, Adrien BAZOGE, Richard DUFOUR, Béatrice DAILLE, Mickael ROUVIER, Emmanuel MORIN and Pierre-Antoine GOURRAUD},
  booktitle = {EMNLP 2022 Workshop - The 13th International Workshop on Health Text Mining and Information Analysis (LOUHI 2022)},
  year      = {2022},
  pdf       = {Coming soon},
  url       = {Coming soon},
  abstract  = {Coming soon}
}
"""

class FrenchMedMCQA(datasets.GeneratorBasedBuilder):
    """FrenchMedMCQA: A French Multiple-Choice Question Answering corpus for the medical domain"""

    VERSION = datasets.Version("1.0.0")

    def _info(self):

        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answer_a": datasets.Value("string"),
                "answer_b": datasets.Value("string"),
                "answer_c": datasets.Value("string"),
                "answer_d": datasets.Value("string"),
                "answer_e": datasets.Value("string"),
                "label": datasets.ClassLabel(
                    names=self.generate_label_names()
                ),
                "correct_answers": datasets.Sequence(
                    datasets.Value("string")
                ),
                "bert_text_no_ctx": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        
        data_dir_dl = dl_manager.download_and_extract(_URL).rstrip("/")
        data_dir = self._convert_to_csv(data_dir_dl)
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "train.csv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev.csv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test.csv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        CLS = "<s>"
        BOS = "<s>"
        SEP = "</s>"
        EOS = "</s>"
        BERT_CLS = "[CLS]"
        BERT_BOS = ""
        BERT_SEP = "[SEP]"
        BERT_EOS = ""

        df = pd.read_csv(filepath, sep=";")
        for key, d in df.iterrows():
            sequence_no_ctx = CLS + " " + d["question"] + f" {SEP} " + f" {SEP} ".join([d[f"answers.{letter}"] for letter in ["a","b","c","d","e"]]) + " " + EOS
            bert_no_ctx = sequence_no_ctx.replace(SEP, BERT_SEP).replace(CLS, BERT_CLS).replace(BOS, BERT_BOS).replace(EOS, BERT_EOS)
            yield key, {
                "id": d["id"],
                "question": d["question"],
                "answer_a": d["answers.a"],
                "answer_b": d["answers.b"],
                "answer_c": d["answers.c"],
                "answer_d": d["answers.d"],
                "answer_e": d["answers.e"],
                "correct_answers": d["correct_answers"].split("|"),
                "label": "".join(sorted(d["correct_answers"].split("|"))),
                "bert_text_no_ctx": bert_no_ctx,
            }

    def _convert_to_csv(self, json_data_dir):
        df_columns = [
            "id",
            "question",
            "answers.a",
            "answers.b",
            "answers.c",
            "answers.d",
            "answers.e",
            "correct_answers",
            "nbr_correct_answers"
        ]
        output_path = os.path.join(json_data_dir, "csv")
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        for split in "train", "dev", "test":
            json_path = os.path.join(json_data_dir, split + ".json")
            if os.path.isfile(json_path):
                with open(json_path) as f:
                    data = json.load(f)
                df = pd.DataFrame(
                    [
                        [
                            d["id"],
                            d["question"],
                            d["answers"]["a"],
                            d["answers"]["b"],
                            d["answers"]["c"],
                            d["answers"]["d"],
                            d["answers"]["e"],
                            "|".join(d["correct_answers"]),
                            str(d["nbr_correct_answers"])
                        ] for d in data
                    ],
                    columns=df_columns
                )
                df.to_csv(os.path.join(output_path, split + ".csv"), sep=';', index=False)
        return output_path

    @staticmethod
    def generate_label_names():
        letters = "abcde"
        combination_lists = []
        for i in range(len(letters)):
            combination_lists.append(["".join(c) for c in combinations(letters, i + 1)])
        return list(chain(*combination_lists))
