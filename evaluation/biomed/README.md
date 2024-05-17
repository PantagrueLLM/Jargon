# Jargon Biomedical Benchmarking ⚕️

This folder contains the experimental recipes for fine-tuning Jargon (or any other BERT-style model from the HuggingFace hub) on the biomedical benchmark from our [paper](https://hal.science/hal-04535557/file/FB2_domaines_specialises_LREC_COLING24.pdf).

### Setup

First of all, assuming you have already installed `Jargon/requirements.txt` in your Python environment, you'll need to install the additional dependencies in this directory:
```bash
pip install -r requirements.txt
```
Once these are installed, run any of `token_classification.py`, `frenchmedmcqa/train.py` or `clister.py` with the `-h` flag for details on how to execute the respective tasks.

### Datasets

- [FrenchMedMCQA](https://huggingface.co/datasets/qanastek/frenchmedmcqa): can be downloaded directly from the HuggingFace hub using `frenchmedmcqa/train.py`
- MCQ: sequence classification task built on[Labforsims](https://aclanthology.org/2020.lrec-1.72/)
- CAS-POS/ESSAI-POS: medical part-of-speech tagging task: datasets available [here](https://clementdalloux.fr/?page_id=28)
- CAS-SG: semantic token classification task built on the [CAS](https://aclanthology.org/W18-5614/) corpus of clinical cases
- MEDLINE/EMEA: token-classification tasks based on the [QUAERO](https://quaerofrenchmed.limsi.fr/) annotated medical dataset: downloaded directly from the HuggingFace hub by `token_classification.py`
- CLISTER: semantic textual similarity task, also built on [CAS](https://aclanthology.org/W18-5614/).

<!-- ### Experiments

To reproduce the results presented in the paper,  -->

