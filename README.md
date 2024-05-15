# Jargon ⚖️  ⚕️  🗣️

[Jargon](https://hal.science/hal-04535557/file/FB2_domaines_specialises_LREC_COLING24.pdf) is an efficient transformer encoder LM for French, combining the LinFormer attention mechanism with the RoBERTa model architecture.

Jargon is available in several versions with different context sizes and types of pre-training corpora.

For more info please check out our [paper](https://hal.science/hal-04535557/file/FB2_domaines_specialises_LREC_COLING24.pdf), accepted for publication at [LREC-COLING 2024](https://lrec-coling-2024.org/list-of-accepted-papers/).


### Pre-trained models

These models were developed in the context of a research project exploring NLP application domains with hightly specialized vocabulary and semantic/syntactic patterns, namely medical science and law.

| **Model**                                                                           | **Initialised from...** |**Training Data**|
|-------------------------------------------------------------------------------------|:-----------------------:|:----------------:|
| [jargon-general-base](https://huggingface.co/PantagrueLLM/jargon-general-base)        |         scratch         |8.5GB Web Corpus|
| [jargon-general-biomed](https://huggingface.co/PantagrueLLM/jargon-general-biomed)    |   jargon-general-base   |5.4GB Medical Corpus|
| jargon-general-legal                                                                |   jargon-general-base   |18GB Legal Corpus
| [jargon-multidomain-base](https://huggingface.co/PantagrueLLM/jargon-multidomain-base) |   jargon-general-base   |Medical+Legal Corpora|
| jargon-legal                                                                        |         scratch         |18GB Legal Corpus|
| [jargon-legal-4096](https://huggingface.co/PantagrueLLM/jargon-legal-4096)   |         scratch         |18GB Legal Corpus|
| [jargon-biomed](https://huggingface.co/PantagrueLLM/jargon-biomed)                    |         scratch         |5.4GB Medical Corpus|
| [jargon-biomed-4096](https://huggingface.co/PantagrueLLM/jargon-biomed-4096)          |         scratch         |5.4GB Medical Corpus|
| [jargon-NACHOS](https://huggingface.co/PantagrueLLM/jargon-NACHOS)                    |         scratch         |[NACHOS](https://drbert.univ-avignon.fr/)|
| [jargon-NACHOS-4096](https://huggingface.co/PantagrueLLM/jargon-NACHOS-4096)        |         scratch         |[NACHOS](https://drbert.univ-avignon.fr/)|


### Using pre-trained Jargon models with 🤗 transformers

You can get started with any of the above using the code snippet below:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("PantagrueLLM/jargon-general-base", trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained("PantagrueLLM/jargon-general-base", trust_remote_code=True)

jargon_maskfiller = pipeline("fill-mask", model=model, tokenizer=tokenizer)
output = jargon_maskfiller("Il est allé au <mask> hier")
```

### Using this repository

You can use the Jargon model framework locally by cloning this repository and installing the `jargon` package in your local environment:
```bash
git clone https://github.com/PantagrueLLM/Jargon.git
cd Jargon
pip install -f requirements.txt
pip install -e .
```

The following snippet will then get you started with masked-language inference using the Python framework:
```python
from jargon import JargonForMaskedLM
from transformers import AutoTokenizer, pipeline

hf_model_path = "PantagrueLLM/jargon-general-base"
tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
model = JargonForMaskedLM.from_pretrained(hf_model_path)
jargon_maskfiller = pipeline("fill-mask", model=model, tokenizer=tokenizer)
output = jargon_maskfiller("Il va aller au <mask> demain")
```

**Note:** The `fairseq` codebase is, as of the time of writing, not compatible with Python 3.11+ (see this [issue](https://github.com/facebookresearch/fairseq/issues/5012) for more details and potential fixes).


## Citation

If you use the Jargon models and/or code for your own research work, please cite as follows:

```bibtex
@inproceedings{segonne:hal-04535557,
  TITLE = {{Jargon: A Suite of Language Models and Evaluation Tasks for French Specialized Domains}},
  AUTHOR = {Segonne, Vincent and Mannion, Aidan and Alonzo Canul, Laura Cristina and Audibert, Alexandre and Liu, Xingyu and Macaire, C{\'e}cile and Pupier, Adrien and Zhou, Yongxin and Aguiar, Mathilde and Herron, Felix and Norr{\'e}, Magali and Amini, Massih-Reza and Bouillon, Pierrette and Eshkol-Taravella, Iris and Esperan{\c c}a-Rodier, Emmanuelle and Fran{\c c}ois, Thomas and Goeuriot, Lorraine and Goulian, J{\'e}r{\^o}me and Lafourcade, Mathieu and Lecouteux, Benjamin and Portet, Fran{\c c}ois and Ringeval, Fabien and Vandeghinste, Vincent and Coavoux, Maximin and Dinarelli, Marco and Schwab, Didier},
  URL = {https://hal.science/hal-04535557},
  BOOKTITLE = {{LREC-COLING 2024 - Joint International Conference on Computational Linguistics, Language Resources and Evaluation}},
  ADDRESS = {Turin, Italy},
  YEAR = {2024},
  MONTH = May,
  KEYWORDS = {Self-supervised learning ; Pretrained language models ; Evaluation benchmark ; Biomedical document processing ; Legal document processing ; Speech transcription},
  PDF = {https://hal.science/hal-04535557/file/FB2_domaines_specialises_LREC_COLING24.pdf},
  HAL_ID = {hal-04535557},
  HAL_VERSION = {v1},
}
```