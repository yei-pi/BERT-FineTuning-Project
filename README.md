# BERT Fine-Tuning for NLP Tasks

This project compares two approaches for using BERT in four Natural Language Processing (NLP) tasks:

* AG News text classification
* Named Entity Recognition (NER)
* Part-of-Speech (POS) tagging
* Extractive Question Answering (SQuAD)

For each task, two training methods were evaluated:

1. **Full Fine-Tuning:** all BERT parameters are trainable.
2. **Feature-Based:** the pretrained BERT parameters are frozen and only the task-specific classification head is trained.

## 1. Base Model

The experiments use:

`bert-base-uncased`

The same base model was used for all four tasks.

A fixed random seed of **42** was used to improve reproducibility.

## 2. Datasets

### AG News

Dataset:

`fancyzhx/ag_news`

This dataset contains four news categories:

* World
* Sports
* Business
* Sci/Tech

Maximum sequence length: 128 tokens.

### CoNLL-2003

Dataset:

`lhoestq/conll2003`

The task is token-level Named Entity Recognition with 9 labels.

Maximum sequence length: 128 tokens.

The labels are:

```text
O
B-PER
I-PER
B-ORG
I-ORG
B-LOC
I-LOC
B-MISC
I-MISC
```

### UD English EWT

Dataset:

`universal-dependencies/universal_dependencies`

Configuration:

`en_ewt`

Revision:

`2.18`

The task uses 17 Universal POS tags.

Maximum sequence length: 128 tokens.

### SQuAD v1.1

Dataset:

`rajpurkar/squad`

The task is extractive question answering, where the model predicts the start and end positions of the answer in the context.

For this experiment, a training subset of **15,000 examples** was used.

Maximum sequence length: 384 tokens.

Document stride: 128 tokens.

## 3. Training Methods

### Full Fine-Tuning

In Full Fine-Tuning, the pretrained BERT parameters are updated during training together with the task-specific head.

Learning rate:

`2e-5`

### Feature-Based

In Feature-Based training, the BERT parameters remain frozen.

Only the task-specific classification or QA head is trained.

Learning rate:

`1e-3`

## 4. Common Training Configuration

The main settings used in the experiments were:

| Parameter                      | Value             |
| ------------------------------ | ----------------- |
| Base model                     | bert-base-uncased |
| Random seed                    | 42                |
| Epochs                         | 3                 |
| Batch size                     | 16                |
| Weight decay                   | 0.01              |
| Full Fine-Tuning learning rate | 2e-5              |
| Feature-Based learning rate    | 1e-3              |
| AG News max length             | 128               |
| NER max length                 | 128               |
| POS max length                 | 128               |
| SQuAD max length               | 384               |
| SQuAD document stride          | 128               |

## 5. Installation

The project uses Python and the following main libraries:

* PyTorch
* Transformers
* Datasets
* Evaluate
* Accelerate
* Hugging Face Hub

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## 6. Reproducing the Experiments

The experiments were developed and trained using Google Colab with GPU acceleration.

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/BERT-FineTuning-Project.git
cd BERT-FineTuning-Project
```

Replace `YOUR_USERNAME` with the GitHub username that owns this repository.

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set the random seed

The experiments use seed 42.

The training scripts should initialize the seed before loading and training the models:

```python
SEED = 42

import os
import random
import numpy as np
import torch
from transformers import set_seed

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

set_seed(SEED)
```

### Step 4: Run the task notebooks

The notebooks contain the complete preprocessing and training workflow for each task:

```text
notebooks/
├── 01_AG_News.ipynb
├── 02_NER_CoNLL2003.ipynb
├── 03_POS_UD_EWT.ipynb
└── 04_SQuAD.ipynb
```

Each notebook contains the dataset loading, tokenization, label alignment when required, model configuration, training and evaluation steps.

### Step 5: Choose the training method

For Full Fine-Tuning, all BERT parameters are trainable.

For Feature-Based training, the BERT encoder is frozen and only the task-specific head is trained.

## 7. Results

The final evaluation results were:

| Task    | Method           | Accuracy | Precision | Recall |     F1 | Exact Match |
| ------- | ---------------- | -------: | --------: | -----: | -----: | ----------: |
| AG News | Full Fine-Tuning |   94.80% |         — |      — | 94.80% |           — |
| AG News | Feature-Based    |   84.74% |         — |      — | 84.70% |           — |
| NER     | Full Fine-Tuning |   98.84% |    98.84% | 98.84% | 98.84% |           — |
| NER     | Feature-Based    |   97.02% |    96.87% | 97.02% | 96.89% |           — |
| POS     | Full Fine-Tuning |   97.23% |         — |      — |      — |           — |
| POS     | Feature-Based    |   93.06% |         — |      — |      — |           — |
| SQuAD   | Full Fine-Tuning |        — |         — |      — | 81.75% |      72.54% |
| SQuAD   | Feature-Based    |        — |         — |      — | 24.23% |      15.38% |

## 8. Training and Validation Loss

| Task    | Method           | Training Loss | Validation Loss |
| ------- | ---------------- | ------------: | --------------: |
| AG News | Full Fine-Tuning |      0.093669 |        0.234656 |
| AG News | Feature-Based    |      0.487442 |        0.432036 |
| NER     | Full Fine-Tuning |      0.017277 |        0.047221 |
| NER     | Feature-Based    |      0.126230 |        0.118002 |
| POS     | Full Fine-Tuning |      0.042513 |        0.115995 |
| POS     | Feature-Based    |      0.260443 |        0.242253 |
| SQuAD   | Full Fine-Tuning |  Not reported |    Not reported |
| SQuAD   | Feature-Based    |  Not reported |    Not reported |

## 9. Hugging Face Models

The trained models were uploaded to the Hugging Face Hub under the account `yei-pi`.

### AG News

* Full Fine-Tuning: https://huggingface.co/yei-pi/agnews-full
* Feature-Based: https://huggingface.co/yei-pi/agnews-feature

### NER

* Feature-Based: https://huggingface.co/yei-pi/ner-feature

The NER Full Fine-Tuning model obtained evaluation results, but its final model weights were not successfully preserved. Therefore, it is not included as a functional model repository.

### POS

* Full Fine-Tuning: https://huggingface.co/yei-pi/pos-full
* Feature-Based: https://huggingface.co/yei-pi/pos-feature

### SQuAD

* Full Fine-Tuning: https://huggingface.co/yei-pi/squad-full
* Feature-Based: https://huggingface.co/yei-pi/squad-feature

## 10. Limitations

The NER evaluation used token-level weighted precision, recall and F1 instead of the standard entity-level SeqEval evaluation.

The NER Full Fine-Tuning model weights were not successfully preserved, so the model could not be uploaded as a functional Hugging Face model.

The SQuAD experiment used only 15,000 training examples instead of the complete training dataset to reduce training time and computational requirements.

Only one fixed seed and one main set of hyperparameters were used. No extensive hyperparameter search was performed.

The experiments were trained using Google Colab, so computational resources and training time depended on the available GPU and session limits.

## 11. Project Structure

```text
BERT-FineTuning-Project/
│
├── README.md
├── requirements.txt
│
├── config/
│   └── training_config.json
│
├── notebooks/
│   ├── 01_AG_News.ipynb
│   ├── 02_NER_CoNLL2003.ipynb
│   ├── 03_POS_UD_EWT.ipynb
│   └── 04_SQuAD.ipynb
│
├── src/
│   ├── train_agnews.py
│   ├── train_ner.py
│   ├── train_pos.py
│   └── train_qa.py
│
└── results/
    ├── agnews_results.json
    ├── ner_results.json
    ├── pos_results.json
    └── squad_results.json
```

## 12. Main Files

### `requirements.txt`

Contains the Python packages required to run the project.

### `training_config.json`

Contains the main datasets, model, seed, sequence lengths and learning rates used in the experiments.

### `notebooks/`

Contains the interactive Google Colab/Jupyter notebooks used for the four NLP tasks.

### `src/`

Contains the training scripts organized by task.

### `results/`

Contains the evaluation results obtained from the experiments.

## 13. Conclusion

This project compares BERT Full Fine-Tuning with Feature-Based training across four NLP tasks.

The experiments show that the two approaches produce different performance levels depending on the task. Full Fine-Tuning obtained higher evaluation scores in all four experiments, while Feature-Based training provided a way to use BERT representations while keeping the pretrained encoder frozen.

The project also demonstrates how the same pretrained BERT model can be adapted to classification, token classification and extractive question answering tasks.
