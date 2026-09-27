import os
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ============================================================
# Configuration
# ============================================================

SEED = 42
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 4
MAX_LENGTH = 128

# Reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

set_seed(SEED)

# ============================================================
# Load dataset
# ============================================================

dataset = load_dataset("fancyzhx/ag_news")

label_names = ["World", "Sports", "Business", "Sci/Tech"]

# ============================================================
# Tokenization
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
)

tokenized_dataset = tokenized_dataset.rename_column(
    "label",
    "labels",
)

tokenized_dataset.set_format("torch")

# ============================================================
# Full Fine-Tuning
# ============================================================

full_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label={i: label for i, label in enumerate(label_names)},
    label2id={label: i for i, label in enumerate(label_names)},
)

full_training_args = TrainingArguments(
    output_dir="checkpoints/agnews_full",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=SEED,
    data_seed=SEED,
    report_to="none",
)

full_trainer = Trainer(
    model=full_model,
    args=full_training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

full_trainer.train()

# ============================================================
# Feature-Based
# ============================================================

feature_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label={i: label for i, label in enumerate(label_names)},
    label2id={label: i for i, label in enumerate(label_names)},
)

# Freeze BERT parameters
for param in feature_model.base_model.parameters():
    param.requires_grad = False

feature_training_args = TrainingArguments(
    output_dir="checkpoints/agnews_feature",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=SEED,
    data_seed=SEED,
    report_to="none",
)

feature_trainer = Trainer(
    model=feature_model,
    args=feature_training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

feature_trainer.train()

print("AG News training completed.")
