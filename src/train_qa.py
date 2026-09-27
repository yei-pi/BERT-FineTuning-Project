import os
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    set_seed,
)

# ============================================================
# Configuration
# ============================================================

SEED = 42
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128
TRAIN_SAMPLES = 15000

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

set_seed(SEED)

# ============================================================
# Load SQuAD v1.1
# ============================================================

dataset = load_dataset("rajpurkar/squad")

train_dataset = dataset["train"].shuffle(seed=SEED).select(
    range(TRAIN_SAMPLES)
)

validation_dataset = dataset["validation"]

# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def prepare_features(examples):
    questions = [q.strip() for q in examples["question"]]

    tokenized = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized.sequence_ids(i)
        sample_index = sample_mapping[i]

        answer = examples["answers"][sample_index]

        if len(answer["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        token_start_index = 0

        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1

        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if (
            offsets[token_start_index][0] > start_char
            or offsets[token_end_index][1] < end_char
        ):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            while (
                token_start_index < len(offsets)
                and offsets[token_start_index][0] <= start_char
            ):
                token_start_index += 1

            start_positions.append(token_start_index - 1)

            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1

            end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized


train_features = train_dataset.map(
    prepare_features,
    batched=True,
    remove_columns=train_dataset.column_names,
)

validation_features = validation_dataset.map(
    prepare_features,
    batched=True,
    remove_columns=validation_dataset.column_names,
)

# ============================================================
# Full Fine-Tuning
# ============================================================

full_model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_NAME
)

full_training_args = TrainingArguments(
    output_dir="checkpoints/squad_full",
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
    train_dataset=train_features,
    eval_dataset=validation_features,
)

full_trainer.train()

# ============================================================
# Feature-Based
# ============================================================

feature_model = AutoModelForQuestionAnswering.from_pretrained(
    MODEL_NAME
)

# Freeze BERT parameters
for param in feature_model.base_model.parameters():
    param.requires_grad = False

feature_training_args = TrainingArguments(
    output_dir="checkpoints/squad_feature",
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
    train_dataset=train_features,
    eval_dataset=validation_features,
)

feature_trainer.train()

print("SQuAD training completed.")
