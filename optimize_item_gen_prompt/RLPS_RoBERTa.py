#  IMPORT PACKAGES
import evaluate
import io
import numpy as np
import pandas as pd
import torch
import os
import wandb
from datasets import Dataset, load_metric, DatasetDict
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers.trainer_utils import IntervalStrategy
from accelerate import Accelerator
import time
import transformers
from argparse import ArgumentParser
from peft import LoraConfig, TaskType, get_peft_model


# training done with wandb sweep for grid search
def train_model():
    run = wandb.init(project="retrain-scoring-model")

    # for distributed training
    accelerator = Accelerator()
    # TODO: put these params in wandb
    peft_config = LoraConfig(
        peft_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=wandb.config.lora_r,
        lora_alpha=wandb.config.lora_alpha,
        lora_dropout=wandb.config.lora_dropout,
    )

    d = pd.read_csv(
        "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/data/CPSTfulldataset3.csv"
    )

    # prefix = "A creative solution for the situation: " # we'll use prefix/conn to construct inputs to the model
    # suffix = "is: " # we'll use prefix/conn to construct inputs to the model

    scaler = StandardScaler()
    np.random.seed(40)  # sets a randomization seed for reproducibility
    transformers.set_seed(40)

    # SET UP DATASET
    d["inputs"] = d["SolutionsWithProblem"]
    d["text"] = d["inputs"]

    if wandb.config.metric == "originality":
        d["label"] = d["FacScoresO"]
    elif wandb.config.metric == "quality":
        d["label"] = d["FacScoresQ"]

    d_input = d.filter(["text", "label", "set"], axis=1)

    #  CREATE TRAIN/TEST SPLIT
    dataset = Dataset.from_pandas(
        d_input, preserve_index=False
    )  # Turns pandas data into huggingface/pytorch dataset
    train_val_test_dataset = DatasetDict(
        {
            "train": dataset.filter(lambda example: example["set"] == "training"),
            "test": dataset.filter(lambda example: example["set"] == "test"),
            "heldout": dataset.filter(lambda example: example["set"] == "heldout"),
        }
    )

    train_val_test_dataset = train_val_test_dataset.remove_columns("set")

    print(train_val_test_dataset)  # show the dataset dictionary
    print(train_val_test_dataset["train"].features)
    time.sleep(10)

    # SET UP MODEL & TOKENIZER
    model = AutoModelForSequenceClassification.from_pretrained(
        wandb.config.model_name, num_labels=1
    )  # labels = 1 is for regression

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(
        wandb.config.model_name
    )  # ...some settings in the tokenizer call

    wandb.watch(model)

    #  DEFINE WRAPPER TOKENIZER FUNCTION (FOR BATCH TRAINING)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_datasets = train_val_test_dataset.map(
        tokenize_function, batched=True
    )  # applies wrapper to our dataset

    #  DEFINE LOSS METRIC (ROOT MEAN SQUARED ERROR [rmse])
    def compute_metrics(eval_preds):
        predictions, references = eval_preds
        mse_metric = evaluate.load("mse")
        mse = mse_metric.compute(predictions=predictions, references=references)
        wandb.log({"eval_mse": mse})
        return mse

    # RETRAIN
    print(wandb.config)

    training_args = TrainingArguments(
        output_dir="/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model",
        report_to="wandb",
        learning_rate=wandb.config.lr,
        num_train_epochs=wandb.config.epochs,
        per_device_train_batch_size=wandb.config.batch_size,
        per_device_eval_batch_size=wandb.config.batch_size,
        disable_tqdm=False,
        load_best_model_at_end=False,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=1,
    )

    trainer = accelerator.prepare(
        Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["heldout"],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
    )

    result = trainer.train()
    trainer.evaluate()
    # wandb.finish()


def train_model_no_sweep():

    config = {
        "batch_size": 16,
        "epochs": 125,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "lora_r": 8,
        "lr": 0.001,
        "metric": "originality",
        "model_name": "roberta-base"
    }

    # for distributed training
    accelerator = Accelerator()
    # TODO: put these params in wandb
    peft_config = LoraConfig(
        peft_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
    )

    d = pd.read_csv(
        "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/data/CPSTfulldataset3.csv"
    )

    # prefix = "A creative solution for the situation: " # we'll use prefix/conn to construct inputs to the model
    # suffix = "is: " # we'll use prefix/conn to construct inputs to the model

    scaler = StandardScaler()
    np.random.seed(40)  # sets a randomization seed for reproducibility
    transformers.set_seed(40)

    # SET UP DATASET
    d["inputs"] = d["SolutionsWithProblem"]
    d["text"] = d["inputs"]

    if config["metric"] == "originality":
        d["label"] = d["FacScoresO"]
    elif config["metric"] == "quality":
        d["label"] = d["FacScoresQ"]

    d_input = d.filter(["text", "label", "set"], axis=1)

    #  CREATE TRAIN/TEST SPLIT
    dataset = Dataset.from_pandas(
        d_input, preserve_index=False
    )  # Turns pandas data into huggingface/pytorch dataset
    train_val_test_dataset = DatasetDict(
        {
            "train": dataset.filter(lambda example: example["set"] == "training"),
            "test": dataset.filter(lambda example: example["set"] == "test"),
            "heldout": dataset.filter(lambda example: example["set"] == "heldout"),
        }
    )

    train_val_test_dataset = train_val_test_dataset.remove_columns("set")

    print(train_val_test_dataset)  # show the dataset dictionary
    print(train_val_test_dataset["train"].features)
    time.sleep(5)

    # SET UP MODEL & TOKENIZER
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"], num_labels=1
    )  # labels = 1 is for regression

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"]
    )  # ...some settings in the tokenizer call

    #  DEFINE WRAPPER TOKENIZER FUNCTION (FOR BATCH TRAINING)
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_datasets = train_val_test_dataset.map(
        tokenize_function, batched=True
    )  # applies wrapper to our dataset

    #  DEFINE LOSS METRIC (ROOT MEAN SQUARED ERROR [rmse])
    def compute_metrics(eval_preds):
        predictions, references = eval_preds
        mse_metric = evaluate.load("mse")
        return mse_metric.compute(predictions=predictions, references=references)

    training_args = TrainingArguments(
        output_dir="/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model_evaluation",
        learning_rate=config["lr"],
        num_train_epochs=config["epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        disable_tqdm=False,
        load_best_model_at_end=False,
        save_strategy=IntervalStrategy.EPOCH,
        do_eval=False,
        save_total_limit=1,
        report_to="none",
    )

    trainer = accelerator.prepare(
        Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["heldout"],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
    )


    result = trainer.train()
    prediction = trainer.predict(tokenized_datasets["heldout"])
    mse_metric = evaluate.load("mse")
    print(mse_metric.compute(predictions=prediction.predictions[1], references=tokenized_datasets["heldout"]["label"]))


# use the trained autoscorer to get results on new item responses
# make sure the prediction metric is the same as the model used to evaluate
# TODO: update with peft
def evaluate_model(
    trained_model_dir: str,
    item_responses: str,
    prediction_name: str,
    save_file: str,
    round: int,
):
    accelerator = Accelerator()
    np.random.seed(40)  # sets a randomization seed for reproducibility
    transformers.set_seed(40)

    # item_responses and save_file should point to the same file
    # we load twice so we can save without losing any columns
    d = pd.read_json(f"{item_responses}_round_{round}.json")
    # d.dropna(inplace=True)

    save_file = pd.read_json(f"{save_file}_round_{round}.json")
    # save_file.dropna(inplace=True)

    d["text"] = d[f"creative_response_round_{round}"]
    d_input = d.filter(["text"], axis=1)
    dataset = Dataset.from_pandas(
        d_input, preserve_index=False
    )  # Turns pandas data into huggingface/pytorch dataset

    model = AutoModelForSequenceClassification.from_pretrained(
        trained_model_dir, num_labels=1
    )  # TONS of settings in the model call, but labels = 1
    tokenizer = AutoTokenizer.from_pretrained(
        trained_model_dir
    )  # ...some settings in the tokenizer call

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True
    )  # applies wrapper to our dataset

    #  DEFINE LOSS METRIC (ROOT MEAN SQUARED ERROR [rmse])
    def compute_metrics(eval_preds):
        predictions, references = eval_preds
        mse_metric = evaluate.load("mse")
        mse = mse_metric.compute(predictions=predictions, references=references)
        return mse

    training_args = TrainingArguments(
        output_dir="/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model_evaluation",
        learning_rate=0.00005,
        num_train_epochs=116,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        disable_tqdm=False,
        load_best_model_at_end=False,
        save_strategy="no",
        evaluation_strategy="no",
        eval_steps=500,
        save_total_limit=1,
    )

    trainer = accelerator.prepare(
        Trainer(
            model=model,
            args=training_args,
            eval_dataset=tokenized_datasets,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
    )

    prediction = trainer.predict(tokenized_datasets)
    test_data = {
        "text": tokenized_datasets["text"],
        f"{prediction_name}": np.squeeze(prediction.predictions),
    }
    save_file[f"{prediction_name}_round_{round}"] = test_data[prediction_name]
    save_file.to_json(f"{item_responses}_round_{round}.json")
    # dataset_test_df = pd.DataFrame(test_data)
    # dataset_test_df.to_json(item_responses)


# TODO: don't want to have to change this manually
if __name__ == "__main__":
    train_model_no_sweep()
