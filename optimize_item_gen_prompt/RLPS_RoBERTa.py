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
from accelerate import Accelerator
import time
import transformers
from argparse import ArgumentParser
from peft import LoraConfig, TaskType, get_peft_model


# Replicate Simones Auto Scorer
def train_model():
    run = wandb.init(project="retrain-scoring-model")
    os.environ["WANDB_WATCH"]="true"

    # for distributed training
    accelerator = Accelerator()
    # TODO: put these params in wandb
    peft_config = LoraConfig(peft_type=TaskType.SEQ_CLS, inference_mode=False, r=wandb.config.lora_r, lora_alpha=wandb.config.lora_alpha, lora_dropout=wandb.config.lora_dropout)

    d = pd.read_csv(
        "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/data/CPSfinalMeanScore.csv"
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
        return mse

    # RETRAIN

    training_args = TrainingArguments(
        output_dir="/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model",
        report_to="wandb",
        learning_rate=wandb.config.learning_rate,
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


if __name__ == "__main__":
    config= {
        "parameters": {
            "model_name": {"values":["roberta-base"]},
            "epochs": {"values": [25, 50, 75, 100, 125]},
            "lr": {"max": 0.1, "min": 0.00001},
            "batch_size": {"values": [8, 16, 32]},
            "lora_r": {"values": [8]},
            "lora_alpha": {"values": [32]},
            "lora_dropout": {"values": [0.1]},
            "metric": {"values": "originality"},
        }
    }
    sweep_id = wandb.sweep(sweep=config, project="retrain-scoring-model")
    wandb.agent(sweep_id, function=train_model, count=10)

    # parser = ArgumentParser()
    # parser.add_argument("--task", type=str)
    # parser.add_argument("--trained_model_dir", type=str)
    # parser.add_argument("--item_responses", type=str)
    # parser.add_argument("--prediction", type=str)
    # parser = parser.parse_args()
    # if parser.task == "train":
    #     train_model()
    # elif parser.task == "evaluate":
    #     evaluate_model(
    #         parser.trained_model_dir, parser.item_responses, parser.prediction
    #     )
    # else:
    #     print("A task needs to be specified!")
