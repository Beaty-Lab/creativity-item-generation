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
from scipy.stats import spearmanr, pearsonr
import json as js
from transformers import (
    TrainerState,
    TrainerControl,
    TrainerCallback,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from pathlib import Path
home = Path.home()

def predict_with_model(
    trained_model_dir: str,
    item_responses: str,
    prediction_name: str,
    save_file: str,
    round: int,
):
    """
    This function predicts using a trained model and saves the predictions to a specified file.

    Parameters:
    - trained_model_dir (str): The directory of the trained model.
    - item_responses (str): The item responses file to predict on.
    - prediction_name (str): The name of the prediction column.
    - save_file (str): The file to save the predictions.
    - round (int): The round number.

    Returns:
    - None
    """
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
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(trained_model_dir)

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
        output_dir=f"{home}/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model_evaluation",
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