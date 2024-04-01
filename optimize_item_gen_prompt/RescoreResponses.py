# Given an originality scoring model, rescore all the CPIG items using this model
# store the rescored responses in a csv
import evaluate
import os
import numpy as np
import torch
import pandas as pd
from os import listdir
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import AutoPeftModel
from accelerate import Accelerator
from os.path import join
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from tqdm import tqdm
from transformers import BitsAndBytesConfig

RESPONSE_DIR = "/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/with_controversial_filter_few_shot"
SCORING_MODEL = "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model/llama-7b-peft-quantized-originality"

quantized_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
model = AutoPeftModel.from_pretrained(SCORING_MODEL, num_labels=1, quantization_config=quantized_config, device_map="auto")
model.config.pad_token_id = model.config.eos_token_id
tokenizer = AutoTokenizer.from_pretrained(SCORING_MODEL)


def predict_with_model(
    item_responses: str,
    prediction_name: str,
    save_file: str,
    round: int,
):
    accelerator = Accelerator()

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
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
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
    save_file.to_json(f"{item_responses}_round_{round}_rescored.json")


if __name__ == "__main__":
    dirs = [
        d
        for d in os.listdir(RESPONSE_DIR)
        if os.path.isdir(os.path.join(RESPONSE_DIR, d))
    ]
    for d in tqdm(dirs):
        files = os.listdir(join(RESPONSE_DIR, d))
        if (
            "config.json" not in files
            or "items.json" not in files
            or "item_responses_round_0.json" not in files
            or "item_responses_round_4.json" not in files
        ):
            continue
        config = pd.read_json(join(RESPONSE_DIR, d, "config.json"), typ="series")
        item_responses_round_0 = pd.read_json(
            join(RESPONSE_DIR, d, "item_responses_round_0.json")
        )
        item_responses_round_4 = pd.read_json(
            join(RESPONSE_DIR, d, "item_responses_round_4.json")
        )
        if len(item_responses_round_0) == 0 or len(item_responses_round_4) == 0:
            continue

        predict_with_model(
            join(RESPONSE_DIR, d, "item_responses"),
            "originality",
            join(RESPONSE_DIR, d, "item_responses"),
            0,
        )
        predict_with_model(
            join(RESPONSE_DIR, d, "item_responses"),
            "originality",
            join(RESPONSE_DIR, d, "item_responses"),
            4,
        )
