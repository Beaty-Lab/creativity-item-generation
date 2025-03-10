"""
    Parameter-efficient finetuning of scorers. Including LoRA, QLoRA, AdaLoRA orthogonal finetuning, and prefix-tuning and p-tuning
    Use LLama-3, Gemma, and Flan-T5.
    Add OFT/BOFT if time permitting and performance justifies it
"""

import numpy as np
import pandas as pd
import torch
import os
import wandb
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from accelerate import Accelerator
import transformers
from scipy.stats import spearmanr
from peft import get_peft_model, PeftConfig, PeftType, PromptEncoderConfig
from transformers import (
    TrainerState,
    TrainerControl,
    TrainerCallback,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from os.path import join
from peft import LoraConfig, TaskType


# TODO: is this used anywhere?
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


class PeftModel:
    def __init__(self, config: dict, peft_config: PeftConfig):
        if config["scorerType"] == "LoRA":
            self.run = wandb.init(
                project=config["WandbProjectName"],
            )
            if config["use_sweep"]:
                self.config = config
                peft_config = LoraConfig(
                    r=wandb.config.lora_r,
                    lora_alpha=config["lora_alpha"],
                    lora_dropout=wandb.config.lora_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_CLS,
                    target_modules=wandb.config.lora_target_modules,
                )
                self.config["lr"] = wandb.config.learning_rate
                self.config["batchSize"] = wandb.config.batch_size
                self.config["lora_dropout"] = wandb.config.lora_dropout
                self.config["lora_r"] = wandb.config.lora_r
                self.config["lora_target_modules"] = wandb.config.lora_target_modules
                self.config["epochs"] = wandb.config.num_train_epochs
                self.config["scorerBaseModel"] = wandb.config.scorer_base_model
                model = AutoModelForSequenceClassification.from_pretrained(
                    wandb.config.scorer_base_model, num_labels=1, device_map="auto"
                )

            else:
                self.config = config
                model = AutoModelForSequenceClassification.from_pretrained(
                    config["scorerBaseModel"], num_labels=1, device_map="auto"
                )


            if "llama" in config["scorerBaseModel"]:
                model.config.pad_token_id = model.config.eos_token_id

            self.peft_config = peft_config
            self.peft_model = get_peft_model(model, peft_config)
            wandb.watch(self.peft_model)
            self.tokenizer = AutoTokenizer.from_pretrained(
                peft_config.base_model_name_or_path
            )

            if "llama" in config["scorerBaseModel"]:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.pad_token = self.tokenizer.eos_token
        elif config["scorerType"] == "P-tuning":
            self.run = wandb.init(
                project=config["WandbProjectName"],
            )
            if config["use_sweep"]:
                self.config = config
                peft_config = PromptEncoderConfig(
                    task_type="SEQ_CLS",
                    num_virtual_tokens=wandb.config.num_virtual_tokens,
                    encoder_hidden_size=wandb.config.encoder_hidden_size,
                )
                self.config["numVirtualTokens"] = wandb.config.num_virtual_tokens
                self.config["encoderHiddenSize"] = wandb.config.encoder_hidden_size
                self.config["lr"] = wandb.config.learning_rate
                self.config["batchSize"] = wandb.config.batch_size
                self.config["epochs"] = wandb.config.num_train_epochs
                self.config["scorerBaseModel"] = wandb.config.scorer_base_model
                model = AutoModelForSequenceClassification.from_pretrained(
                    wandb.config.scorer_base_model, num_labels=1, device_map="auto"
                )

            else:
                self.config = config
                model = AutoModelForSequenceClassification.from_pretrained(
                    config["scorerBaseModel"], num_labels=1, device_map="auto"
                )

            if "llama" in config["scorerBaseModel"]:
                model.config.pad_token_id = model.config.eos_token_id

            self.peft_config = peft_config
            self.peft_model = get_peft_model(model, peft_config)
            wandb.watch(self.peft_model)
            self.tokenizer = AutoTokenizer.from_pretrained(
                peft_config.base_model_name_or_path
            )

            if "llama" in config["scorerBaseModel"]:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize_function(self, examples: Dataset):
        return self.tokenizer(examples["text"], truncation=True, padding=True)

    @staticmethod
    def compute_metrics(eval_preds):
        predictions, references = eval_preds
        correlation = spearmanr(predictions, references)[0]
        wandb.log({"Spearman Correlation": correlation})
        return {"Spearman Correlation": correlation}

    def fit(self):
        # for distributed training
        accelerator = Accelerator()

        d = pd.read_csv(
            self.config["scorerDataPath"],
        ).sample(frac=1)

        np.random.seed(
            self.config["random_seed"]
        )  # sets a randomization seed for reproducibility
        transformers.set_seed(self.config["random_seed"])

        # SET UP DATASET
        d["inputs"] = d[self.config["scorerInputColumn"]]
        d["text"] = d["inputs"]
        d["label"] = d[self.config["scorerLabelColumn"]]

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

        tokenized_datasets = train_val_test_dataset.map(
            self.tokenize_function, batched=True
        )  # applies wrapper to our dataset

        training_args = TrainingArguments(
            # TODO: append wandb id to output dir
            output_dir=join(self.config["scorerOutputDir"], self.config["scorerType"]),
            report_to="wandb",
            learning_rate=self.config["lr"],
            num_train_epochs=self.config["epochs"],
            per_device_train_batch_size=self.config["batchSize"],
            per_device_eval_batch_size=self.config["batchSize"],
            disable_tqdm=False,
            load_best_model_at_end=False,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            save_total_limit=1,
            fp16=self.config["use_fp16"],
        )

        trainer = accelerator.prepare(
            Trainer(
                model=self.peft_model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"],
                compute_metrics=self.compute_metrics,
                tokenizer=self.tokenizer,
            )
        )

        trainer.train()
        prediction = trainer.predict(tokenized_datasets["test"])
        wandb.finish()

    def predict(self, prediction_name: str, output_file_name: str, round: int):
        accelerator = Accelerator()
        np.random.seed(
            self.config["random_seed"]
        )  # sets a randomization seed for reproducibility
        transformers.set_seed(self.config["random_seed"])

        # use the trained autoscorer to get results on new item responses# item_responses and save_file should point to the same file
        # we load twice so we can save without losing any columns
        d = pd.read_json(self.config["scorerDataPath"])

        save_file = pd.read_json(self.config["scorerDataPath"])
        d["text"] = d[f"creative_response_round_{round}"]
        d_input = d.filter(["text"], axis=1)
        dataset = Dataset.from_pandas(
            d_input, preserve_index=False
        )  # Turns pandas data into huggingface/pytorch dataset
        tokenized_datasets = dataset.map(
            self.tokenize_function, batched=True
        )  # applies wrapper to our dataset

        training_args = TrainingArguments(
            output_dir=self.config["scorerOutputDir"],
            learning_rate=self.config["lr"],
            num_train_epochs=self.config["epochs"],
            per_device_train_batch_size=self.config["batchSize"],
            per_device_eval_batch_size=self.config["batchSize"],
            disable_tqdm=False,
            load_best_model_at_end=False,
            save_strategy="no",
            evaluation_strategy="no",
            save_total_limit=1,
        )

        trainer = accelerator.prepare(
            Trainer(
                model=self.peft_model,
                args=training_args,
                eval_dataset=tokenized_datasets,
                compute_metrics=self.compute_metrics,
                tokenizer=self.tokenizer,
            )
        )

        prediction = trainer.predict(tokenized_datasets)
        test_data = {
            "text": tokenized_datasets["text"],
            f"{prediction_name}": np.squeeze(prediction.predictions),
        }
        save_file[prediction_name] = test_data[prediction_name]
        save_file.to_json(f"{output_file_name}_round_{round}.json")
