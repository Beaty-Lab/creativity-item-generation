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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from accelerate import Accelerator
import time
import transformers
from argparse import ArgumentParser

# os.environ["CUDA_VISIBLE_DEVICES"]="1,2"

# Replicate Simones Auto Scorer
def train_model(metric: str):
  # set the wandb project where this run will be logged
  os.environ["WANDB_PROJECT"]="creativity-item-generation"

  # save your trained model checkpoint to wandb
  os.environ["WANDB_LOG_MODEL"]="true"
  # for distributed training
  accelerator = Accelerator()

  d = pd.read_csv('/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/data/CPSfinalMeanScore.csv')

  model_name = "roberta-base"  # a multilingual transformer model
  #prefix = "A creative solution for the situation: " # we'll use prefix/conn to construct inputs to the model
  #suffix = "is: " # we'll use prefix/conn to construct inputs to the model

  scaler = StandardScaler()
  np.random.seed(40) # sets a randomization seed for reproducibility
  transformers.set_seed(40)


  # SET UP DATASET
  d['inputs'] = d['Solutions']
  d['text'] = d['inputs']
  
  if metric == 'originality':
    d['label'] = d['FacScoresO']
  elif metric == 'quality':
    d['label'] = d['FacScoresQ']

  d_input = d.filter(['text','label', 'set'], axis = 1)


  #  CREATE TRAIN/TEST SPLIT
  dataset = Dataset.from_pandas(d_input, preserve_index = False) # Turns pandas data into huggingface/pytorch dataset
  train_val_test_dataset = DatasetDict({
      'train': dataset.filter(lambda example: example['set'] == 'training'),
      'test': dataset.filter(lambda example: example['set'] == 'test'),
      'heldout': dataset.filter(lambda example: example['set'] == 'heldout')
  })

  train_val_test_dataset = train_val_test_dataset.remove_columns('set')

  print(train_val_test_dataset) # show the dataset dictionary
  print(train_val_test_dataset['train'].features)
  time.sleep(10)


  # SET UP MODEL & TOKENIZER
  model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 1) # TONS of settings in the model call, but labels = 1
  tokenizer = AutoTokenizer.from_pretrained(model_name) # ...some settings in the tokenizer call

  #  DEFINE WRAPPER TOKENIZER FUNCTION (FOR BATCH TRAINING)
  def tokenize_function(examples):
    return tokenizer(examples['text'], truncation = True, padding = True)

  tokenized_datasets = train_val_test_dataset.map(tokenize_function, batched = True) # applies wrapper to our dataset


  #  DEFINE LOSS METRIC (ROOT MEAN SQUARED ERROR [rmse])
  def compute_metrics(eval_preds):
    predictions, references = eval_preds
    mse_metric = evaluate.load("mse")
    mse = mse_metric.compute(predictions = predictions, references = references)
    return mse



  # RETRAIN

  training_args = TrainingArguments(         
      output_dir= "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model",
      report_to="wandb", 
      learning_rate=0.00005,
      num_train_epochs=116,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16, 
      disable_tqdm=False,
      load_best_model_at_end=False,
      save_strategy = 'steps',
      evaluation_strategy = 'steps',
      eval_steps = 5500,
      save_steps = 5500,
      fp16=True,
      save_total_limit = 1)     


  trainer = accelerator.prepare(Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_datasets["train"],
      eval_dataset=tokenized_datasets["test"],
      compute_metrics=compute_metrics,
      tokenizer=tokenizer)
  )

  result = trainer.train() 
  trainer.evaluate()
  wandb.finish()



  # EVALUATION

  prediction = trainer.predict(tokenized_datasets['test'])
  train_prediction = trainer.predict(tokenized_datasets['train'])
  heldout_prediction = trainer.predict(tokenized_datasets['heldout'])


  test_data = {'text':tokenized_datasets['test']['text'],'label':tokenized_datasets['test']['label'],'prediction':np.squeeze(prediction.predictions)}
  train_data = {'text':tokenized_datasets['train']['text'],'label':tokenized_datasets['train']['label'],'prediction':np.squeeze(train_prediction.predictions)}
  heldout_data = {'text':tokenized_datasets['heldout']['text'],'label':tokenized_datasets['heldout']['label'],'prediction':np.squeeze(heldout_prediction.predictions)}


  dataset_test_df = pd.DataFrame(test_data)
  dataset_train_df = pd.DataFrame(train_data)
  dataset_heldout_df = pd.DataFrame(heldout_data)


  dataset_train_df.to_csv('./scoring_model/PredictedTrainSet.csv')
  dataset_heldout_df.to_csv('./scoring_model/PredictedHeldoutSet.csv')


# use the trained autoscorer to get results on new item responses
# make sure the prediction metric is the same as the model used to evaluate
def evaluate_model(trained_model_dir: str, item_responses: pd.DataFrame, prediction_name: str):
  accelerator = Accelerator()
  np.random.seed(40) # sets a randomization seed for reproducibility
  transformers.set_seed(40)
  d = pd.read_csv(item_responses)
  d['text'] = d['response']
  d_input = d.filter(['text'], axis = 1)
  dataset = Dataset.from_pandas(d_input, preserve_index = False) # Turns pandas data into huggingface/pytorch dataset
  print(dataset) # show the dataset dictionary
  print(dataset.features)
  time.sleep(5)

  model = AutoModelForSequenceClassification.from_pretrained(trained_model_dir, num_labels = 1) # TONS of settings in the model call, but labels = 1
  tokenizer = AutoTokenizer.from_pretrained(trained_model_dir) # ...some settings in the tokenizer call

  def tokenize_function(examples):
    return tokenizer(examples['text'], truncation = True, padding = 'max_length')

  tokenized_datasets = dataset.map(tokenize_function, batched = True) # applies wrapper to our dataset


  #  DEFINE LOSS METRIC (ROOT MEAN SQUARED ERROR [rmse])
  def compute_metrics(eval_preds):
    predictions, references = eval_preds
    mse_metric = evaluate.load("mse")
    mse = mse_metric.compute(predictions = predictions, references = references)
    return mse
  
  training_args = TrainingArguments(         
      output_dir= "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model_evaluation", 
      learning_rate=0.00005,
      num_train_epochs=116,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16, 
      disable_tqdm=False,
      load_best_model_at_end=False,
      save_strategy = 'no',
      evaluation_strategy = 'no',
      eval_steps = 500,
      save_total_limit = 1)     


  trainer = accelerator.prepare(Trainer(
      model=model,
      args=training_args,
      eval_dataset=tokenized_datasets,
      compute_metrics=compute_metrics,
      tokenizer=tokenizer)
  )

  prediction = trainer.predict(tokenized_datasets)
  test_data = {'text':tokenized_datasets['text'],f'{prediction_name}':np.squeeze(prediction.predictions)}
  dataset_test_df = pd.DataFrame(test_data)
  dataset_test_df.to_csv(f'PredictedAISet{prediction_name}.csv')



# TODO: keep ALL the columns in the original df
if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--task", type=str)
  parser.add_argument("--trained_model_dir", type=str)
  parser.add_argument("--item_responses", type=str)
  parser.add_argument("--metric", type=str)
  parser.add_argument("--prediction", type=str)
  parser = parser.parse_args()
  if parser.task == "train":
    train_model(parser.metric)
  elif parser.task == "evaluate":
    evaluate_model(parser.trained_model_dir, parser.item_responses, parser.prediction)
  else:
    print("A task needs to be specified!")