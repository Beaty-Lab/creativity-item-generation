import pandas as pd
import numpy as np
import torch

import evaluate
import json
import transformers
from scipy.stats import spearmanr
from os.path import join
from pathlib import Path
from typing import List
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
from tqdm import tqdm
from langchain.schema import StrOutputParser
from openai import OpenAI


# Claude
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOpenAI
from transformers import pipeline as hf_pipeline
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from datasets import load_from_disk

from prompts import prompts
from Config import few_shot_config
from key import ANTHROPIC_KEY, OPENAI_KEY
from random import choice
import time


def compute_metrics(predictions, references):
    correlation = spearmanr(predictions, references)[0]
    return {"Spearman Correlation": correlation}

# def to_tokens_and_logprobs(model, tokenizer, input_texts):        
#     input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").input_ids.cuda()
#     outputs = model(input_ids)
#     probs = torch.log_softmax(outputs.logits, dim=-1).detach()

#     # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
#     probs = probs[:, :-1, :]
#     input_ids = input_ids[:, 1:]
#     gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

#     batch = []
    
#     for input_sentence, input_probs in zip(input_ids, gen_probs):
#         text_sequence = []
#         for token, p in zip(input_sentence, input_probs):
#             if token not in tokenizer.all_special_ids:
#                 text_sequence.append((tokenizer.decode(token), p.item()))
#         batch.append(text_sequence)
#     return batch


# a version of the data prep function for prompting
def PrepareFewShotDataset(df: str, label: str) -> List:
    d = pd.read_csv(df)
    d = d.sample(frac=1.0)
    if label == "complexity_aggregrate":
        d["text"] = "Item: " + d["creative_scenario_round_4"] + " Complexity: " + d["complexity_aggregrate"].astype(str)
    elif label == "difficulty_aggregrate":
        d["text"] = "Item: " + d["creative_scenario_round_4"] + " Difficulty: " + d["difficulty_aggregrate"].astype(str)
    return d


def LLMTrial():
    train_df = PrepareFewShotDataset(
        few_shot_config["TrainSet"], few_shot_config["label"]
    )
    test_df = PrepareFewShotDataset(
        few_shot_config["TestSet"], few_shot_config["label"]
    )
    # sample n exemplars to include in all prompts, use seed for reproducability
    # can't do stratified sampling due to having only one 5 in the train set
    train_shots = train_df.sample(
        n=few_shot_config["NumShots"],
        replace=False,
        random_state=few_shot_config["random_seed"],
    )
    # mix the order of positives and negatives
    train_shots = train_shots.sample(
        n=len(train_shots),
        random_state=few_shot_config["random_seed"],
        replace=False,
    )
    train_shots = list(train_shots["text"])

   
    test_df["pred_label"] = None

    np.random.seed(
        few_shot_config["random_seed"]
    )  # sets a randomization seed for reproducibility
    transformers.set_seed(few_shot_config["random_seed"])

    prompt = ChatPromptTemplate.from_messages(prompts[few_shot_config["PromptIdx"]])

    try:
        if few_shot_config["ModelName"] == "claude-3":
            model = ChatAnthropic(
                model_name="claude-3-haiku-20240307",  # TODO: no hard code here
                max_tokens_to_sample=few_shot_config["MaxTokens"],
                temperature=few_shot_config["Temperature"],
                anthropic_api_key=ANTHROPIC_KEY,
            )
        elif "gpt" in few_shot_config["ModelName"]:
            model_kwargs = {
                "top_p": few_shot_config["TopP"],
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
            model = ChatOpenAI(
                model_name=few_shot_config["ModelName"],
                openai_api_key=OPENAI_KEY,
                temperature=few_shot_config["Temperature"],
                max_tokens=few_shot_config["MaxTokens"],
                model_kwargs=model_kwargs,
            )
        else:
            bnb_config = bnb_config = BitsAndBytesConfig(
                bnb_4bit_compute_dtype=torch.float16,
                # load_in_4bit=True,
            )
            model_kwargs = {
                "top_p": few_shot_config["TopP"],
                "temperature": few_shot_config["Temperature"],
                "device_map": "auto",
                # "torch_dtype": torch.bfloat16, # don't use with 8 bit mode
            }
            tokenizer = AutoTokenizer.from_pretrained(
                few_shot_config["TokenizerName"], add_prefix_space=True, **model_kwargs
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if "gemma" in few_shot_config["ModelName"]:
                model = AutoModelForCausalLM.from_pretrained(
                    few_shot_config["ModelName"],
                    do_sample=True,
                    torch_dtype=torch.bfloat16,
                    # revision="float16",
                    **model_kwargs,
                )
            elif "t5" in few_shot_config["ModelName"]:
                print("SeqtoSeq models not supported!")
                exit(-1)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    few_shot_config["ModelName"],
                    do_sample=True,
                    quantization_config=bnb_config,
                    **model_kwargs,
                )
            if model.config.pad_token_id is None:
                model.config.pad_token_id = model.config.eos_token_id

            pipeline = hf_pipeline(
                "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2
            )
            
            model = HuggingFacePipeline(pipeline=pipeline)

    except Exception as e:
        print(e)
        exit(-1)

    # main inference loop
    if few_shot_config["use_human_test_set"]:
        # score quality of human written items
        # TODO: finish
        pass
        # for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        #     # invoke the llm to predict the val sample
        #     completion_chain = prompt | model | StrOutputParser()
            # try:
            #     result = completion_chain.invoke(
            #         {
            #             "examples": "\n###\n".join(train_shots),
            #             "exemplar": row["Response"] + " Label: ",
            #             "heldout_problem": row["ProblemFull"],
            #             "train_problem": train_df.iloc[0]["ProblemFull"], # TODO: needs to change when theres more than one train item
            #         }
            #     )
            # except Exception as e:
            #     # most likely antropic error, wait a few seconds and try 1 more time
            #     time.sleep(5)
            #     result = completion_chain.invoke(
            #         {
            #             "examples": "\n###\n".join(train_shots),
            #             "exemplar": row["Response"] + " Label: ",
            #             "heldout_problem": row["ProblemFull"],
            #             "train_problem": train_df.iloc[0]["ProblemFull"], # TODO: needs to change when theres more than one train item
            #         }
            #     )

            # try:
            #     result = int(result)
            # except Exception:
            #     if result[-1].isdigit():
            #         result = int(result[-1])
            #     else:
            #         result = choice([0, 1, 2, 3])
            # test_df.at[index, "pred_label"] = result
    else:
        # use the AI item test set
        for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            # invoke the llm to predict the val sample
            if few_shot_config["label"] == "complexity_aggregrate":
                label = "Complexity"
            else:
                label = "Difficulty"
            completion_chain = prompt | model | StrOutputParser()
            try:
                
                result = completion_chain.invoke(
                    {
                        "examples": "\n###\n".join(train_shots),
                        "exemplar": "Item: " + row["creative_scenario_round_4"] + f" {label}: ",
                    }
                )
            except Exception as e:
                # most likely antropic error, wait a few seconds and try 1 more time
                time.sleep(5)
                result = completion_chain.invoke(
                    {
                        "examples": "\n###\n".join(train_shots),
                        "exemplar": "Item: " + row["creative_scenario_round_4"] + f" {label}: ",
                    }
                )
            try:
                result = int(result)
            except Exception:
                if result[-1].isdigit():
                    result = int(result[-1])
                else:
                    result = choice([0, 1, 2, 3])
            test_df.at[index, "pred_label"] = result


    if few_shot_config["use_test_set"]:
        preds = list(test_df["pred_label"])
        labels = list(test_df["label"])

    metrics = compute_metrics(preds, labels)
    print(metrics)


if __name__ == "__main__":
    LLMTrial()
