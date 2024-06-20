"""
    Use LLMs (Claude-3 and GPT-4) and prompt engineering to score.
    Try many (100+) shot with API models.
    For all other models, try vanilla prompting, RAG, and structure prompting (CoT, Z-CoT, Self Consistency, Tree / Graph of thoughts, etc.)
    Optionally, explore methods for prompt optimization.
    NOTE: unlike Peft, these are all hard prompts.
"""

# TODO: finish, separateme methods for tuning and inference using sklearn-like API
import pandas as pd
import numpy as np
import torch

# import wandb
import evaluate
import json
import transformers
from scipy.stats import spearmanr
from os.path import join
from pathlib import Path
from typing import List
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
from tqdm import tqdm
from langchain.schema import StrOutputParser
from openai import OpenAI


# Claude
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import ChatOpenAI
from transformers import pipeline as hf_pipeline
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from prompts import prompts, assistant_prompts
from Config import few_shot_config
from key import ANTHROPIC_KEY, OPENAI_KEY
from random import choice

RESPONSE_COL = "Solutions"
INPUT_COL = "ProblemFull"


def compute_metrics(predictions, references):
    # predictions = [int(p) for p in predictions]
    # new_preds = []
    # for index, value in enumerate(predictions):
    #     if type(value) == int:
    #         new_preds.append(value)
    #     else:
    #         references.pop(index)
    accuracy = evaluate.load("accuracy")
    accuracy = accuracy.compute(predictions=predictions, references=references)[
        "accuracy"
    ]
    correlation = spearmanr(predictions, references)
    return {"accuracy": accuracy, "Spearman Correlation": correlation}


# a version of the data prep function for prompting
def PrepareFewShotDataset(df: str, item: str) -> List:
    d = pd.read_csv(df)
    d.dropna(inplace=True)
    d = d.sample(frac=1.0)
    # convert labels using the quartiles approach
    describe = d["FacScoresO"].describe()
    lower_quartile = describe["25%"]
    middle_quartile = describe["50%"]
    upper_quartile = describe["75%"]
    d["label"] = d["FacScoresO"].apply(
        lambda x: (
            0
            if x < lower_quartile
            else 1 if x < middle_quartile else 2 if x < upper_quartile else 3
        )
    )

    d["text"] = "Problem: " + d[INPUT_COL] + "\nResponse: " + d[RESPONSE_COL] + "\n"
    problem = d[d["ProblemID"] == item].iloc[0]["ProblemFull"]
    train = d[d["set"] == "training"]
    test = d[d["set"] == "test"]
    heldout = d[d["set"] == "heldout"]
    train = train[train["ProblemID"] == item]
    test = test[test["ProblemID"] == item]
    train = train.filter(["Solutions", "text", "label"], axis=1)
    test = test.filter(["Solutions", "text", "label"], axis=1)
    heldout = heldout.filter(["text", "label"], axis=1)
    return problem, train, test, heldout


def LLMTrial():
    # if not few_shot_config["use_sweep"]:
    #     wandb.init(
    #         project="LLM Invalid Response Detection",
    #         config={
    #             "model": few_shot_config["ModelName"],
    #             "task": few_shot_config["Task"],
    #             "NumShots": few_shot_config["NumShots"],
    #             "Temperature": few_shot_config["Temperature"],
    #             "TopP": few_shot_config["TopP"],
    #             "seed": few_shot_config["random_seed"],
    #             "MaxTokens": few_shot_config["MaxTokens"],
    #             "PromptIdx": few_shot_config["PromptIdx"],
    #         },
    #     )
    problem, train_df, test_df, heldout_df = PrepareFewShotDataset(
        few_shot_config["dataset"], few_shot_config["TrainItem"]
    )
    # sample n exemplars to include in all prompts, use seed for reproducability
    if "asst" not in few_shot_config["ModelName"]:
        train_shots = train_df.groupby("label").sample(
            n=few_shot_config["NumShots"] // len(train_df["label"].unique()),
            replace=False,
            random_state=few_shot_config["random_seed"],
        )
        train_shots["label"] = train_shots["label"].astype("str")
        train_shots["example"] = (
            train_shots["Solutions"] + " Label: " + train_shots["label"]
        )
        # mix the order of positives and negatives
        train_shots = train_shots.sample(
            n=len(train_shots),
            random_state=few_shot_config["random_seed"],
            replace=False,
        )
        train_shots = list(train_shots["example"])
    test_df["pred_label"] = None

    if few_shot_config["use_test_set"]:
        heldout_df["pred_label"] = None

    np.random.seed(  # TODO: add a few_shot_config file for all params
        few_shot_config["random_seed"]
    )  # sets a randomization seed for reproducibility
    transformers.set_seed(few_shot_config["random_seed"])

    few_shot_config_path = Path(few_shot_config["OutputFile"])
    with open(join(few_shot_config_path, "few_shot_config.json"), "w+") as cf:
        json.dump(few_shot_config, cf)

    with open(few_shot_config["logFile"], "w+") as log:
        log.writelines("Starting Trial...\n")

    if "asst" in few_shot_config["ModelName"]:
        prompt = ChatPromptTemplate.from_messages(
            assistant_prompts[few_shot_config["PromptIdx"]]
        )
    else:
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
        elif "asst" in few_shot_config["ModelName"]:
            # model = OpenAIAssistantRunnable(
            #     assistant_id=few_shot_config["ModelName"],
            #     as_agent=True,
            # )
            try:
                client = OpenAI()
                model = client.beta.assistants.retrieve(few_shot_config["ModelName"])
            except Exception as e:
                print("Specified assistant not found!")
        else:
            bnb_config = bnb_config = BitsAndBytesConfig(
                bnb_16bit_compute_dtype=torch.bfloat16,
                bnb_4bit_compute_dtype=torch.float16,
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

            model = AutoModelForCausalLM.from_pretrained(
                few_shot_config["ModelName"],
                quantization_config=bnb_config,
                **model_kwargs,
            )
            if model.config.pad_token_id is None:
                model.config.pad_token_id = model.config.eos_token_id

    except Exception as e:
        with open(few_shot_config["logFile"], "a") as log:
            print(e)
            log.writelines(str(e) + "\n")
        exit(-1)

    # main inference loop
    if few_shot_config["use_test_set"]:  # TODO: update
        for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            # invoke the llm to predict the val sample
            prompt_f = prompt.format(
                **{
                    "examples": " ".join(train_shots),
                    "exemplar": row["text"] + " Label: ",
                }
            )
            if few_shot_config["PromptIdx"] == 1:  # zero shot chain of thought
                input_ids = tokenizer(
                    prompt_f, padding=True, return_tensors="pt"
                ).input_ids.cuda()
                prompt_cot = model.generate(input_ids, max_new_tokens=40)
                prompt_cot = tokenizer.batch_decode(
                    prompt_cot, skip_special_tokens=True
                )[0]
                prompt_0 = prompt_cot + "Label: 0"
                prompt_1 = prompt_cot + "Label: 1"
            else:
                prompt_0 = prompt_f + " 0"
                prompt_1 = prompt_f + " 1"
            if "t5" in few_shot_config["ModelName"]:
                # the logprob function doesn't work for t5, '1' seems to be a special token
                # so we have no choice but to rely on standard text generation.
                if few_shot_config["PromptIdx"] == 1:
                    t5_prompt = prompt_cot
                else:
                    t5_prompt = prompt_f

                input_ids = tokenizer(
                    t5_prompt, padding=True, return_tensors="pt"
                ).input_ids.cuda()
                pred = tokenizer.batch_decode(
                    model.generate(input_ids, max_new_tokens=2),
                    skip_special_tokens=True,
                )[0]
                try:
                    pred = int(pred)
                except Exception:
                    pred = choice([0, 1])
                test_df.at[index, "pred_label"] = pred
            else:
                prob_0 = to_tokens_and_logprobs(model, tokenizer, prompt_0)[0][-1][1]
                prob_1 = to_tokens_and_logprobs(model, tokenizer, prompt_1)[0][-1][1]
                pred = np.argmax([prob_0, prob_1])
                test_df.at[index, "pred_label"] = pred
    else:
        for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            # invoke the llm to predict the val sample
            if "asst" in few_shot_config["ModelName"]:
                thread = client.beta.threads.create()
                final_prompt = prompt.format(**{
                        "exemplar": row["Solutions"],
                }).replace("System:","").replace("Human:","") + " Originality:"

                message = client.beta.threads.messages.create(
                    thread_id=thread.id,
                    role="user",
                    content=final_prompt,
                )
                run = client.beta.threads.runs.create_and_poll(
                    thread_id=thread.id,
                    assistant_id=model.id,
                )
                result = client.beta.threads.messages.list(
                    thread_id=thread.id
                ).data[0].content[0].text.value
            else:
                completion_chain = prompt | model | StrOutputParser()
                result = completion_chain.invoke(
                    {
                        "examples": "\n###\n".join(train_shots),
                        "exemplar": row["Solutions"] + " Label: ",
                        "problem": problem,
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
        preds = list(heldout_df["pred_label"])
        labels = list(heldout_df["label"])
    else:
        preds = list(test_df["pred_label"])
        labels = list(test_df["label"])

    metrics = compute_metrics(preds, labels)
    print(metrics)
    test_df.to_csv(join(few_shot_config["OutputFile"], "test.csv"), index=False)
    # wandb.log(metrics)


if __name__ == "__main__":
    # if few_shot_config["use_sweep"]:
    #     run = wandb.init(project="LLM Invalid Response Detection")
    #     few_shot_config["NumShots"] = wandb.config.NumShots
    #     few_shot_config["random_seed"] = wandb.config.random_seed
    #     few_shot_config["topP"] = wandb.config.TopP
    #     few_shot_config["PromptIdx"] = wandb.config.PromptIdx
    #     print(few_shot_config["NumShots"])
    #     LLMTrial()
    # else:
    LLMTrial()
