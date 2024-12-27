"""
Main driver script for running AIG experiments.
We have a few different components that may be combined in different ways:
1. Item generation
2. Item evaluation
3. Item response generation
4. Item response evaluation

The purpose of this script is allow the useer to run AIG trials incroporating each component,
And giving control of each.
Initially, for simplicity, we will only support CPS.
All parameters are stored in config.py
"""

import pandas as pd
import numpy as np
import warnings
import transformers
import json
from pathlib import Path
from os.path import join
from typing import Tuple

warnings.filterwarnings(
    "ignore"
)  # so we don't have to see the sequential pipeline warning

import ItemGeneration, ItemEvaluation
from SelectItemGenShots import SelectItemGenShots
# from ETL import UploadtoDB

from optimize_item_gen_prompt import GenerateCPSResponses 
# RLPS_RoBERTa
from config import config
from key import OPENAI_API_KEY, GEMINI_KEY, ANTHROPIC_API_KEY

# OpenAI
from langchain_openai import ChatOpenAI

# Claude
from langchain_anthropic import ChatAnthropic

# HF
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import pipeline as hf_pipeline

from Task import init_task
from Prompts import load_prompts

# load all LLMs
def _init_models(config: dict) -> Tuple:
    try:
        # load item gen model
        if (
            config["itemGenModelName"] == "gpt-4"
            or config["itemGenModelName"] == "gpt-3.5-turbo"
            or config["itemGenModelName"] == "gpt-4o-mini"
            or config["itemGenModelName"] == "gpt-4o"
        ):
            model_kwargs = {
                "top_p": config["itemGenTopP"],
                "frequency_penalty": config["itemGenFrequencyPenalty"],
                "presence_penalty": config["itemGenPresencePenalty"],
            }
            item_gen_llm = ChatOpenAI(
                model_name=config["itemGenModelName"],
                openai_api_key=OPENAI_API_KEY,
                temperature=config["itemGenTemperature"],
                max_tokens=config["itemGenMaxTokens"],
                model_kwargs=model_kwargs,
            )
        elif config["itemGenModelName"] == "claude-3-haiku" or config["itemGenModelName"] == "claude-3-5-haiku-20241022":
            item_gen_llm = ChatAnthropic(
                model_name=config["itemGenModelName"],
                max_tokens_to_sample=config["itemGenMaxTokens"],
                temperature=config["itemGenTemperature"],
                anthropic_api_key=ANTHROPIC_API_KEY,
            )
        else:
            model_kwargs = {
                "top_p": config["itemGenTopP"],
                "temperature": config["itemGenTemperature"],
                "device_map": "auto",
                # "torch_dtype": torch.bfloat16, # don't use with 8 bit mode
            }
            tokenizer = AutoTokenizer.from_pretrained(
                config["itemGenModelName"], **model_kwargs
            )
            model = AutoModelForCausalLM.from_pretrained(
                config["itemGenModelName"],
                load_in_4bit=True,
                # max_new_tokens=config["itemGenMaxTokens"],
                **model_kwargs,
            )

            pipeline = hf_pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                batch_size=1,
                max_new_tokens=config["itemGenMaxTokens"],
                model_kwargs=model_kwargs,
            )
            item_gen_llm = HuggingFacePipeline(pipeline=pipeline)

    except Exception as e:
        with open(config["logFile"], "a") as log:
            print(e)
            log.writelines(str(e) + "\n")
        exit(-1)

    if config["useItemEvalModel"]:
        with open(config["logFile"], "a") as log:
            print("Evaluating Items")
            log.writelines("Evaluating Items\n")
        try:
            if (
                config["itemEvalModelName"] == "gpt-4"
                or config["itemEvalModelName"] == "gpt-3.5-turbo"
                or config["itemEvalModelName"] == "gpt-4o-mini"
                or config["itemEvalModelName"] == "gpt-4o"
            ):
                model_kwargs = {
                    "top_p": config["itemEvalTopP"],
                    "frequency_penalty": config["itemEvalFrequencyPenalty"],
                    "presence_penalty": config["itemEvalPresencePenalty"],
                }
                item_eval_llm = ChatOpenAI(
                    model_name=config["itemEvalModelName"],
                    openai_api_key=OPENAI_API_KEY,
                    temperature=config["itemEvalTemperature"],
                    max_tokens=config["itemEvalMaxTokens"],
                    model_kwargs=model_kwargs,
                )
            elif config["itemEvalModelName"] == "claude-3-haiku" or config["itemEvalModelName"] == "claude-3-5-haiku-20241022":
                item_eval_llm = ChatAnthropic(
                    model_name=config["itemEvalModelName"],
                    max_tokens_to_sample=config["itemEvalMaxTokens"],
                    temperature=config["itemEvalTemperature"],
                    anthropic_api_key=ANTHROPIC_API_KEY,
                )
            else:
                model_kwargs = {
                    "top_p": config["itemEvalTopP"],
                    "temperature": config["itemEvalTemperature"],
                    "device_map": "auto",
                    # "torch_dtype": torch.bfloat16, # don't use with 8 bit mode
                }
                tokenizer = AutoTokenizer.from_pretrained(
                    config["itemEvalModelName"], **model_kwargs
                )
                model = AutoModelForCausalLM.from_pretrained(
                    config["itemEvalModelName"],
                    load_in_4bit=True,
                    **model_kwargs,
                )

            pipeline = hf_pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                batch_size=1,
                max_new_tokens=config["itemEvalMaxTokens"],
                model_kwargs=model_kwargs,
            )
            item_eval_llm = HuggingFacePipeline(pipeline=pipeline)

        except Exception as e:
            with open(config["logFile"], "a") as log:
                print(e)
                log.writelines(str(e) + "\n")
            exit(-1)
    else:
        item_eval_llm = None
    try:
        # TODO: this cause item response gen parameters to be ignored when the same model is used for both!
        # if config["itemGenModelName"] == config["itemResponseGenModelName"]:
        #     item_response_llm = item_gen_llm
        if (
            config["itemResponseGenModelName"] == "gpt-4"
            or config["itemResponseGenModelName"] == "gpt-3.5-turbo"
            or config["itemResponseGenModelName"] == "gpt-4o-mini"
            or config["itemResponseGenModelName"] == "gpt-4o"
        ):
            model_kwargs = {
                "top_p": config["itemResponseGenTopP"],
                "frequency_penalty": config["itemResponseGenFrequencyPenalty"],
                "presence_penalty": config["itemResponseGenPresencePenalty"],
            }
            item_response_llm = ChatOpenAI(
                model_name=config["itemResponseGenModelName"],
                openai_api_key=OPENAI_API_KEY,
                temperature=config["itemResponseGenTemperature"],
                max_tokens=config["itemResponseGenMaxTokens"],
                model_kwargs=model_kwargs,
            )
        elif config["itemResponseGenModelName"] == "claude-3-haiku" or config["itemResponseGenModelName"] == "claude-3-5-haiku-20241022":
            item_response_llm = ChatAnthropic(
                model_name=config["itemResponseGenModelName"],
                max_tokens_to_sample=config["itemGenMaxTokens"],
                temperature=config["itemGenTemperature"],
                anthropic_api_key=ANTHROPIC_API_KEY,
            )
        else:
            model_kwargs = {
                "top_p": config["itemResponseGenTopP"],
                "temperature": config["itemResponseGenTemperature"],
                "device_map": "auto",
                # "torch_dtype": torch.bfloat16, # don't use with 8 bit mode
            }
            tokenizer = AutoTokenizer.from_pretrained(
                config["itemResponseGenModelName"], **model_kwargs
            )
            model = AutoModelForCausalLM.from_pretrained(
                config["itemResponseGenModelName"],
                load_in_4bit=True,
                **model_kwargs,
            )

        pipeline = hf_pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            batch_size=1,
            max_new_tokens=config["itemResponseGenMaxTokens"],
            model_kwargs=model_kwargs,
        )
        item_response_llm = HuggingFacePipeline(pipeline=pipeline)
    except Exception as e:
        with open(config["logFile"], "a") as log:
            print(e)
            log.writelines(str(e) + "\n")
        exit(-1)
    
    return item_gen_llm, item_eval_llm, item_response_llm


# TODO: add your task to the init function
# TODO: this can be moved to Task.py and just pass


"""
A function to run an AIG trial. Only LLama and OpenAI models should be used.
Parameters:
    num_iter: int, how many iterations of item gen to run
    useItemEvalModel: bool, whether to use GPT-4 item evaluation
    itemResponseEvalMetric: str, the metric for item response evaluation
    numItemsPerList: int, the number of items per wordlist to generate
"""


# TODO The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead.
def RunExperiment(config: dict):
    """
    A function to run an AIG trial. Only LLama and OpenAI models should be used.
    Parameters:
        num_iter: int, how many iterations of item gen to run
        useItemEvalModel: bool, whether to use GPT-4 item evaluation
        itemResponseEvalMetric: str, the metric for item response evaluation
        numItemsPerList: int, the number of items per wordlist to generate
    """
    
    np.random.seed(
        config["random_seed"]
    )  # sets a randomization seed for reproducibility
    transformers.set_seed(config["random_seed"])
    # save the config file
    config_path = Path(config["itemGenOutputFile"]).parent.absolute()
    try:
        with open(join(config_path, "config.json"), "w+") as cf:
            json.dump(config, cf)
    except FileNotFoundError as e:
        Path(config_path).mkdir(parents=True, exist_ok=True)
        with open(join(config_path, "config.json"), "w+") as cf:
            json.dump(config, cf)

    with open(join(config_path, config["logFile"]), "w+") as log:
        log.writelines("Starting Trial...\n")

    item_gen_llm, item_eval_llm, item_response_llm = _init_models(config)

    # TODO: generalize wordlist creation
    item_gen_prompt, item_eval_prompt, item_response_gen_prompt, worlist_gen_prompt = load_prompts()
    # TODO: the above is just prompt templates
    # the task needs to create the final prompt
    task = init_task(config)

    for i in range(config["numIter"]):
        with open(join(config_path, config["logFile"]), "a") as log:
            print(f"Starting iteration {i} of experiment")
            print("Generating items")
            log.writelines(f"Starting iteration {i} of experiment\n")
            log.writelines("Generating items\n")

        if i == 0:
            ItemGeneration.create_scenarios(
                item_gen_prompt,
                config["itemGenModelName"],
                item_gen_llm,
                i,  # round
                config["itemGenMaxTokens"],
                config["itemGenPresencePenalty"],
                config["itemGenFrequencyPenalty"],
                config["itemGenTemperature"],
                config["itemGenTopP"],
                config["itemGenOutputFile"],
                config["numItemGenerationAttempts"],
                input_file=None,
                wordlist_file=config["wordlistFile"],
                task_parser=task
            )
        elif i >= 1:
            if not config["useItemScoring"]:
                ItemGeneration.create_scenarios(
                item_gen_prompt,
                config["itemGenModelName"],
                item_gen_llm,
                0,  # round
                config["itemGenMaxTokens"],
                config["itemGenPresencePenalty"],
                config["itemGenFrequencyPenalty"],
                config["itemGenTemperature"],
                config["itemGenTopP"],
                config["itemGenOutputFile"],
                config["numItemGenerationAttempts"],
                input_file=config[
                    "itemGenOutputFile"
                ],  # TODO: make sure the output file is consistent from the item evaluator, etc
                wordlist_file=config["wordlistFile"],
                item_shots=item_shots,  # The k shots to give to the prompt
                task_parser=task
            )
            else:
                ItemGeneration.create_scenarios(
                    item_gen_prompt,
                    config["itemGenModelName"],
                    item_gen_llm,
                    i,  # round
                    config["itemGenMaxTokens"],
                    config["itemGenPresencePenalty"],
                    config["itemGenFrequencyPenalty"],
                    config["itemGenTemperature"],
                    config["itemGenTopP"],
                    config["itemGenOutputFile"],
                    config["numItemGenerationAttempts"],
                    input_file=config[
                        "itemGenOutputFile"
                    ],  # TODO: make sure the output file is consistent from the item evaluator, etc
                    wordlist_file=config["wordlistFile"],
                    item_shots=item_shots,  # The k shots to give to the prompt
                    task_parser=task
                )
        # evaluate items
        if config["useItemEvalModel"]:
            ItemEvaluation.evaluate_scenarios(
                item_eval_prompt,
                config["itemEvalOutputFile"],
                config["itemEvalModelName"],
                item_eval_llm,
                i,  # round
                f"creative_scenario_round_{i}",
                config["itemEvalOutputFile"],
                config["itemGenOutputFile"],
                config["itemEvalFrequencyPenalty"],
                config["itemEvalPresencePenalty"],
                config["itemEvalMaxTokens"],
                config["itemEvalTemperature"],
                config["itemEvalTopP"],
            )

        with open(join(config_path, config["logFile"]), "a") as log:
            print("Generating Item Responses")
            log.writelines("Generating Item Responses\n")
        # generate item responses

        # TODO: whether or not item eval was used should be check to make sure the correct file is updated.
        if not config["useItemScoring"]:
            GenerateCPSResponses.create_scenario_responses(
                item_response_gen_prompt,
                item_response_llm,
                0,  # round
                config["itemGenOutputFile"],
                config["demographicsFile"],
                config["itemResponseGenOutputFile"],
                config["itemResponseGenModelName"],
                config["numResponsesPerItem"],
                config["itemResponseGenPromptIdx"],
                task_parser=task,
            )
        else:
            GenerateCPSResponses.create_scenario_responses(
                item_response_gen_prompt,
                item_response_llm,
                i,  # round
                config["itemGenOutputFile"],
                config["demographicsFile"],
                config["itemResponseGenOutputFile"],
                config["itemResponseGenModelName"],
                config["numResponsesPerItem"],
                config["itemResponseGenPromptIdx"],
                task_parser=task,
            )

        # evaluate item responses
        if config["useItemScoring"]:
            task.RunScorers(i)
            item_shots = SelectItemGenShots(
                config["itemResponseGenOutputFile"],
                config["shotSelectionMetric"],
                config["itemGenNumShots"],
                i,
                config["shotSelectionSort"],
                config["shotSelectionAggregate"],
                config["random_seed"],
                config["shotSelectionAlgorithm"],
                config["baselineSimScore"],
                config["baselineOriginality"],
            )
    # TODO: add DB support for writing items, checking for similarity within database


RunExperiment(config)
