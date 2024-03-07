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

warnings.filterwarnings(
    "ignore"
)  # so we don't have to see the sequential pipeline warning

import ItemGeneration, ItemEvaluation
from SelectItemGenShots import SelectItemGenShots

from optimize_item_gen_prompt import GenerateCPSResponses, RLPS_RoBERTa
from config import config
from key import OPENAI_KEY, GEMINI_KEY, ANTHROPIC_KEY

# OpenAI
from langchain.chat_models import ChatOpenAI

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI

# Claude
from langchain_anthropic import ChatAnthropic

# HF
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import pipeline as hf_pipeline


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
    np.random.seed(
        config["random_seed"]
    )  # sets a randomization seed for reproducibility
    transformers.set_seed(config["random_seed"])
    # save the config file
    config_path = Path(config["itemGenOutputFile"]).parent.absolute()
    with open(join(config_path, "config.json"), "w+") as cf:
        json.dump(config, cf)

    for i in range(config["numIter"]):
        print(f"Starting iteration {i} of experiment")
        print("Generating items")
        try:
            # load item gen model
            if (
                config["itemGenModelName"] == "gpt-4"
                or config["itemGenModelName"] == "gpt-3.5-turbo"
            ):
                model_kwargs = {
                    "top_p": config["itemGenTopP"],
                    "frequency_penalty": config["itemGenFrequencyPenalty"],
                    "presence_penalty": config["itemGenPresencePenalty"],
                }
                llm = ChatOpenAI(
                    model_name=config["itemGenModelName"],
                    openai_api_key=OPENAI_KEY,
                    temperature=config["itemGenTemperature"],
                    max_tokens=config["itemGenMaxTokens"],
                    model_kwargs=model_kwargs,
                )
            elif config["itemGenModelName"] == "google":
                # gemini doesn't have a frequency or presence penalty
                llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    convert_system_message_to_human=True,
                    google_api_key=GEMINI_KEY,
                    temperature=config["itemGenTemperature"],
                    top_p=config["itemGenTopP"],
                    max_output_tokens=config["itemGenMaxTokens"],
                    max_retries=1,
                )
            elif config["itemGenModelName"] == "claude-3":
                llm = ChatAnthropic(
                    model_name="claude-3-sonnet-20240229",
                    max_tokens_to_sample=config["itemGenMaxTokens"],
                    temperature=config["itemGenTemperature"],
                    anthropic_api_key=ANTHROPIC_KEY,
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
                llm = HuggingFacePipeline(pipeline=pipeline)

        except Exception as e:
            print(e)
            exit(-1)

        if i == 0:
            ItemGeneration.create_scenarios(
                config["itemGenPromptIdx"],
                config["itemGenModelName"],
                llm,
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
            )
        elif i >= 1:
            ItemGeneration.create_scenarios(
                config["itemGenPromptIdx"],
                config["itemGenModelName"],
                llm,
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
            )
        # evaluate items
        if config["useItemEvalModel"]:
            print("Evaluating Items")
            try:
                if (
                    config["itemEvalModelName"] == "gpt-4"
                    or config["itemEvalModelName"] == "gpt-3.5-turbo"
                ):
                    model_kwargs = {
                        "top_p": config["itemEvalTopP"],
                        "frequency_penalty": config["itemEvalFrequencyPenalty"],
                        "presence_penalty": config["itemEvalPresencePenalty"],
                    }
                    llm = ChatOpenAI(
                        model_name=config["itemEvalModelName"],
                        openai_api_key=OPENAI_KEY,
                        temperature=config["itemEvalTemperature"],
                        max_tokens=config["itemEvalMaxTokens"],
                        model_kwargs=model_kwargs,
                    )
                elif config["itemEvalModelName"] == "google":
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-pro",
                        convert_system_message_to_human=True,
                        google_api_key=GEMINI_KEY,
                        temperature=config["itemEvalTemperature"],
                        top_p=config["itemEvalTopP"],
                        max_output_tokens=config["itemEvalMaxTokens"],
                        max_retries=1,
                    )
                elif config["itemEvalModelName"] == "claude-3":
                    llm = ChatAnthropic(
                        model_name="claude-3-sonnet-20240229",
                        max_tokens_to_sample=config["itemEvalMaxTokens"],
                        temperature=config["itemEvalTemperature"],
                        anthropic_api_key=ANTHROPIC_KEY,
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
                    llm = HuggingFacePipeline(pipeline=pipeline)

            except Exception as e:
                print(e)
                exit(-1)

            ItemEvaluation.evaluate_scenarios(
                config["itemEvalPromptIdx"],
                config["itemEvalOutputFile"],
                config["itemEvalModelName"],
                llm,
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

        print("Generating Item Responses")
        # generate item responses
        try:
            if (
                config["itemResponseGenModelName"] == "gpt-4"
                or config["itemResponseGenModelName"] == "gpt-3.5-turbo"
            ):
                model_kwargs = {
                    "top_p": config["itemResponseGenTopP"],
                    "frequency_penalty": config["itemResponseGenFrequencyPenalty"],
                    "presence_penalty": config["itemResponseGenPresencePenalty"],
                }
                llm = ChatOpenAI(
                    model_name=config["itemResponseGenModelName"],
                    openai_api_key=OPENAI_KEY,
                    temperature=config["itemResponseGenTemperature"],
                    max_tokens=config["itemResponseGenMaxTokens"],
                    model_kwargs=model_kwargs,
                )
            elif config["itemResponseGenModelName"] == "google":
                llm = ChatGoogleGenerativeAI(
                    model="gemini-pro",
                    convert_system_message_to_human=True,
                    google_api_key=GEMINI_KEY,
                    temperature=config["itemGenTemperature"],
                    top_p=config["itemGenTopP"],
                    max_output_tokens=config["itemGenMaxTokens"],
                    max_retries=1,
                )
            elif config["itemResponseGenModelName"] == "claude-3":
                llm = ChatAnthropic(
                    model_name="claude-3-sonnet-20240229",
                    max_tokens_to_sample=config["itemGenMaxTokens"],
                    temperature=config["itemGenTemperature"],
                    anthropic_api_key=ANTHROPIC_KEY,
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
                llm = HuggingFacePipeline(pipeline=pipeline)
        except Exception as e:
            print(e)
            exit(-1)

        # TODO: whether or not item eval was used should be check to make sure the correct file is updated.
        GenerateCPSResponses.create_scenario_responses(
            llm,
            i,  # round
            config["itemGenOutputFile"],
            config["demographicsFile"],
            config["itemResponseGenOutputFile"],
            config["itemResponseGenModelName"],
        )

        # evaluate item responses
        if config["useItemResponseEvalModel"]:
            RLPS_RoBERTa.predict_with_model(
                config["itemResponseOriginalityModelDir"],
                config["itemResponseGenOutputFile"],
                "originality",
                config[
                    "itemResponseGenOutputFile"
                ],  # we need to chop off most columns from the first instance, so send another copy to save to
                i,  # round
            )
            RLPS_RoBERTa.predict_with_model(
                config["itemResponseQualityModelDir"],
                config["itemResponseGenOutputFile"],
                "quality",
                config[
                    "itemResponseGenOutputFile"
                ],  # we need to chop off most columns from the first instance, so send another copy to save to
                i,  # round
            )

            item_shots = SelectItemGenShots(
                config["itemResponseGenOutputFile"],
                config["shotSelectionMetric"],
                config["itemGenNumShots"],
                i,
                config["shotSelectionSort"],
                config["shotSelectionAggregate"],
                False # TODO make config arg
            )


RunExperiment(config)
