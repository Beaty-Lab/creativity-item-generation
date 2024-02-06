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

import ItemGeneration, ItemEvaluation
import uuid  # used for experiment ids
from optimize_item_gen_prompt import GenerateCPSResponses, RLPS_RoBERTa
from config import config
from key import key

# OpenAI
from langchain.chat_models import ChatOpenAI

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


def RunExperiment(config: dict):
    # TODO (MAYBE): log metrics to wandb!
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
                    openai_api_key=key,
                    temperature=config["itemGenTemperature"],
                    max_tokens=config["itemGenMaxTokens"],
                    model_kwargs=model_kwargs,
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
                    config["itemGenModelName"], load_in_8bit=True, **model_kwargs
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

        except Exception:
            print("Model failed to initialize. Please check your API key.")
            exit(-1)

        if i == 0:
            ItemGeneration.create_scenarios(
                config["itemGenPromptIdx"],
                config['itemGenOutputFile'],
                config["itemGenModelName"],
                llm,
                i,  # round
                config["itemGenMaxTokens"],
                config["itemGenPresencePenalty"],
                config["itemGenFrequencyPenalty"],
                config["itemGenTemperature"],
                config["itemGenTopP"],
                config["itemGenOutputFile"],
                input_file=None,
                wordlist_file=config["wordlistFile"],
                num_items_per_prompt=config["numItemsPerList"],
            )
        elif i >= 1:
            ItemGeneration.create_scenarios(
                config["itemGenPromptIdx"],
                config['itemGenOutputFile'],
                config["itemGenModelName"],
                llm,
                i,  # round
                config["itemGenMaxTokens"],
                config["itemGenPresencePenalty"],
                config["itemGenFrequencyPenalty"],
                config["itemGenTemperature"],
                config["itemGenTopP"],
                config["itemGenOutputFile"],
                input_file=config['itemGenOutputFile'],  # TODO: make sure the output file is consistent from the item evaluator, etc
                wordlist_file=config["wordlistFile"],
                num_items_per_prompt=config["numItemsPerList"],
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
                        openai_api_key=key,
                        temperature=config["itemEvalTemperature"],
                        max_tokens=config["itemEvalMaxTokens"],
                        model_kwargs=model_kwargs,
                    )

                else:
                    print("Only OpenAI models are supporting for evaluating items.")

            except Exception:
                print("Model failed to initialize. Please check your API key.")
                exit(-1)

            ItemEvaluation.evaluate_scenarios(
                config["itemEvalPromptIdx"],
                config['itemEvalOutputFile'],
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
                config["itemEvalTopP"]
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
                    openai_api_key=key,
                    temperature=config["itemResponseGenTemperature"],
                    max_tokens=config["itemResponseGenMaxTokens"],
                    model_kwargs=model_kwargs,
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
                    load_in_8bit=True,
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
        except Exception:
            print("Model failed to initialize. Please check your API key.")
            exit(-1)

        # TODO: whether or not item eval was used should be check to make sure the correct file is updated.
        GenerateCPSResponses.create_scenario_responses(
            llm,
            i,  # round
            f"{config['itemGenOutputFile']}",
            config["demographicsFile"],
        )

        # evaluate item responses
        if config["useItemEvalModel"]:
            RLPS_RoBERTa.evaluate_model(
                config["itemResponseOriginalityModelDir"],
                config['itemEvalOutputFile'],
                "originality",
                config['itemEvalOutputFile'],  # we need to chop off most columns from the first instance, so send another copy to save to
                i,  # round
            )
            RLPS_RoBERTa.evaluate_model(
                config["itemResponseQualityModelDir"],
                config['itemEvalOutputFile'],
                "quality",
                config['itemEvalOutputFile'],  # we need to chop off most columns from the first instance, so send another copy to save to
                i,  # round
            )


RunExperiment(config)
