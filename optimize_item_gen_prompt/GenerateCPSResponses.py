import time
import torch
import bitsandbytes
import re
import pandas as pd
import config

# HF
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import pipeline as hf_pipeline
from key import OPENAI_KEY
import time


from langchain.schema import BaseOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# API key stored in key.py, and should NOT be committed
from tqdm import tqdm
from argparse import ArgumentParser


# class for storing and manipulating prompts
class PromptGenerator:
    @staticmethod
    def make_creative_scenario_response_prompt(scendario_prompt_idx: int):
        # keep track of the different prompts
        # TODO: move the base prompts to a separate file that is imported, passed to function
        scenario_base_prompts = [
            [
                (
                    "system",
                    "You are a participant in an experiment. You will be presented with a problem scenario, and must come up with a solution to the problem. Be creative in your response, but keep it at no more than 4 sentences in length. Respond in a single paragraph.",
                ),
                (
                    "human",  # 1
                    """Scenario:
                    {creative_scenario}

        ###

        Solution:""",
                ),
            ],
            [
                (
                    "system",
                    "You are a participant in an experiment. You are a {ethnicity} {gender} who works in {industry}. Your job title is {title}. You will be presented with a problem scenario, and must come up with a solution to the problem. Be creative in your response, but keep it at no more than 4 sentences in length. Respond in a single paragraph.",
                ),
                (
                    "human",  # 2
                    """Scenario:
                    {creative_scenario}

        ###

        Solution:""",
                ),
            ],
        ]
        creative_scenario_generation_prompt = ChatPromptTemplate.from_messages(
            scenario_base_prompts[scendario_prompt_idx]
        )
        return creative_scenario_generation_prompt


class CreativityScenarioResponseParser(BaseOutputParser):
    # TODO: for the next round of items after feedback, get rid of everything generated before and after the start of the original item
    @staticmethod
    def parse(text: str) -> str:
        try:
            text = text.strip("\n").strip(" ")
        except Exception:
            # OpenAIs models with yield an AIMessage object
            text = text.content.strip("\n").strip(" ")
        # Remove intervening newlines
        text = re.sub("\n", "", text)
        text = re.sub("\t", "", text)

        return text


# TODO: refactor to parallelize
# 0. once the sequential code is stable, commit to main
# 1. change "chain" to "batch", which can invoke a chain on a batch of inputs
# 2. pass the entire set of items, not just this one
# 3. put this in a feature branch for parallelization
# See: https://python.langchain.com/docs/expression_language/interface#batch
def test_creative_response(
    problem,
    prompt_idx: int,
    llm,
    ethnicity: str = None,
    gender: str = None,
    industry: str = None,
    title: str = None,
):
    prompt = PromptGenerator.make_creative_scenario_response_prompt(
        prompt_idx
    )  # the prompt type

    chain = prompt | llm
    if prompt_idx == 0:
        result = chain.invoke({"creative_scenario": problem})
    elif prompt_idx == 1:
        result = chain.invoke(
            {
                "creative_scenario": problem,
                "ethnicity": ethnicity,
                "gender": gender,
                "industry": industry,
                "title": title,
            }
        )

    result = CreativityScenarioResponseParser.parse(result)

    return result


def create_scenario_responses(
    llm,
    round,
    input_file_name: str,
    demographics_file: str,
    response_file_name: str,
    model_name: str,
):
    input_file = pd.read_json(input_file_name)

    if demographics_file is not None:
        demographics_file = pd.read_csv(demographics_file, index_col=0)

    if demographics_file is not None:
        ai_responses = pd.DataFrame(
            columns=[
                f"creative_scenario_round_{round}",
                f"creative_response_round_{round}",
                f"ethnicity",
                f"gender",
                f"industry",
                f"title",
            ]
        )
    else:
        ai_responses = pd.DataFrame(
            columns=[
                f"creative_scenario_round_{round}",
                f"creative_response_round_{round}",
            ]
        )

    for index, row in tqdm(input_file.iterrows(), total=input_file.shape[0]):
        # generate 30 responses to each scenario
        # we assume demographics want to be included if the file was specified
        # if so, create a "profile" by drawing a random sample from that file
        # do not sample them again twice
        for i in tqdm(range(10)):  # TODO: make this a param, 30
            if (
                model_name == "gpt-3.5-turbo"
                or model_name == "gpt-4"
                or model_name == "google"
                or model_name == "claude-3"
            ):
                time.sleep(5)
            if demographics_file is not None:
                participant = demographics_file.sample(n=1)
                # especially for gemini models, the prompt may be blocked due to the safety filters
                # in those cases, skip and move on
                try:
                    result = test_creative_response(
                        row[f"creative_scenario_round_{round}"],
                        1,  # prompt_idx
                        llm,
                        participant["Q15"].values[0],  # ethnicity
                        participant["Q14"].values[0],  # gender
                        participant["Q24"].values[0],  # industry
                        participant["Q23"].values[0],  # title
                    )
                    print(result)
                except Exception as e:
                    with open(config["logFile"], "w") as log:
                        print(e)
                        log.writelines(e)
                    continue
            else:
                try:
                    result = test_creative_response(
                        row[f"creative_scenario_round_{round}"],
                        0,  # prompt_idx
                        llm,
                    )
                    print(result)
                except Exception as e:
                    with open(config["logFile"], "w") as log:
                        print(e)
                        log.writelines(e)
                    continue

            if demographics_file is not None:
                ai_responses = pd.concat(
                    [
                        ai_responses,
                        pd.DataFrame(
                            {
                                f"creative_scenario_round_{round}": row[
                                    f"creative_scenario_round_{round}"
                                ],
                                f"creative_response_round_{round}": result,
                                f"ethnicity": participant["Q15"].values[0],
                                "gender": participant["Q14"].values[0],
                                "industry": participant["Q24"].values[0],
                                "title": participant["Q23"].values[0],
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )
            else:
                ai_responses = pd.concat(
                    [
                        ai_responses,
                        pd.DataFrame(
                            {
                                f"creative_scenario_round_{round}": row[
                                    f"creative_scenario_round_{round}"
                                ],
                                f"creative_response_round_{round}": result,
                            },
                            index=[0],
                        ),
                    ],
                    ignore_index=True,
                )

    ai_responses.to_json(f"{response_file_name}_round_{round}.json", orient="records")


# test prompt X number of times, and save in df
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--frequency_penalty", type=float)
    parser.add_argument("--presence_penalty", type=float)
    parser.add_argument("--prompt_idx", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--demographics_file", type=str, default=None)
    parser = parser.parse_args()
    model_name = parser.model_name
    temperature = parser.temperature
    max_tokens = parser.max_tokens
    top_p = parser.top_p
    frequency_penalty = parser.frequency_penalty
    presence_penalty = parser.presence_penalty

    if model_name == "gpt-4" or model_name == "gpt-3.5-turbo":
        model_kwargs = {
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        llm = ChatOpenAI(
            model_name=model_name,
            openai_api_key=OPENAI_KEY,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
        )
    else:
        model_kwargs = {
            "top_p": top_p,
            "temperature": temperature,
            "device_map": "auto",
            # "torch_dtype": torch.bfloat16, # don't use with 8 bit mode
        }
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, load_in_8bit=True, **model_kwargs
        )
        pipeline = hf_pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            batch_size=1,
            max_new_tokens=max_tokens,
            model_kwargs=model_kwargs,
        )
        llm = HuggingFacePipeline(pipeline=pipeline)

    create_scenario_responses(
        llm,
        parser.input_file,
        parser.demographics_file,
    )
