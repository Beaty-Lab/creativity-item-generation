import time
import torch
import bitsandbytes
import re
import pandas as pd

# HF
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import pipeline as hf_pipeline


from langchain.schema import BaseOutputParser
from langchain.prompts.chat import ChatPromptTemplate

# API key stored in key.py, and should NOT be committed
# from key import key
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
        ]
        creative_scenario_generation_prompt = ChatPromptTemplate.from_messages(
            scenario_base_prompts[scendario_prompt_idx]
        )
        return creative_scenario_generation_prompt


class CreativityScenarioResponseParser(BaseOutputParser):
    # TODO: for the next round of items after feedback, get rid of everything generated before and after the start of the original item
    @staticmethod
    def parse(text: str) -> str:
        text = text.strip("\n").strip(" ")
        # Remove intervening newlines
        text = re.sub("\n", "", text)
        text = re.sub("\t", "", text)

        return text


def test_creative_response(
    problem,
    prompt_idx: int,
    llm,
):
    prompt = PromptGenerator.make_creative_scenario_response_prompt(
        prompt_idx
    )  # the prompt type

    chain = prompt | llm
    result = chain.invoke({"creative_scenario": problem})
    result = CreativityScenarioResponseParser.parse(result)

    return result


def create_scenario_responses(
    prompt_idx: int,
    llm,
    input_file: str = None,
):
    input_file = pd.read_csv(
        input_file,
    )
    ai_responses = pd.DataFrame(
        columns=["Problem", "Dataset", "ProblemID", "set", "response"]
    )
    for index, row in tqdm(input_file.iterrows(), total=input_file.shape[0]):
        # generate 30 responses to each scenario
        print(row["Problem"])
        for i in tqdm(range(30)):
            result = test_creative_response(
                row["Problem"],
                prompt_idx,
                llm,
            )
            ai_responses = pd.concat(
                [
                    ai_responses,
                    pd.DataFrame(
                        {
                            "Problem": row["Problem"],
                            "response": result,
                            "Dataset": row["Dataset"],
                            "ProblemID": row["ProblemID"],
                            "set": row["set"],
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

    ai_responses.to_csv(
        "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/ai_outputs.csv",
    )


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
    parser = parser.parse_args()
    model_name = parser.model_name
    temperature = parser.temperature
    max_tokens = parser.max_tokens
    top_p = parser.top_p
    frequency_penalty = parser.frequency_penalty
    presence_penalty = parser.presence_penalty
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
        parser.prompt_idx,
        llm,
        parser.input_file,
    )
