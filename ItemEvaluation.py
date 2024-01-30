# Using LLMs to evaluate the quality of CPS items
import time
import torch
import re
import json as js
import pandas as pd

# OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.prompts.chat import ChatPromptTemplate

# API key stored in key.py, and should NOT be committed
from key import key
from tqdm import tqdm
from random import randint
from argparse import ArgumentParser
from nltk import word_tokenize


# class for storing and manipulating prompts
class PromptGenerator:
    @staticmethod
    def make_creative_scenario_evaluation_prompt(evaluation_prompt_idx: int):
        scenario_base_prompts = [
            [
                (
                    "system",
                    """You are an author charged with evaluating scenarios for short stories. Given a scenario, you will evaluate the quality of the scenario in terms of how well it obeys these criteria:

                    1. Scenarios should present complex situations with more than just two competing demands or considerations. Avoid framing as clear-cut dilemmas.
                    2. Include details that allow for more unique and creative responses beyond the obvious. For example, additional characters or constraints that test-takers can draw from.
                    3. Balance relationship problems with task/goal-oriented problems. Relationships issues alone limit solutions.
                    4. Ensure consistent reading level across scenarios. Avoid unnecessarily complex vocabulary.
                    5. Orient scenarios towards adults. Avoid student/school settings.
                    6. Provide enough background details so the current dilemma is understandable. Scenarios should give test-takers enough to work with.
                    7. Competing goals should be more complex than just preferences or feelings. Include tangible stakes, goals, or external pressures.
                    8. Do not focus solely on emotions like jealousy or relationships. These limit viable solutions.
                    9. Avoid scenarios requiring niche knowledge that may not be equally familiar to all test-takers.
                    10. The best scenarios allow room for a wide range of creative responses beyond the obvious, with interpersonal issues as well as task/goal-related pressures.

                    Produce a numerical rating on a scale of 1-3 evaluating the quality of the scenario along multiple dimensions:

                    Complexity
                    1 = Fewer than two competing demands
                    2 = Exactly two competing demands
                    3 = More than two competing demands

                    Open-endedness
                    1 = Few unique responses possible
                    2 = Some opportunity for creativity
                    3 = Allows many creative responses

                    Constraints
                    1 = No real constraints or goals
                    2 = Some external pressures
                    3 = Multiple complex constraints

                    Relationships
                    1 = Purely interpersonal
                    2 = Mix of relationships and tasks
                    3 = Mostly task/goal-oriented

                    Accessibility
                    1 = Requires niche knowledge
                    2 = Mostly relatable
                    3 = Universally understandable

                    Emotional Focus
                    1 = Feelings or relationship focused
                    2 = Neutral
                    3 = Purely task/goal-focused

                    Provide your response in JSON.""",
                ),
                (
                    "human",  # 1
                    """Scenario:
                    {scenario}

                    ###

                    Ratings:""",
                ),
            ],
        ]
        creative_scenario_generation_prompt = ChatPromptTemplate.from_messages(
            scenario_base_prompts[evaluation_prompt_idx]
        )
        return creative_scenario_generation_prompt


class CreativityScenarioItemEvaluationParser(BaseOutputParser):
    # the output should be formatted as json
    def parse(self, text: str) -> dict:
        try:
            js.loads(text)
        except Exception:
            print("Json output failed to load, trying next item...")
            return None

        return text


def test_creative_problem_eval(prompt_idx: int, scenario: str, llm):
    prompt = PromptGenerator.make_creative_scenario_evaluation_prompt(
        prompt_idx
    )  # the prompt type

    chain = prompt | llm | CreativityScenarioItemEvaluationParser()
    result = chain.invoke({"scenario": scenario})

    return result


def evaluate_scenarios(
    prompt_idx: int, output_file: str, model_name: str, llm, round: int, scenario_col: str
):
    if round == 1:
        try:
            scenarios = pd.read_csv(
                f"/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/{output_file}.tsv",
                sep="\t",
                index_col=0,
            )
        except Exception:
            scenarios = pd.read_json(
                f"/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/{output_file}.json",
            )
        scenarios["ratings_round_1"] = ""
        scenarios["Evaluator"] = model_name
        for index, row in tqdm(scenarios.iterrows(), total=scenarios.shape[0]):
            time.sleep(2)
            evaluation = test_creative_problem_eval(
                prompt_idx,
                row[scenario_col], # "creative_scenario_without_feedback"
                llm,
            )
            if evaluation == None:
                continue
            scenarios.at[index, "ratings_round_1"] = evaluation

        # drop rows that failed quality control metrics
        scenarios = scenarios[scenarios["ratings_round_2"] != ""]
        scenarios.to_json(
            f"/home/aml7990/Code/creativity-item-generation/outputs/with_eval_scores/{output_file}.json",
        )
    elif round == 2:
        try:
            scenarios = pd.read_csv(
                f"/home/aml7990/Code/creativity-item-generation/outputs/with_eval_scores/{output_file}.tsv",
                sep="\t",
                # index_col=0,
            )
        except Exception:
            scenarios = pd.read_json(
                f"/home/aml7990/Code/creativity-item-generation/outputs/with_eval_scores/{output_file}.json",
            )
        scenarios["ratings_round_2"] = ""
        scenarios["Evaluator"] = model_name
        for index, row in tqdm(scenarios.iterrows(), total=scenarios.shape[0]):
            time.sleep(2)
            evaluation = test_creative_problem_eval(
                prompt_idx,
                row[scenario_col], # "creative_scenario_with_feedback"
                llm,
            )
            if evaluation == None:
                continue
            scenarios.at[index, "ratings_round_2"] = evaluation

        # drop rows that failed quality control metrics
        scenarios = scenarios[scenarios["ratings_round_2"] != ""]
        scenarios.to_json(
            f"/home/aml7990/Code/creativity-item-generation/outputs/with_eval_scores/{output_file}.tsv",
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)  # the LLM used to evaluate items
    parser.add_argument("--task", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--frequency_penalty", type=float)
    parser.add_argument("--presence_penalty", type=float)
    parser.add_argument("--prompt_idx", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--output_file", type=str)  # the prefix to the output file name
    parser.add_argument("--round", type=int)
    parser = parser.parse_args()
    try:
        task = parser.task
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
                openai_api_key=key,
                temperature=temperature,
                max_tokens=max_tokens,
                model_kwargs=model_kwargs,
            )

        else:
            print("Only OpenAI models are supporting for evaluating items.")

    except Exception:
        print("Model failed to initialize. Please check your API key.")
        exit(-1)
    if task == "evaluate_CPS":
        evaluate_scenarios(
            parser.prompt_idx,
            parser.output_file,
            parser.model_name,
            llm,
            parser.round,
        )
    elif task == "evaluate_consequences":
        print("Consequences eval not implemented!")
        exit(-1)
    else:
        print("Not a valid task name!")
        exit(-1)
