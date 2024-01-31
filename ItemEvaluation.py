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
                    3. Balance relationship problems with task/goal-oriented problems. Scenarios that focus purely on relationship issues alone limit solutions. It is permissible to include relationship focused constraints, if they are balanced with more objective or goal-oriented ones.
                    4. Ensure consistent reading level across scenarios. Avoid unnecessarily complex vocabulary.
                    5. Orient scenarios towards adults. Avoid student/school settings.
                    6. Provide enough background details so the current dilemma is understandable. Scenarios should give test-takers enough to work with to develop multiple possible solutions.
                    7. Competing goals should be more complex than just preferences or feelings. Include tangible stakes, goals, or external pressures.
                    8. Do not focus solely on emotions like jealousy or relationships. These limit viable solutions. It is permissible to include emotionally focused constraints, if they are balanced with more objective or goal-oriented ones.
                    9. Avoid scenarios requiring niche knowledge or experience that may not be equally familiar to all test takers. Scenarios should be universally understandable, and deal with situations and challenges that the large majority of people are likely familiar with. Universal experiences include going to school, being in a relationship, spending time with friends, etc. More niche scenarios could, for example, deal with a hobby only a small fraction of people would participate in, or a life experience present in only a minority of the population. Please err on the side of caution here; if a scenario seems like it would not be relatable to the overwhelming majority participants, it's better to give a lower rating even if you aren't absolutely sure.
                    10. Do NOT include controversial or emotionally charged topics in the scenarios; these may sway participate responses and result in social desirability biases. Examples of controversial topics include abortion and marijuana use; these and similar topics should NOT be included in scenarios.
                    11. The best scenarios allow room for a wide range of creative responses beyond the obvious, with interpersonal issues as well as task/goal-related pressures.

                    Produce a numerical rating on a scale of 1-3 evaluating the quality of the scenario along multiple dimensions:

                    Complexity
                    1 = No competing demands
                    2 = Some competing demands
                    3 = Many competing demands

                    Open-endedness
                    1 = Only one possible solution
                    2 = Some possible solutions
                    3 = Many possible solutions

                    Constraints
                    1 = No constraints or goals
                    2 = Some constraints or goals
                    3 = Many constraints or goals

                    Relationships
                    1 = No relationship focused constraints
                    2 = Some relationship focused constraints
                    3 = Many relationship focused constraints

                    Accessibility
                    1 = Significant specialized experience needed
                    2 = Some specialized experience needed
                    3 = No specialized experience needed

                    Emotional Focus
                    1 = No emotionally focused constraints
                    2 = Some emotionally focused constraints
                    3 = Many emotionally focused constraints

                    Controversial
                    1 = Many constraints involving controversial topics
                    2 = Some constraints involving controversial topics
                    3 = No constraints involving controversial topics

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
        scenarios = scenarios[scenarios["ratings_round_1"] != ""]
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
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--frequency_penalty", type=float)
    parser.add_argument("--presence_penalty", type=float)
    parser.add_argument("--prompt_idx", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--output_file", type=str)  # the prefix to the output file name
    parser.add_argument("--round", type=int)
    parser.add_argument("--scenario_col", type=str)
    parser = parser.parse_args()
    try:
        task = parser.task
        model_name = parser.model_name
        temperature = 0 # output should be as close to deterministic as possible
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
            parser.scenario_col,
        )
    elif task == "evaluate_consequences":
        print("Consequences eval not implemented!")
        exit(-1)
    else:
        print("Not a valid task name!")
        exit(-1)
