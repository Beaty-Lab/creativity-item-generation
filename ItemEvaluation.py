# TODO: need to completely rewrite this when we're ready to integrate the code
# Follow the same refactoring as item gen
# Using LLMs to evaluate the quality of CPS items
import time
import torch
import re
import json as js
import pandas as pd
import config

# OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser

# API key stored in key.py, and should NOT be committed
# TODO: add support for other api models called directly from this script
from key import OPENAI_API_KEY
from tqdm import tqdm


class CreativityScenarioItemEvaluationParser(BaseOutputParser):
    # the output should be formatted as json
    @staticmethod
    def parse(text: str) -> dict:
        #
        def fix_json(s: str):
            s = s[next(idx for idx, c in enumerate(s) if c in "{[") :]
            try:
                return js.loads(s)
            except js.JSONDecodeError as e:
                return js.loads(s[: e.pos])

        return js.dumps(fix_json(text))


def test_creative_problem_eval(prompt_idx: int, scenario: str, llm):
    # TODO: update with exemplars rated by humans using the new scale.
    prompts = item_eval_prompts

    output = llm.predict_messages(prompts[prompt_idx])

    output = CreativityScenarioItemEvaluationParser().parse(output.content)

    return output


def evaluate_scenarios(
    prompt_idx: int,
    output_file: str,
    model_name: str,
    llm,
    round: int,
    scenario_col: str,
    itemEvalOutputFile: str,
    itemGenOutputFile: str,
    itemEvalFrequencyPenalty: float,
    itemEvalPresencePenalty: float,
    itemEvalMaxTokens: int,
    itemEvalTemperature: float,
    itemEvalTopP: float,
):

    scenarios = pd.read_json(
        itemGenOutputFile,
    )
    scenarios[f"ratings_round_{round}"] = ""
    scenarios["item_eval_frequency_penalty"] = itemEvalFrequencyPenalty
    scenarios["item_eval_presence_penalty"] = itemEvalPresencePenalty
    scenarios["item_eval_max_tokens"] = itemEvalMaxTokens
    scenarios["item_eval_temperature"] = itemEvalTemperature
    scenarios["item_eval_top_p"] = itemEvalTopP
    scenarios["Evaluator"] = model_name
    for index, row in tqdm(scenarios.iterrows(), total=scenarios.shape[0]):
        evaluation = "None"
        time.sleep(10)
        for i in range(6):  # retry a maximum of x times
            try:
                evaluation = test_creative_problem_eval(
                    prompt_idx,
                    row[scenario_col],  # "creative_scenario_round_i"
                    llm,
                )
            except Exception as e:
                with open(config["logFile"], "w") as log:
                    print(e)
                    log.writelines(e)
                evaluation = "None"
                continue
            if evaluation != "None":
                break

        scenarios.at[index, f"ratings_round_{round}"] = evaluation

    # drop rows that failed quality control metrics
    scenarios = scenarios[scenarios[f"ratings_round_{round}"] != "None"]
    scenarios.to_json(
        itemEvalOutputFile,
    )
