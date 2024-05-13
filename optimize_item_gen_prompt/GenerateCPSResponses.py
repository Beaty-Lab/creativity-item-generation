import time
import torch
import bitsandbytes
import re
import pandas as pd
from config import config

from transformers import pipeline as hf_pipeline
from key import OPENAI_KEY
import time


from langchain.schema import BaseOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

# API key stored in key.py, and should NOT be committed
from tqdm import tqdm


def create_scenario_responses(
    item_response_gen_prompt,
    llm,
    round,
    input_file_name: str,
    demographics_file: str,
    response_file_name: str,
    model_name: str,
    num_item_responses: int,
    prompt_idx: int,
    task_parser,
):
    input_file = pd.read_json(input_file_name)

    if demographics_file is not None:
        demographics_file = pd.read_csv(demographics_file, index_col=0)

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
        for i in tqdm(range(num_item_responses)):
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
                    if prompt_idx == 1:  # demographics (biased)
                        result = task_parser.RunItemResponseGeneration(
                            row[f"creative_scenario_round_{round}"],
                            item_response_gen_prompt,
                            prompt_idx,  # prompt_idx
                            llm,
                            participant["Q15"].values[0],  # ethnicity
                            participant["Q14"].values[0],  # gender
                            participant["Q24"].values[0],  # industry
                            participant["Q23"].values[0],  # title
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                    elif prompt_idx == 2:
                        result = task_parser.RunItemResponseGeneration(
                            row[f"creative_scenario_round_{round}"],
                            item_response_gen_prompt,
                            prompt_idx,  # prompt_idx
                            llm,
                            None,
                            None,
                            None,
                            None,
                            participant["FirstName"].values[0],
                            participant["LastName"].values[0],
                            participant["Occupation"].values[0],
                            participant["Field"].values[0],
                            participant["Psychometric"].values[
                                0
                            ],  # psychometric traits
                        )
                    print(result)
                except Exception as e:
                    with open(config["logFile"], "a") as log:
                        print(e)
                        log.writelines(str(e) + "\n")
                    continue
            else:
                try:
                    result = task_parser.RunItemResponseGeneration(
                        row[f"creative_scenario_round_{round}"],
                        item_response_gen_prompt,
                        prompt_idx,  # prompt_idx
                        llm,
                    )
                    print(result)
                except Exception as e:
                    with open(config["logFile"], "a") as log:
                        print(e)
                        log.writelines(str(e) + "\n")
                    continue

            if demographics_file is not None:
                if prompt_idx == 1:
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
                elif prompt_idx == 2:
                    ai_responses = pd.concat(
                        [
                            ai_responses,
                            pd.DataFrame(
                                {
                                    f"creative_scenario_round_{round}": row[
                                        f"creative_scenario_round_{round}"
                                    ],
                                    f"creative_response_round_{round}": result,
                                    f"FirstName": participant["FirstName"].values[0],
                                    "LastName": participant["LastName"].values[0],
                                    "Occupation": participant["Occupation"].values[0],
                                    "Field": participant["Field"].values[0],
                                    "Psychometric": participant["Psychometric"].values[
                                        0
                                    ],
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
