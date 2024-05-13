import time
import numpy as np
import re
import pandas as pd
from config import config


from langchain.prompts.chat import _convert_to_message
from langchain.schema import BaseOutputParser, StrOutputParser, OutputParserException
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import RetryOutputParser
from langchain.prompts.chat import ChatPromptTemplate


from tqdm import tqdm
from readability import Readability
from nltk import word_tokenize


# class for storing and manipulating prompts
"""
Generates a prompt for creative scenario generation based on a given word list.

Parameters:
    word_list (str): A list of 4 words, consisting of 2 names, a place, and an action.

Returns:
    str: The generated prompt for creative scenario generation.
"""
# class PromptGenerator:
#     @staticmethod
#     def make_creative_scenario_generation_prompt(scendario_prompt_idx: int):
#         scenario_base_prompts = item_gen_prompts
#         creative_scenario_generation_prompt = ChatPromptTemplate.from_messages(
#             scenario_base_prompts[scendario_prompt_idx],
#         )
#         return creative_scenario_generation_prompt


# class ConsequencesItemParser(BaseOutputParser):
#     def parse(self, text: str) -> dict:
#         js_output = {
#             "model_name": model_name,
#             "temperature": temperature,
#             "max_tokens": max_tokens,
#             "top_p": top_p,
#             "frequency_penalty": frequency_penalty,
#             "presence_penalty": presence_penalty,
#             "output": text,
#             "item_type": "consequences",
#         }
#         return js_output


# class CreativeWordlistItemParser(BaseOutputParser):
#     def parse(self, text: str) -> dict:
#         text = text.split("\n\n")
#         text = [re.sub(r"([0-9.])+", "", t) for t in text]
#         text = [t.strip("\n").strip(" ") for t in text]
#         js_output = {
#             "model_name": model_name,
#             "temperature": temperature,
#             "max_tokens": max_tokens,
#             "output": text,
#             "top_p": top_p,
#             "frequency_penalty": frequency_penalty,
#             "presence_penalty": presence_penalty,
#             "item_type": "creative_scenario",
#         }
#         return js_output


# test the chain
# TODO: refactor, move tests into task class
# def test_consequences():
#     prompt = PromptGenerator.make_consequence_prompt()
#     chain = prompt | llm | ConsequencesItemParser()
#     result = chain.invoke(
#         {"": ""}
#     )  # invoking a chain requires passing an input for the template, which can be a blank dict
#     return result


"""
    Item generation tracks both the current round and the output file
    Items are always saved out at the end of every iteration, and loaded from memory on the next one
"""


def create_scenarios(
    item_gen_prompt: str,
    model_name: str,
    llm,
    round,
    max_tokens: int,
    presence_penalty: float,
    frequency_penalty: float,
    temperature: float,
    top_p: float,
    itemGenOutputFile: str,
    numItemGenerationAttempts: int,
    input_file: str = None,
    wordlist_file: str = None,
    item_shots: list = None,
    task_parser=None,
):
    # when true, will use add new items to an existing file
    if input_file != None and round >= 1:
        assert item_shots != None

        input_file = pd.read_json(input_file)
        if f"ratings_round_{round-1}" not in input_file.columns:
            input_file[f"ratings_round_{round-1}"] = None

        input_file[f"creative_scenario_round_{round}"] = ""
        input_file[f"prompt_round_{round}"] = ""
        for index, row in tqdm(input_file.iterrows(), total=input_file.shape[0]):
            result = "None"
            if (
                model_name == "gpt-4"
                or model_name == "gpt-3.5-turbo"
                or model_name == "google"
                or model_name == "claude-3"
            ):
                time.sleep(3)

            # scenario_names = row["word_list"].split(",")[::2]
            # scenario_names = [
            #     re.sub(r"([0-9]{1}\.)", "", s).strip() for s in scenario_names
            # ]
            # generation may fail due to google API filters
            try:
                result, prompt = task_parser.RunItemGeneration(
                    row["word_list"],
                    item_gen_prompt,
                    llm,
                    previous_llm_output=row[f"creative_scenario_round_{round-1}"],
                    ratings_from_file=row[f"ratings_round_{round-1}"],
                    item_shots=item_shots,
                    numAttempts=numItemGenerationAttempts,  # keep on generating scenarios until the model passes all quality control checks
                )
                print(result)

            except Exception as e:
                with open(config["logFile"], "a") as log:
                    print(e)
                    log.writelines(str(e) + "\n")
                result = np.nan
                prompt = np.nan
                continue

            input_file.at[index, f"creative_scenario_round_{round}"] = result
            input_file.at[index, f"prompt_round_{round}"] = prompt

        # drop rows that failed quality controls
        # TODO: over multiple rounds, there could be many scenarios for the same wordlist + topic
        # if at any point one of those comes back null, so long as prior iterations weren't null
        # the row SHOULD NOT be deleted!
        # change the drop to account for this, will need some careful engineering
        input_file = input_file[
            input_file[f"creative_scenario_round_{round}"] != "None"
        ]
        input_file = input_file[input_file[f"creative_scenario_round_{round}"] != ""]
        input_file = input_file[input_file[f"creative_scenario_round_{round}"] != None]
        input_file.dropna(subset=f"creative_scenario_round_{round}", inplace=True)
        input_file.to_json(itemGenOutputFile, orient="records")
        with open(config["logFile"], "a") as log:
            print(f"Item gen finished, total items: {len(input_file)}")
            log.writelines(f"Item gen finished, total items: {len(input_file)}\n")
    elif input_file == None and round == 0:
        # path for a fresh round of item generation
        # TODO: refactor to generalize
        # think about how to make this work even if the wordlist step isn't required
        if wordlist_file != None:
            wordlists = task_parser.prep_wordlist(wordlist_file, round)
            wordlists_with_s = pd.DataFrame()
            for index, row in tqdm(wordlists.iterrows(), total=wordlists.shape[0]):
                result = "None"  # keep on generating scenarios until the model passes all quality control checks
                if (
                    model_name == "gpt-4"
                    or model_name == "gpt-3.5-turbo"
                    or model_name == "google"
                    or model_name == "claude-3"
                ):
                    time.sleep(5)
                # grab just the names in the wordlist, need for preprocessing
                scenario_names = row["word_list"].split(",")[::2]
                scenario_names = [
                    re.sub(r"([0-9]{1}\.)", "", s).strip() for s in scenario_names
                ]
                # generation may fail due to google API filters
                try:
                    result, prompt = task_parser.RunItemGeneration(
                        row["word_list"],
                        item_gen_prompt,
                        llm,
                        numAttempts=numItemGenerationAttempts,  # keep on generating scenarios until the model passes all quality control checks
                    )
                    print(result)
                except Exception as e:
                    with open(config["logFile"], "a") as log:
                        print(e)
                        log.writelines(str(e) + "\n")
                    result = np.nan
                    prompt = np.nan
                    continue

                new_scenario = pd.DataFrame(
                    {
                        f"creative_scenario_round_{round}": result,
                        f"prompt_round_{round}": prompt,
                        "item_gen_model_name": model_name,
                        "item_gen_max_tokens": max_tokens,
                        "item_gen_presence_penalty": presence_penalty,
                        "item_gen_frequency_penalty": frequency_penalty,
                        "item_gen_temperature": temperature,
                        "item_type": row["item_type"],
                        "item_gen_top_p": top_p,
                        "word_list": row["word_list"],
                    },
                    index=[0],
                )
                wordlists_with_s = pd.concat((wordlists_with_s, new_scenario))

            with open(config["logFile"], "a") as log:
                print(f"Item gen finished, total items {len(wordlists_with_s)}")
                log.writelines(
                    f"Item gen finished, total items {len(wordlists_with_s)}\n"
                )
            # drop rows that failed quality control metrics
            wordlists_with_s.reset_index(drop=True, inplace=True)
            wordlists_with_s = wordlists_with_s[
                wordlists_with_s[f"creative_scenario_round_{round}"] != "None"
            ]
            wordlists_with_s = wordlists_with_s[
                wordlists_with_s[f"creative_scenario_round_{round}"] != ""
            ]
            wordlists_with_s = wordlists_with_s[
                wordlists_with_s[f"creative_scenario_round_{round}"] != None
            ]
            wordlists_with_s.dropna(
                subset=f"creative_scenario_round_{round}", inplace=True
            )
            wordlists_with_s.to_json(
                itemGenOutputFile,
                orient="records",
            )
        elif wordlist_file == None:
            pass
            # TODO: path for consequences like tasks that don't
            # have a set of constriants for the initial generation
            # in this case, the same prompt is just repeated x times to build the item pool
            # we must also drop duplicate items, we don't have the wordlist to ground this.
    else:
        print("Unsupported combination of arguments!")
