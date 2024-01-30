import time
import torch
import bitsandbytes
import re
import pandas as pd

# OpenAI
from langchain.chat_models import ChatOpenAI

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
from key import key
from tqdm import tqdm
from readability import Readability
from random import randint
from argparse import ArgumentParser
from nltk import word_tokenize


# class for storing and manipulating prompts
class PromptGenerator:
    @staticmethod
    def make_consequence_prompt():
        consequences_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a scenario writer."),
                (
                    "human",
                    """Think of a scenario the would change the way human life or the world works. The scenario should alter people's daily lives in important ways. Please describe only the scenario, and don't hint at any potential implications of the scenario. Please describe this scenario in 12 words at most.

        ###

        Scenario:""",
                ),
            ]
        )
        return consequences_prompt

    # NOTE: few shots are currently baked into prompt
    @staticmethod
    def make_creative_scenario_wordlist_generation_prompt(wordlist_prompt_idx: int):
        prompts = [
            [  # 1
                ("system", "You are a helpful assistant."),
                (
                    "human",
                    """Create a list of 4 words. In the list, include 2 human names, a place, and an action. You don't need to use names of actual people or places, but only use words which relate to the experiences of typical people; for example, do not include a word like "starship" for the place since no one alive today has been on a starship. You should list words in exactly this order: name, place, name, action.

                    ###

                    Example word lists: 

                    1. Becky, pizzeria, Jim, theft

                    2. Wentworth, Acme Company, Scott, employment

                    ###
                    New word list:
                    1. """,
                ),
            ],
            [  # 2
                (
                    "system",
                    'You are an author tasked with coming up with scenarios for a short story. Create a list of 4 words. In the list, include 2 human names, a place, and an action. You don\'t need to use names of actual people or places, but only use words which relate to the experiences of typical people; for example, do not include a word like "starship" for the place since no one alive today has been on a starship. Each entry in the list should consist of only a single word You should list words in exactly this order: name, place, name, action.',
                ),
                (
                    "human",
                    "Create 10 wordlists, make sure to use a different action each time. Separate wordlists by two newlines, do not number them.",
                ),
            ],
            [  # 3
                (
                    "system",
                    'You are an author tasked with coming up with scenarios for a short story. Create a list of 5 words. In the list, include 3 human names, a place, and an action. You don\'t need to use names of actual people or places, but only use words which relate to the experiences of typical people; for example, do not include a word like "starship" for the place since no one alive today has been on a starship. Each entry in the list should consist of only a single word You should list words in exactly this order: name, place, name, action, name.',
                ),
                (
                    "human",
                    "Create 10 wordlists, make sure to use a different action each time. Separate wordlists by two newlines, do not number them.",
                ),
            ],
        ]
        creative_scenario_wordlist_generation_prompt = ChatPromptTemplate.from_messages(
            prompts[wordlist_prompt_idx]
        )
        return creative_scenario_wordlist_generation_prompt

    """
    Generates a prompt for creative scenario generation based on a given word list.

    Parameters:
        word_list (str): A list of 4 words, consisting of 2 names, a place, and an action.

    Returns:
        str: The generated prompt for creative scenario generation.
    """

    @staticmethod
    def make_creative_scenario_generation_prompt(scendario_prompt_idx: int):
        # keep track of the different prompts
        scenario_base_prompts = [
            [
                ("system", "You are a scenario writer."),
                (
                    "human",  # 1
                    """You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of 3 different scenarios that involve all the words. These scenarios should involve dilemmas that one of the named people from the list needs to solve. These dilemmas should be relatable to an average person, and they should also have no clear solution. Include as many details about the situations as you can, and try to keep your scenarios at about a paragraph each. Make sure what you write is not too difficult to read: avoid complex jargon wherever possible.

        ###

        Word list:
        {word_list}

        ###

        Scenario:
        """,
                ),
            ],
            [
                ("system", "You are a screenwriter."),
                (
                    "human",  # 2
                    """You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of 3 different scenarios that involve all the words. This scenarios should involve dilemmas that one of the named people from the list needs to solve. These dilemmas should be relatable to an average person, and they should also have no clear solution. Include as many details about the situations as you can, and try to keep your scenarios at about a paragraph each. Make sure what you write is not too difficult to read: avoid complex jargon wherever possible.

        ###

        Word list:
        {word_list}

        ###

        Scenario:
        """,
                ),
            ],
            [
                ("system", "You are a young adult novelist."),
                (
                    "human",  # 3
                    """You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of 3 different scenarios that involve all the words. This scenarios should involve dilemmas that one of the named people from the list needs to solve. These dilemmas should be relatable to an average person, and they should also have no clear solution. Include as many details about the situations as you can, and try to keep your scenarios at about a paragraph each. Make sure what you write is not too difficult to read: avoid complex jargon wherever possible.

        ###

        Word list:
        {word_list}

        ###

        Scenario:
        """,
                ),
            ],
            [
                ("system", "You are a film director."),
                (
                    "human",  # 4
                    """You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of 3 different scenarios that involve all the words. This scenarios should involve dilemmas that one of the named people from the list needs to solve. These dilemmas should be relatable to an average person, and they should also have no clear solution. Include as many details about the situations as you can, and try to keep your scenarios at about a paragraph each. Make sure what you write is not too difficult to read: avoid complex jargon wherever possible.

        ###

        Word list:
        {word_list}

        ###

        Scenario:
        """,
                ),
            ],
            [
                ("sytem", "You are a YouTube content creator."),
                (
                    "human",  # 5
                    """You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of 3 different scenarios that involve all the words. This scenarios should involve dilemmas that one of the named people from the list needs to solve. These dilemmas should be relatable to an average person, and they should also have no clear solution. Include as many details about the situations as you can, and try to keep your scenarios at about a paragraph each. Make sure what you write is not too difficult to read: avoid complex jargon wherever possible.

        ###

        Word list:
        {word_list}

        ###

        Scenario:
        """,
                ),
            ],
            [  # 6
                (
                    "system",
                    """You are an author tasked with producing scenarios for a short story. You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of a scenario that involves all the words. This scenario should involve a dilemma that one of the named people from the list, the main character, needs to solve. Here is a list of rules you should follow when writing the scenario:

                    1. The dilemma should be relatable to an average college student and must involve scenarios that a typical college student might need to confront.
                    2. Do not suggest any possible solution to the dilemma in the scenario, avoid phrases like "She is torn between...", "On the one hand...", "On the other hand...", or "He is not sure whether he should do X or Y..." as these may imply possible solutions to the dilemma. The scenario will be given to another writer as part of a writing prompt, and we do not want to bias their writing by suggesting how the story will unfold. Focus only on describing the dilemma and its significance to the main character.
                    3. Include as many details about the scenario as you can.
                    4. Respond in at least 8 sentences and include as many details as possible.
                    5. In the last sentence, state something like "Z does not know what to do.", where Z is the name of the main character.
                    6. Make sure what you write is not too difficult to read; avoid complex jargon wherever possible. It should be easy for someone with a high school education to read.
                    7. Avoid scenarios that deal with either jealousy in relationships or involve ethical or moral dilemmas.
                    8. Avoid scenarios that require specific domain knowledge or experience to solve. Remember, the dilemma needs to be relatable to an average college student.
                    9. The scenario should be open-ended and have more than 2 possible solutions. The scenario should also be ambiguous and have no solution that is clearly better or more obvious than the others. Remember: do not suggest any possible solution in the scenario.
                    10. Your scenario should involve higher stakes than the personal preferences of the main character; there should be clear repercussions from any potential action, such that solving the dilemma requires critical thinking.""",
                ),
                (
                    "human",
                    """Word list:
        {word_list}

        ###

        Scenario:""",
                ),
            ],
            [  # 7, 5 word scenario
                (
                    "system",
                    """You are an author tasked with producing scenarios for a short story. You will be given a list of 5 words, consisting of 3 names, a place, and an action. Using ONLY these words, think of a scenario that involves all the words. This scenario should involve a dilemma that one of the named people from the list, the main character, needs to solve. Here is a list of rules you should follow when writing the scenario:

                    1. The dilemma should be relatable to an average college student and must involve scenarios that a typical college student might need to confront.
                    2. Do not suggest any possible solution to the dilemma in the scenario, avoid phrases like "She is torn between...", "On the one hand...", "On the other hand...", or "He is not sure whether he should do X or Y..." as these may imply possible solutions to the dilemma. The scenario will be given to another writer as part of a writing prompt, and we do not want to bias their writing by suggesting how the story will unfold. Focus only on describing the dilemma and its significance to the main character.
                    3. Include as many details about the scenario as you can.
                    4. Respond in at least 8 sentences and in a single paragraph.
                    5. In the last sentence, state something like "Z does not know what to do.", where Z is the name of the main character.
                    6. Make sure what you write is not too difficult to read; avoid complex jargon wherever possible. It should be easy for someone with a high school education to read.
                    7. Avoid scenarios that deal with either jealousy in relationships or involve ethical or moral dilemmas.
                    8. Avoid scenarios that require specific domain knowledge or experience to solve. Remember, the dilemma needs to be relatable to an average college student.
                    9. The scenario should be open-ended and have more than 2 possible solutions. The scenario should also be ambiguous and have no solution that is clearly better or more obvious than the others. Remember: do not suggest any possible solution in the scenario.
                    10. Your scenario should involve higher stakes than the personal preferences of the main character; there should be clear repercussions from any potential action, such that solving the dilemma requires critical thinking.""",
                ),
                (
                    "human",
                    """Word list:
        {word_list}

        ###

        Scenario:""",
                ),
            ],
            [  # 8, including topic of dilemnia
                (
                    "system",
                    """You are an author tasked with producing scenarios for a short story. You will be given a list of 5 words, consisting of 3 names, a place, and an action. Using ONLY these words, think of a scenario that involves all the words. This scenario should involve a dilemma that one of the named people from the list, the main character, needs to solve. Here is a list of rules you should follow when writing the scenario:

                    1. The dilemma should be relatable to an average college student and must involve scenarios that a typical college student might need to confront.
                    2. Do not suggest any possible solution to the dilemma in the scenario, avoid phrases like "She is torn between...", "On the one hand...", "On the other hand...", or "He is not sure whether he should do X or Y..." as these may imply possible solutions to the dilemma. The scenario will be given to another writer as part of a writing prompt, and we do not want to bias their writing by suggesting how the story will unfold. Focus only on describing the dilemma and its significance to the main character.
                    3. Include as many details about the scenario as you can.
                    4. Respond in at least 8 but no more than 12 sentences and in a single paragraph.
                    5. In the last sentence, state something like "Z does not know what to do.", where Z is the name of the main character.
                    6. Make sure what you write is not too difficult to read; avoid complex jargon wherever possible. It should be easy for someone with a high school education to read.
                    7. Avoid scenarios that deal with either jealousy in relationships or involve ethical or moral dilemmas.
                    8. Avoid scenarios that require specific domain knowledge or experience to solve. Remember, the dilemma needs to be relatable to an average college student.
                    9. The scenario should be open-ended and have more than 2 possible solutions. The scenario should also be ambiguous and have no solution that is clearly better or more obvious than the others. Remember: do not suggest any possible solution in the scenario.
                    10. Your scenario should involve higher stakes than the personal preferences of the main character; there should be clear repercussions from any potential action, such that solving the dilemma requires critical thinking.""",
                ),
                (
                    "human",
                    """Word list:
        {word_list}

        Dilemma topic:
        {topic}

        ###

        Scenario:""",
                ),
            ],
            [  # 9, including the evaluation scale (for LLM feedback)
                (
                    "system",
                    """You are an author tasked with producing scenarios for a short story. You will be given a list of 5 words, consisting of 3 names, a place, and an action. Using ONLY these words, think of a scenario that involves all the words. This scenario should involve a dilemma that one of the named people from the list, the main character, needs to solve. Here is a list of rules you should follow when writing the scenario:

                    1. The dilemma should be relatable to an average college student and must involve scenarios that a typical college student might need to confront.
                    2. Do not suggest any possible solution to the dilemma in the scenario, avoid phrases like "She is torn between...", "On the one hand...", "On the other hand...", or "He is not sure whether he should do X or Y..." as these may imply possible solutions to the dilemma. The scenario will be given to another writer as part of a writing prompt, and we do not want to bias their writing by suggesting how the story will unfold. Focus only on describing the dilemma and its significance to the main character.
                    3. Include as many details about the scenario as you can.
                    4. Respond in at least 8 but no more than 12 sentences and in a single paragraph.
                    5. In the last sentence, state something like "Z does not know what to do.", where Z is the name of the main character.
                    6. Make sure what you write is not too difficult to read; avoid complex jargon wherever possible. It should be easy for someone with a high school education to read.
                    7. Avoid scenarios that deal with either jealousy in relationships or involve ethical or moral dilemmas.
                    8. Avoid scenarios that require specific domain knowledge or experience to solve. Remember, the dilemma needs to be relatable to an average college student.
                    9. The scenario should be open-ended and have more than 2 possible solutions. The scenario should also be ambiguous and have no solution that is clearly better or more obvious than the others. Remember: do not suggest any possible solution in the scenario.
                    10. Your scenario should involve higher stakes than the personal preferences of the main character; there should be clear repercussions from any potential action, such that solving the dilemma requires critical thinking.
                    
                    Each scenario you create will be rated according to this rubric:
                    Complexity
                    1 = Only two competing demands
                    2 = Some complexity but still a dilemma
                    3 = Multiple competing considerations

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
                    3 = Purely task/goal-focused""",
                ),
                (
                    "human",
                    """Word list:
                    {word_list}

                    Dilemma topic:
                    {topic}

                    ###

                    Scenario:""",
                ),
            ],
        ]
        creative_scenario_generation_prompt = ChatPromptTemplate.from_messages(
            scenario_base_prompts[scendario_prompt_idx]
        )
        return creative_scenario_generation_prompt


class ConsequencesItemParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        js_output = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "output": text,
            "item_type": "consequences",
        }
        return js_output


class CreativeWordlistItemParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        text = text.split("\n\n")
        text = [re.sub(r"([0-9.])+", "", t) for t in text]
        text = [t.strip("\n").strip(" ") for t in text]
        js_output = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "output": text,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "item_type": "creative_scenario",
        }
        return js_output


class CreativityScenarioItemParser(BaseOutputParser):
    # TODO: for the next round of items after feedback, get rid of everything generated before and after the start of the original item
    # TODO: OpenAI returns an AIMessage object, test and confirm if this also happends with hf (we unwrap here by accessing content)
    @staticmethod
    def parse(text: str, scenario_names) -> dict:
        text = text.content
        text = text.strip("\n").strip(" ")
        # Remove intervening newlines
        text = re.sub("\n", "", text)
        text = re.sub("\t", "", text)
        readability = Readability(text)
        if len(word_tokenize(text)) < 120:  # drop scenarios that are too short
            print("Scenario too short.")
            text = "None"
        # elif "dilemma" in text:
        #     print("Scenario contains a forbidden keyword.")
        #     text = "None"
        elif (
            readability.flesch().score < 45
        ):  # based on some initial feedback on the results
            print("Scenario too difficult to read.")
            text = "None"
        # remove all text after "X does not know what to do".
        elif len(re.findall(r"(does not know what to do\.)", text)) != 0:
            split_on_final_question = re.split(r"(does not know what to do\.)", text)
            text = split_on_final_question[0] + split_on_final_question[1]
        elif "###" in text:
            text = text.split("###")[0]

        # Sometimes, the LLM will output a list of possible solutions
        # if it does that, drop the output
        elif len(re.findall(r"([0-9]{1}\.[a-zA-Z\W]+\?)", text)) != 0:
            print("LLM output possible solutions.")
            text = "None"
        
        # the scenario should start with one of the named characters from the wordlist
        # check for this and remove all text preceding it
        if text != "None":
            text_split_on_word = word_tokenize(text)
            first_word = text_split_on_word[0]
            if first_word not in scenario_names:
                print("Scenario does not begin with character name")
                text = "None"
        

        js_output = {
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "output": text,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "item_type": "creative_scenario",
        }
        return js_output


# test the chain
def test_consequences():
    prompt = PromptGenerator.make_consequence_prompt()
    chain = prompt | llm | ConsequencesItemParser()
    result = chain.invoke(
        {"": ""}
    )  # invoking a chain requires passing an input for the template, which can be a blank dict
    return result


def test_creative_wordlist_generation(prompt_idx: int, llm):
    prompt = PromptGenerator.make_creative_scenario_wordlist_generation_prompt(
        prompt_idx
    )  # the prompt type
    chain = prompt | llm | CreativeWordlistItemParser()
    word_list = chain.invoke({"": ""})
    return word_list


def test_creative_problem(
    word_list,
    prompt_idx: int,
    llm,
    scenario_names: str,
    previous_llm_output=None,
    topic_from_file=None,
    ratings_from_file=False,
):
    # when true, will use AI feedback to improve the model outputs
    # llm_evaluator should be either the name of the model to call for evaluation,
    # or the name of a file with saved output for each prompt (for a second round of item gen)
    # both use_eval and llm_evaluator must be set for this path
    if (
        topic_from_file != None
        and ratings_from_file != None
        and previous_llm_output != None
    ):
        prompt = PromptGenerator.make_creative_scenario_generation_prompt(
            prompt_idx
        )  # the prompt type

        # add previous output to the prompt
        prompt.append(("ai", "{ai_output}"))
        # add the LLM evaluation to the prompt
        prompt.append(
            (
                "human",
                """
                Here is some feedback for your scenario on a scale of 1-3:
                {ai_feedback}

                Please revise your scenario, and try to score a 3 in each category.""",
            )
        )
        chain = prompt | llm
        result = chain.invoke(
            {
                "word_list": word_list,
                "topic": topic_from_file,
                "ai_output": previous_llm_output,
                "ai_feedback": ratings_from_file,
            }
        )
        result = CreativityScenarioItemParser.parse(result, scenario_names)
        return result
    else:
        # choose a topic at random to build the scenario
        # these are written manually for now, could try LLM generated in the future
        dilemma_topics = [
            "secret crush",
            "morality and ethics",
            "greatest fear",
            "greatest dream",
            "past trauma",
            "friendship versus work",
            "multiple competing demands",
            "family versus career",
        ]
        prompt = PromptGenerator.make_creative_scenario_generation_prompt(
            prompt_idx
        )  # the prompt type
        topic = dilemma_topics[randint(0, len(dilemma_topics) - 1)]

        chain = prompt | llm
        result = chain.invoke({"word_list": word_list, "topic": topic})
        result = CreativityScenarioItemParser.parse(result, scenario_names)

        return result, topic


# cookbooks for item gen
def create_wordlists(prompt_idx: int, output_file: str, llm):
    js_array = []
    for i in tqdm(range(5)):  # each wordlist call creates 10 lists
        time.sleep(2)  # rate limit
        result = test_creative_wordlist_generation(prompt_idx, llm)
        js_array.append(result)

    df = (
        pd.json_normalize(js_array)
        .explode("output")
        .reset_index()
        .drop("index", axis=1)
    )
    df.to_csv(f"{output_file}", sep="\t")


def create_scenarios(
    prompt_idx: int,
    output_file: str,
    model_name: str,
    llm,
    input_file: str = None,
    llm_evaluator: str = None,
    wordlist_file: str = None,
    presence_penalty: float = 0.0
):
    # when true, will use AI feedback to improve the model outputs
    if input_file != None and llm_evaluator == None:
        try:
            input_file = pd.read_csv(
                input_file,
                sep="\t",
                index_col=0,
            )
        except Exception:
            input_file = pd.read_json(
                input_file
            )
        if "ratings" not in input_file.columns or (
            "creative_scenario" not in input_file.columns
            and "creative_scenario_without_feedback" not in input_file.columns
        ):
            print("Input file does not contain both item ratings and item content!")
            exit(-1)

        input_file.rename(
            {"creative_scenario": "creative_scenario_without_feedback"},
            axis=1,
            inplace=True,
        )
        input_file["creative_scenario_with_feedback"] = ""
        for index, row in tqdm(input_file.iterrows(), total=input_file.shape[0]):
            if model_name == "gpt-4" or model_name == "gpt-3.5-turbo":
                time.sleep(2)
            result = test_creative_problem(
                row["word_list"],
                prompt_idx,
                llm,
                row["creative_scenario_without_feedback"],
                row["topic"],
                row["ratings"],
            )
            input_file.at[index, "creative_scenario_with_feedback"] = result["output"]

        # drop rows that failed quality controls
        input_file = input_file[input_file["output"] != "None"]
        model_dir = model_name.replace("/", "-")
        input_file.to_json(
            f"/home/aml7990/Code/creativity-item-generation/outputs/with_eval_scores/{output_file}_{model_dir}.json",
        )

    elif llm_evaluator != None and input_file == None:
        # path for a fresh round of item generation with evaluation
        pass
    elif llm_evaluator == None and input_file == None:
        # path for a fresh round of item generation without evalution
        try:
            wordlists = pd.read_csv(
                f"/home/aml7990/Code/creativity-item-generation/outputs/{wordlist_file}.tsv",
                sep="\t",
                index_col=0,
            )
        except Exception:
            wordlists = pd.read_json(
                f"/home/aml7990/Code/creativity-item-generation/outputs/{wordlist_file}.tsv",
            )
        wordlists.rename({"output": "word_list"}, axis=1, inplace=True)
        wordlists["creative_scenario"] = ""
        wordlists["topic"] = ""
        for index, row in tqdm(wordlists.iterrows(), total=wordlists.shape[0]):
            if model_name == "gpt-4" or model_name == "gpt-3.5-turbo":
                time.sleep(2)
            # grab just the names in the wordlist, need for preprocessing
            scenario_names = row['word_list'].split(",")[::2]
            scenario_names = [re.sub(r"([0-9]{1}\.)","", s).strip() for s in scenario_names]
            result, topic = test_creative_problem(row["word_list"], prompt_idx, llm, scenario_names)
            wordlists.at[index, "creative_scenario"] = result["output"]
            wordlists.at[index, "model_name"] = model_name
            wordlists.at[index, "topic"] = topic
            wordlists.at[index, "max_tokens"] = max_tokens
            wordlists.at[index, "presence_penalty"] = presence_penalty

        # drop rows that failed quality control metrics
        
        wordlists = wordlists[wordlists["creative_scenario"] != "None"]
        model_dir = model_name.replace("/", "-")
        wordlists.to_json(
            f"/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/{output_file}_{model_dir}.json",
        )
    else:
        print("Unsupported combination of arguments!")


# test prompt X number of times, and save in df
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--task", type=str)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    parser.add_argument("--frequency_penalty", type=float)
    parser.add_argument("--presence_penalty", type=float)
    parser.add_argument("--prompt_idx", type=int)
    parser.add_argument("--max_tokens", type=int)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--llm_evaluator", type=str, default=None)
    parser.add_argument("--wordlist_file", type=str, default=None)
    parser = parser.parse_args()
    try:
        task = parser.task
        model_name = parser.model_name
        temperature = parser.temperature
        max_tokens = parser.max_tokens
        top_p = parser.top_p
        frequency_penalty = parser.frequency_penalty
        presence_penalty = parser.presence_penalty
        batch_size = parser.batch_size
        input_file = parser.input_file
        llm_evaluator = parser.llm_evaluator
        wordlist_file = parser.wordlist_file
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
            model_kwargs = {
                "top_p": top_p,
                "temperature": temperature,
                "device_map": "auto",
                # "torch_dtype": torch.bfloat16, # don't use with 8 bit mode
            }
            tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
            model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, **model_kwargs)
            pipeline = hf_pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_new_tokens=max_tokens,
                model_kwargs=model_kwargs,
            )
            llm = HuggingFacePipeline(pipeline=pipeline)

    except Exception:
        print("Model failed to initialize. Please check your API key.")
        exit(-1)
    if task == "scenario_generation":
        create_scenarios(
            parser.prompt_idx,
            parser.output_file,
            parser.model_name,
            llm,
            input_file,
            llm_evaluator,
            wordlist_file,
            presence_penalty=presence_penalty,
        )
    elif task == "wordlist generation":
        create_wordlists(parser.prompt_idx, parser.output_file, llm)
    else:
        print("Not a valid task name!")
        exit(-1)
