import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from key import key
from tqdm import tqdm
import time

# API key stored in key.py, and should NOT be committed
# TODO: add support for more LLMs
try:
    model_name = "gpt-4"
    temperature = 1
    max_tokens = 1500
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=key,
        temperature=temperature,
        max_tokens=max_tokens,
        model_kwargs={
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        },
    )
except Exception:
    print("Model failed to initialize. Please check your API key.")
    exit(-1)


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
    # TODO: make a parameter
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
                    "Create 10 wordlists, make sure to use a different action each time. Separate wordlists by newlines, do not number them.",
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
                    """You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of 3 different scenarios that involve all the words. This scenarios should involve dilemmas that one of the named people from the list needs to solve. These dilemmas should be relatable to an average person, and it should also have no clear solution. Include as many details about the situations as you can, and try to keep your scenarios at about a paragraph each.

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
                    """You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of 3 different scenarios that involve all the words. This scenarios should involve dilemmas that one of the named people from the list needs to solve. These dilemmas should be relatable to an average person, and it should also have no clear solution. Include as many details about the situations as you can, and try to keep your scenarios at about a paragraph each.

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
                    """You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of 3 different scenarios that involve all the words. This scenarios should involve dilemmas that one of the named people from the list needs to solve. These dilemmas should be relatable to an average person, and it should also have no clear solution. Include as many details about the situations as you can, and try to keep your scenarios at about a paragraph each.

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
                    """You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of 3 different scenarios that involve all the words. This scenarios should involve dilemmas that one of the named people from the list needs to solve. These dilemmas should be relatable to an average person, and it should also have no clear solution. Include as many details about the situations as you can, and try to keep your scenarios at about a paragraph each.

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
                    """You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of 3 different scenarios that involve all the words. This scenarios should involve dilemmas that one of the named people from the list needs to solve. These dilemmas should be relatable to an average person, and it should also have no clear solution. Include as many details about the situations as you can, and try to keep your scenarios at about a paragraph each.

        ###

        Word list:
        {word_list}

        ###

        Scenario:
        """,
                ),
            ],  # 6
            [
                (
                    "system",
                    'You are an author tasked with coming up with scenarios for a short story. You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of a scenario that involves all the words. This scenario should involve a dilemma that one of the named people from the list, the main character, needs to solve. The dilemma should be relatable to an average person, and it should also have no clear solution. Do not suggest any possible solution to the dilemma in the scenario, avoid phrases like "She is torn between..." or "He is not sure whether he should do X or Y..." as these may imply possible solutions to the dilemma. The scenario will be given to another writer as part of a writing prompt, and we do not want to bias their writing by suggesting how the story will unfold. Focus only on describing the dilemma and its significance to the main character. Include as many details about the scenario as you can, and try to keep your scenario at about a paragraph in length, using at least 6 sentences. In the last sentence, state something similar to "Z does not know what to do.", where Z is the name of the main character.',
                ),
                (
                    "human",
                    """Word list:
        {word_list}

        ###

        Scenario:""",
                ),
            ],
        ]
        creative_scenario_generation_prompt = ChatPromptTemplate.from_messages(
            scenario_base_prompts[scendario_prompt_idx]
        )
        return creative_scenario_generation_prompt


# TODO: more extensive output parsing
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
    def parse(self, text: str) -> dict:
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


# TODO: set up code for saving to csv for analysis
# test the chain
def test_consequences():
    prompt = PromptGenerator.make_consequence_prompt()
    chain = prompt | llm | ConsequencesItemParser()
    result = chain.invoke(
        {"": ""}
    )  # invoking a chain requires passing an input for the template, which can be a blank dict
    return result


def test_creative_wordlist_generation():
    prompt = PromptGenerator.make_creative_scenario_wordlist_generation_prompt(
        1
    )  # the prompt type
    chain = prompt | llm | CreativeWordlistItemParser()
    word_list = chain.invoke({"": ""})
    return word_list


def test_creative_problem(word_list):
    prompt = PromptGenerator.make_creative_scenario_generation_prompt(
        5
    )  # the prompt type
    chain = prompt | llm | CreativityScenarioItemParser()
    result = chain.invoke({"word_list": word_list})
    return result


# cookbooks for item gen
def create_wordlists():
    js_array = []
    for i in tqdm(range(5)):  # each wordlist call creates 10 lists
        time.sleep(2)  # rate limit
        result = test_creative_wordlist_generation()
        js_array.append(result)

    df = (
        pd.json_normalize(js_array)
        .explode("output")
        .reset_index()
        .drop("index", axis=1)
    )
    df.to_csv("creative_wordlist.tsv", sep="\t")


# test prompt X number of times, and save in df
if __name__ == "__main__":
    wordlists = pd.read_csv("outputs/creative_wordlist_v2.tsv", sep="\t", index_col=0)
    wordlists.rename({"output": "word_list"}, axis=1, inplace=True)
    wordlists["creative_scenario"] = ""
    for index, row in wordlists.iterrows():
        time.sleep(2)
        result = test_creative_problem(row["word_list"])
        wordlists.at[index, "creative_scenario"] = result["output"]

    wordlists.to_csv("outputs/creative_scenario.tsv", sep="\t")
