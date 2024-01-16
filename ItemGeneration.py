import time
import torch
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
    # TODO: make the shots a parameter
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
        # TODO: move the base prompts to a separate file that is imported, passed to function
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
    def parse(self, text: str) -> dict:
        text = text.strip("\n").strip(" ")
        # Remove intervening newlines
        text = re.sub("\n", "", text)
        readability = Readability(text)
        if len(word_tokenize(text)) < 120:  # drop scenarios that are too short
            text = None
        elif "dilemma" in text:
            text = None
        elif (
            readability.flesch().score < 45
        ):  # based on some initial feedback on the results
            text = None
        # remove all text after "X does not know what to do", or drop if regex doesn't match this.
        if len(re.findall(r'(does not know what to do\.)', text)) != 0:
            split_on_final_question = re.split(r'(does not know what to do\.)', text)
            text = split_on_final_question[0] + split_on_final_question[1]
        else:
            text = None
        if "###" in text:
            text = text.split("###")[0]
        
        # Sometimes, the LLM will output a list of possible solutions
        # if it does that, drop the output
        if len(re.findall(r'([0-9]{1}\.[a-zA-Z\W]+\?)', text)) != 0:
            text = None

        # TODO:
        # 1. check that all keywords appear in the prompt
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


def test_creative_problem(word_list, prompt_idx: int, model_name: str, llm):
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

    chain = prompt | llm | CreativityScenarioItemParser()
    result = chain.invoke({"word_list": word_list, "topic": topic})

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


def create_scenarios(prompt_idx: int, output_file: str, model_name: str, llm):
    wordlists = pd.read_csv(
        f"/home/aml7990/Code/creativity-item-generation/outputs/creative_wordlist_5_words.tsv",
        sep="\t",
        index_col=0,
    )
    wordlists.rename({"output": "word_list"}, axis=1, inplace=True)
    wordlists["creative_scenario"] = ""
    wordlists["topic"] = ""
    # wordlists = wordlists.iloc[0:15] # TODO: REMOVE
    for index, row in tqdm(wordlists.iterrows(), total=wordlists.shape[0]):
        if model_name == "gpt-4" and model_name == "gpt-3.5-turbo":
            time.sleep(2)
        result, topic = test_creative_problem(
            row["word_list"], prompt_idx, model_name, llm
        )
        if result == None:
            continue
        wordlists.at[index, "creative_scenario"] = result["output"]
        wordlists.at[index, "model_name"] = model_name
        wordlists.at[index, "topic"] = topic
        wordlists.at[index, "max_tokens"] = max_tokens

    # drop rows that failed quality control metrics
    wordlists = wordlists[wordlists["creative_scenario"] != ""]
    model_dir = model_name.replace("/", "-")
    wordlists.to_csv(
        f"/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/{output_file}_{model_dir}.tsv",
        sep="\t",
    )


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
                "torch_dtype": torch.bfloat16,
            }
            tokenizer = AutoTokenizer.from_pretrained(model_name, **model_kwargs)
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
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
        create_scenarios(parser.prompt_idx, parser.output_file, parser.model_name, llm)
    elif task == "wordlist generation":
        create_wordlists(parser.prompt_idx, parser.output_file, llm)
    else:
        print("Not a valid task name!")
        exit(-1)
