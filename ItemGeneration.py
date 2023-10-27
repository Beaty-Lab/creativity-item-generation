import json
from langchain.llms import OpenAIChat
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from key import key

# API key stored in key.py, and should NOT be committed
# TODO: add support for more LLMs
try:
    model_name="gpt-4"
    temperature=1
    max_tokens=2048
    llm = OpenAIChat(model_name=model_name, openai_api_key=key, temperature=temperature, max_tokens=max_tokens)
except Exception:
    print("Model failed to initialize. Please check your API key.")
    exit(-1)


# class for storing and manipulating prompts
class PromptGenerator:
    @staticmethod
    def make_consequence_prompt():
        consequences_prompt =  ChatPromptTemplate.from_messages([("system", "You are a scenario writer."),("human",
            """Think of a scenario the would change the way human life or the world works. The scenario should alter people's daily lives in important ways. Please describe only the scenario, and don't hint at any potential implications of the scenario. Please describe this scenario in 12 words at most.

        ###

        Scenario:"""
        )])
        return consequences_prompt

    # NOTE: few shots are currently baked into prompt
    # TODO: make a parameter
    @staticmethod
    def make_creative_scenario_wordlist_generation_prompt():
        creative_scenario_wordlist_generation_prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant."),("human",
            """Create a list of 4 words. In the list, include 2 human names, a place, and an action. You don't need to use names of actual people or places, but only use words which relate to the experiences of typical people; for example, do not include a word like "starship" for the place since no one alive today has been on a starship. You should list words in exactly this order: name, place, name, action.

        ###

        Example word lists: 

        1. Becky, pizzeria, Jim, theft

        2. Wentworth, Acme Company, Scott, employment

        ###
        New word list:
        1. """
        )])
        return creative_scenario_wordlist_generation_prompt

    """
    Generates a prompt for creative scenario generation based on a given word list.

    Parameters:
        word_list (str): A list of 4 words, consisting of 2 names, a place, and an action.

    Returns:
        str: The generated prompt for creative scenario generation.
    """

    @staticmethod
    def make_creative_scenario_generation_prompt():
        creative_scenario_generation_prompt = ChatPromptTemplate.from_messages([("system","You are a scenario writer."),("human",
            """You will be given a list of 4 words, consisting of 2 names, a place, and an action. Using ONLY these words, think of 3 different scenarios that involve all the words. This scenarios should involve dilemmas that one of the named people from the list needs to solve. These dilemmas should be relatable to an average person, and it should also have no clear solution. Include as many details about the situations as you can, and try to keep your scenarios at about a paragraph each.

        ###

        Word list:
        {word_list}

        ###

        Scenario:
        """
        )])
        return creative_scenario_generation_prompt

# TODO: more extensive output parsing
class ConsequencesItemParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        js_output = {
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "output": text,
        "item_type": "consequences"                
        }
        return js_output
    
class CreativityScenarioItemParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        scenario_list = text.split("\n\n")
        js_output = {
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "output": scenario_list,
        "item_type": "creative_scenario"                
        }
        return js_output

# TODO: set up code for saving to csv for analysis
# test the chain
def test_consequences():
    prompt = PromptGenerator.make_consequence_prompt()
    chain = prompt | llm | ConsequencesItemParser()
    result = chain.invoke({"":""}) # invoking a chain requires passing an input for the template, which can be a blank dict
    print(result)

def test_creative_problem():
    prompt_1 = PromptGenerator.make_creative_scenario_wordlist_generation_prompt()
    chain_1 = prompt_1 | llm
    word_list = chain_1.invoke({"":""})
    prompt_2 = PromptGenerator.make_creative_scenario_generation_prompt()
    chain_2 = prompt_2 | llm | CreativityScenarioItemParser()
    result = chain_2.invoke({"word_list":word_list})
    print(result)


if __name__ == "__main__":
    # print("Consequences:")
    # test_consequences()
    print("Creative Problem:")
    test_creative_problem()