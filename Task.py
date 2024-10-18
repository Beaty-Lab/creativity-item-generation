"""
    Defines classes for the tasks to run item gen on.
    Conceptually, a task is a creativity datasets that the LLM both solves and generates items for.
    Tasks must consist of:
    1. A prompt template for each component of the pipeline.
    2. A series of methods for scoring the responses that can be passed to the constraint optimizer.
"""

from langchain.prompts.chat import _convert_to_message
from langchain.schema import BaseOutputParser, StrOutputParser, OutputParserException
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain.output_parsers import RetryOutputParser
from langchain.prompts.chat import ChatPromptTemplate

from abc import ABC, abstractmethod
from typing import List, Tuple

from config import config
import imp
import re
import pandas as pd
from readability import Readability
from nltk import word_tokenize

# TODO update the task classses to use the scoring predict function and load the correct py scrip


class AbstractTask(ABC):
    # each task must define item parsers that specify how to check LLM outputs
    # these must be implemented in langchain
    class CreativityScenarioItemParser(BaseOutputParser):
        pass

    class CreativeResponseItemParser(BaseOutputParser):
        pass

    @abstractmethod
    def RunScorers(self) -> List:
        pass

    @abstractmethod
    def RunItemGeneration(self) -> str:
        pass

    @abstractmethod
    def RunItemResponseGeneration(self) -> str:
        pass


class CPS(AbstractTask):
    # add custom modules under the class def
    roberta_scorer = imp.load_source(
        "RLPS_RoBERTa",
        "/home/aml7990/Code/creativity-item-generation/scorers/CPS/CPSFinetuneTLM.py",
    )

    def __init__(self) -> None:
        super().__init__()
        self.task_id = "CPS"

    # item generation requires there be a word_list column from which the list can be extracted
    def prep_wordlist(self, wordlist_file: str, round: int) -> pd.DataFrame:
        wordlists = pd.read_csv(wordlist_file, sep="\t", index_col=0)
        wordlists.rename({"output": "word_list"}, axis=1, inplace=True)
        wordlists[f"creative_scenario_round_{round}"] = ""
        wordlists[f"prompt_round_{round}"] = ""
        return wordlists

    class CreativityScenarioResponseParser(BaseOutputParser):
        # TODO: for the next round of items after feedback, get rid of everything generated before and after the start of the original item
        @staticmethod
        def parse(text: str) -> str:
            try:
                text = text.split("Solution:")[1]
                text = text.strip("\n").strip(" ")
            except Exception:
                # OpenAIs models with yield an AIMessage object
                text = text.content.split("Solution:")[1]
                text = text.strip("\n").strip(" ")
            # Remove intervening newlines
            text = re.sub("\n", "", text)
            text = re.sub("\t", "", text)

            return text

    class CreativityScenarioItemParser(BaseOutputParser):
        # TODO: is it possible for the user to specify some of the output formatitng, like the forbidden strings?
        # Add this to config file
        # TODO: can we pass the parsing exception into the retry parser, add it to the retry prompt?
        @staticmethod
        def parse(text: str) -> str:
            # get rid of the instructions
            text = text.split("Scenario:")[1]
            print(text)
            forbidden_strings = [
                "On the one hand",
                "On the other hand",
                "dilemma",
                "must navigate",
                "must decide",
                "has to decide",
                "is torn between",
            ]

            # Remove intervening newlines
            text = re.sub("\n", "", text)
            text = re.sub("\t", "", text)

            if text is None:
                print("Empty string generated.")
                raise OutputParserException("Empty string generated.")

            # remove all text after stop sequence
            if "I am finished with this scenario." not in text:
                print("Termination string not found.")
                raise OutputParserException("Termination string not found.")
            else:
                head, sep, tail = text.partition("I am finished with this scenario.")
                text = head

            # remove phrases indicating LLM is "spelling out" solution
            for f in forbidden_strings:
                if f in text:
                    print("Scenario contains forbidden string.")
                    raise OutputParserException("Scenario contains forbidden string.")

            readability = Readability(text)
            if len(word_tokenize(text)) < 140:  # drop scenarios that are too short
                print("Scenario too short.")
                raise OutputParserException("Scenario too short.")

            elif (
                readability.flesch().score < 45
            ):  # based on some initial feedback on the results
                print("Scenario too difficult to read.")
                raise OutputParserException("Scenario too difficult to read.")

            text = text.strip("\n").strip(" ")

            return text  # , "OK"

    @staticmethod
    def RunItemGeneration(
        word_list,
        prompt: List[Tuple[str]],
        llm,
        previous_llm_output=None,
        ratings_from_file=None,
        item_shots: list = None,
        numAttempts: int = 1,
        task: str = None,
    ):
        # item generation
        parser = CPS.CreativityScenarioItemParser()
        retry_parser = RetryOutputParser.from_llm(
            parser=parser, llm=llm, max_retries=numAttempts
        )
        # when true, will use AI feedback to improve the model outputs
        if previous_llm_output is not None:
            prompt_formatted = ChatPromptTemplate.from_messages(
                prompt,
            )
            human_query = prompt_formatted.messages[1].prompt.template
            prompt_formatted.messages.pop()

            # I think put them as one message where the human lists some example items
            # TODO: make sure the items are properly formatted
            item_shots = _convert_to_message(
                (
                    "human",  # TODO: refactor: add prompt template to prompts.py
                    "\nHere are some more examples of high quality scenarios from other authors. Use these scenarios as guidance, but avoid drawing from them too heavily when developing your own:\n"
                    + "\n###\n".join(item_shots)
                    + f"""\n###\nWord list:
                        {word_list}

                        ###

                        Scenario:""",
                )
            )
            prompt_formatted.messages.insert(1, item_shots)

            # add AI feedback, if it exists
            if ratings_from_file is not None:
                # add previous output to the prompt
                prompt_formatted.messages.insert(
                    2, _convert_to_message(("ai", "{ai_output}"))
                )
                # add the LLM evaluation to the prompt
                prompt_formatted.messages.insert(
                    3,
                    _convert_to_message(
                        (
                            "human",
                            """
                        Here is some feedback for your scenario:
                        {ai_feedback}

                        Please revise your scenario, and try improve your score in each category. Remember, maximizing the scores doesn't mean your scenario is better. Also, please don't make all your edits at the end of the scenario, spread them throughout.
                        
                        ###
                        
                        """
                            + human_query,
                        )
                    ),
                )
            completion_chain = (
                prompt_formatted | llm | StrOutputParser()
            )  # StrOutputParser grabs the content field from chat models
            validation_chain = RunnableParallel(
                completion=completion_chain, prompt_value=prompt_formatted
            ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

            if ratings_from_file is not None:
                final_prompt = prompt_formatted.format(
                    **{
                        "word_list": word_list,
                        "ai_output": previous_llm_output,
                        "ai_feedback": ratings_from_file,
                    }
                )
                result = validation_chain.invoke(
                    {
                        "word_list": word_list,
                        "ai_output": previous_llm_output,
                        "ai_feedback": ratings_from_file,
                    }
                )
                print(result)
            else:
                final_prompt = prompt_formatted.format(
                    **{
                        "word_list": word_list,
                    }
                )
                result = validation_chain.invoke(
                    {
                        "word_list": word_list,
                    }
                )
                print(result)

            return result, final_prompt
        else:
            prompt_formatted = ChatPromptTemplate.from_messages(
                prompt,
            )

            # StrOutputParser grabs the content field from chat models
            completion_chain = prompt_formatted | llm | StrOutputParser()

            # We try to regenerate a few times if the LLM fails validation
            # Should we be unable to fix the scenario, we return "None", these get dropped later
            validation_chain = RunnableParallel(
                completion=completion_chain, prompt_value=prompt_formatted
            ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

            final_prompt = prompt_formatted.format(word_list=word_list)
            result = validation_chain.invoke({"word_list": word_list})
            return result, final_prompt

    @staticmethod
    def RunItemResponseGeneration(
        problem,
        prompt: List[Tuple[str]],
        prompt_idx: int,
        llm,
        ethnicity: str = None,
        gender: str = None,
        industry: str = None,
        title: str = None,
        FirstName: str = None,
        LastName: str = None,
        Occuptaion: str = None,
        Field: str = None,
        Psychometric: str = None,
    ):
        prompt_formatted = ChatPromptTemplate.from_messages(prompt)

        chain = prompt_formatted | llm
        if prompt_idx == 0:
            result = chain.invoke({"creative_scenario": problem})
        # depding on the type of item response prompt, different demographices are needed and they map to a different prompt template
        elif prompt_idx == 1:
            result = chain.invoke(
                {
                    "creative_scenario": problem,
                    "ethnicity": ethnicity,
                    "gender": gender,
                    "industry": industry,
                    "title": title,
                }
            )
        elif prompt_idx == 2:
            result = chain.invoke(
                {
                    "creative_scenario": problem,
                    "FirstName": FirstName,
                    "LastName": LastName,
                    "Occupation": Occuptaion,
                    "Field": Field,
                    "Psychometric": Psychometric,
                }
            )

        result = CPS.CreativityScenarioResponseParser.parse(result)
        return result

    # run scoring for the task
    # to ensure correct behavior, round must always be passed
    def RunScorers(self, i: int) -> None:
        """
        Run the scoring process for the task.

        Args:
            i (int): The round number.

        Returns:
            None
        """
        self.roberta_scorer.predict_with_model(
            config["itemResponseOriginalityModelDir"],
            config["itemResponseGenOutputFile"],
            config["shotSelectionMetric"],
            config[
                "itemResponseGenOutputFile"
            ],  # we need to chop off most columns from the first instance, so send another copy to save to
            i,  # round
            config["useItemScoringModelPeft"],
        )


class Consequences(AbstractTask):
    scorer = imp.load_source(
        "predict_with_model",
        "/home/aml7990/Code/creativity-item-generation/scorers/consequences/flan-t5/ConsequencesFinetuneTLM.py",
    )

    def __init__(self) -> None:
        super().__init__()
        self.task_id = "consequences"

    class CreativityScenarioResponseParser(BaseOutputParser):
        @staticmethod
        def parse(text: str) -> str:
            forbidden_strings = [
                "###",
                "*"
            ]
            try:
                if "Consequences:" in text:
                    text = text.split("Consequences:")[1]
                text = text.strip("\n").strip(" ")
            except Exception:
                # OpenAIs models with yield an AIMessage object
                if "Consequences:" in text.content:
                    text = text.content.split("Consequences:")[1]
                text = text.strip("\n").strip(" ")

            for f in forbidden_strings:
                if f in text:
                    print("Scenario contains forbidden string.")
                    raise OutputParserException("Scenario contains forbidden string.")
            # Remove intervening newlines
            text = re.sub("\n", "", text)
            text = re.sub("\t", "", text)
            if not text.endswith("."):
                print("Response consequences should end with a period.")
                raise OutputParserException("Response consequences should end with a period.")

            return text

    class CreativityScenarioItemParser(BaseOutputParser):
        @staticmethod
        def parse(text: str) -> str:
            forbidden_strings = [
                "your",
                "this scenario would",
                "you respond",
                "human:",
                "this is a great scenario",
                "this is a bad scenario",
                "the author",
                "Please describe the scenario in 12 words at most."
            ]

            text = text.split("Scenario:")[-1].split("###")[0]
            for f in forbidden_strings:
                if f in text.lower():
                    print("Scenario contains forbidden string.")
                    raise OutputParserException("Scenario contains forbidden string.")
            if len(re.findall(r"([0-9]\.)", text)) > 0:
                text = re.split(r"([0-9]\.)", text)[::2][1]

            # Remove intervening newlines
            text = re.sub("\n", "", text)
            text = re.sub("\t", "", text)
            text = text.strip("\n").strip(" ")

            return text # OK

    @staticmethod
    def RunItemGeneration(
        prompt: List[Tuple[str]],
        llm,
        previous_llm_output=None,
        ratings_from_file=None,
        item_shots: list = None,
        numAttempts: int = 1,
        task: str = None,
    ):
        # item generation
        parser = Consequences.CreativityScenarioItemParser()
        retry_parser = RetryOutputParser.from_llm(
            parser=parser, llm=llm, max_retries=numAttempts
        )
         # when true, will use AI feedback to improve the model outputs
        if previous_llm_output is not None:
            prompt_formatted = ChatPromptTemplate.from_messages(
                prompt,
            )

            # I think put them as one message where the human lists some example items
            # TODO: make sure the items are properly formatted
            item_shots = _convert_to_message(
                (
                    "human",  # TODO: refactor: add prompt template to prompts.py
                    "\nHere are some examples of high quality scenarios from other authors. Use these scenarios as guidance, but avoid drawing from them too heavily when developing your own:\n"
                    + "\n###\n".join(item_shots)
                    + f"""###

                        Scenario:""",
                )
            )
            prompt_formatted.messages.insert(len(prompt_formatted.messages), item_shots)

            # add AI feedback, if it exists
            if ratings_from_file is not None:
                # add previous output to the prompt
                prompt_formatted.messages.insert(
                    2, _convert_to_message(("ai", "{ai_output}"))
                )
                # add the LLM evaluation to the prompt
                prompt_formatted.messages.insert(
                    3,
                    _convert_to_message(
                        (
                            "human",
                            """
                        Here is some feedback for your scenario:
                        {ai_feedback}

                        Please revise your scenario, and try improve your score in each category. Remember, maximizing the scores doesn't mean your scenario is better. Also, please don't make all your edits at the end of the scenario, spread them throughout.
                        
                        ###
                        
                        """
                            + human_query,
                        )
                    ),
                )
            completion_chain = (
                prompt_formatted | llm | StrOutputParser()
            )  # StrOutputParser grabs the content field from chat models
            validation_chain = RunnableParallel(
                completion=completion_chain, prompt_value=prompt_formatted
            ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

            if ratings_from_file is not None:
                final_prompt = prompt_formatted.format(
                    **{
                        "ai_output": previous_llm_output,
                        "ai_feedback": ratings_from_file,
                    }
                )
                result = validation_chain.invoke(
                    {
                        "ai_output": previous_llm_output,
                        "ai_feedback": ratings_from_file,
                    }
                )
                print(result)
            else:
                result = validation_chain.invoke({})
                final_prompt = prompt_formatted.format()
                print(result)

            return result, final_prompt
        else:
            prompt_formatted = ChatPromptTemplate.from_messages(
                prompt,
            )

            # StrOutputParser grabs the content field from chat models
            completion_chain = prompt_formatted | llm | StrOutputParser()

            # We try to regenerate a few times if the LLM fails validation
            # Should we be unable to fix the scenario, we return "None", these get dropped later
            validation_chain = RunnableParallel(
                completion=completion_chain, prompt_value=prompt_formatted
            ) | RunnableLambda(lambda x: retry_parser.parse_with_prompt(**x))

            final_prompt = prompt_formatted.format()
            result = validation_chain.invoke({})
            return result, final_prompt

    @staticmethod
    def RunItemResponseGeneration(
        problem,
        prompt: List[Tuple[str]],
        prompt_idx: int,
        llm,
        ethnicity: str = None,
        gender: str = None,
        industry: str = None,
        title: str = None,
        FirstName: str = None,
        LastName: str = None,
        Occuptaion: str = None,
        Field: str = None,
        Psychometric: str = None,
    ):
        prompt_formatted = ChatPromptTemplate.from_messages(prompt)

        chain = prompt_formatted | llm
        if prompt_idx == 0:
            result = chain.invoke({"scenario": problem})
        # depending on the type of item response prompt, different demographices are needed and they map to a different prompt template
        elif prompt_idx == 1:
            result = chain.invoke(
                {
                    "scenario": problem,
                    "ethnicity": ethnicity,
                    "gender": gender,
                    "industry": industry,
                    "title": title,
                }
            )
        elif prompt_idx == 2:
            result = chain.invoke(
                {
                    "scenario": problem,
                    "FirstName": FirstName,
                    "LastName": LastName,
                    "Occupation": Occuptaion,
                    "Field": Field,
                    "Psychometric": Psychometric,
                }
            )

        result = Consequences.CreativityScenarioResponseParser.parse(result)
        return result

    # run scoring for the task
    # to ensure correct behavior, round must always be passed
    # TODO: implement sentiment analysis scorer
    def RunScorers(self, i: int) -> None:
        self.scorer.predict_with_model(
            config["itemResponseOriginalityModelDir"],
            config["itemResponseGenOutputFile"],
            config["shotSelectionMetric"],
            config[
                "itemResponseGenOutputFile"
            ],  # we need to chop off most columns from the first instance, so send another copy to save to
            i,  # round
        )


def init_task(config: dict):
    if config["task"] == "CPS":
        task = CPS()
    elif config["task"] == "consequences":
        task = Consequences()
    return task
