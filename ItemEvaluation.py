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
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# API key stored in key.py, and should NOT be committed
from key import key
from tqdm import tqdm
from random import randint
from argparse import ArgumentParser
from nltk import word_tokenize


class CreativityScenarioItemEvaluationParser(BaseOutputParser):
    # the output should be formatted as json
    @staticmethod
    def parse(text: str) -> dict:
        try:
            js.loads(text)
        except Exception:
            print("Json output failed to load, trying next item...")
            return None

        return text


def test_creative_problem_eval(prompt_idx: int, scenario: str, llm):
    system_context = """You are a scientist designing an experiment testing for problem solving ability.  Participants will be given scenarios which they must come up with possible solutions for, and it is crucial that these scenarios obey the criteria set out by the study design. Given a scenario written for you by a team member, you will evaluate the quality of the scenario in terms of how well it obeys these criteria:

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

                    Provide your response in JSON."""

    output = llm.predict_messages(
        [
            SystemMessage(content=system_context),
            HumanMessage(
                content="""Scenario:
                    Noah is a college student who has been working at a gardening store for a year. His boss, James, has been pushing him to take on more responsibilities and become a manager, but Noah is hesitant. He values his friendship with Lily, who works at the store too, but she is not interested in becoming a manager. Noah is torn between his loyalty to James, who has been a mentor to him, and his friendship with Lily, who he knows will be negatively affected if he takes the promotion. Moreover, Noah is not sure if he wants the added stress and responsibility of being a manager, and he worries that taking the promotion would impact his schoolwork. He also knows that James is under pressure from the store's owner to increase profits, and Noah does not want to let him down. Noah does not know what to do.

                    ###

                    Ratings:"""
            ),
            AIMessage(
                content="""{
                        "Complexity": 3,
                        "Open-endedness": 3,
                        "Constraints": 3,
                        "Relationships": 2,
                        "Accessibility" 3,
                        "Emotional Focus": 2,
                        "Controversial": 3
                    }"""
            ),
            HumanMessage(
                content="""Scenario:
                    Alex is in the office, typing away at her computer when Lisa walks in and sits down across from her. Lisa looks worried, and Alex can tell that something is bothering her. Lisa starts to explain that she has been struggling with a project and is worried that she won't be able to meet the deadline. Alex listens intently, trying to offer support and suggestions where she can. However, as Lisa continues to talk, Alex starts to feel a sense of dread creeping in. She knows that if Lisa doesn't finish the project on time, it will reflect poorly on their boss, Ethan, who is also Alex's boyfriend. Alex doesn't know what to do - should she help Lisa and risk appearing to take sides, or should she avoid getting involved and risk damaging her relationship with Ethan? 

                    ###

                    Ratings:"""
            ),
            AIMessage(
                content="""{
                    "Complexity": 2,
                    "Open-endedness": 3,
                    "Constraints" 1,
                    "Relationships": 2,
                    "Accessibility": 3,
                    "Emotional Focus": 2,
                    "Controversial": 3
                    }"""
            ),
            HumanMessage(
                content="""Scenario:
                    Ava is a senior in college who has always been very close to her family. She has recently landed a job at a prestigious finance firm, which would require her to relocate to a different city. Her younger brother, William, who is still in high school, has been struggling with his studies and has confided in Ava about his desire to drop out of school. Ava is torn between her career ambitions and her family responsibilities. She knows that if she takes the job, she will not be able to support her brother the way she wants to. On the other hand, if she passes up the job, she risks jeopardizing her own future. Ava is unsure of what to do and feels overwhelmed by the weight of her decision.

                    ###

                    Ratings:"""
            ),
            AIMessage(
                content="""{
                    "Complexity": 2,
                    "Open-endedness": 3,
                    "Constraints": 2,
                    "Relationships": 2,
                    "Accessibility": 3,
                    "Emotional Focus": 2,
                    "Controversial": 3
                    }"""
            ),
            HumanMessage(
                content="""Scenario:
                    Amelia has been secretly crushing on Benjamin for months. One day, while browsing at the bookstore, she discovers a book that she knows he has been wanting. She is torn between buying it for him and keeping her crush a secret, or leaving it behind and risking the possibility of never having the opportunity to reveal her true feelings. As she stands there, she notices Lily, her best friend, walking towards her. Lily is also Benjamin's ex-girlfriend, and Amelia knows that if she buys the book, Lily will likely find out about her crush. Amelia is unsure of what to do, as she values her friendship with Lily but also really wants to take the opportunity to express her feelings to Benjamin.

                    ###

                    Ratings:"""
            ),
            AIMessage(
                content="""{
                    "Complexity": 2,
                    "Open-endedness": 3,
                    "Constraints": 2,
                    "Relationships": 3,
                    "Accessibility": 3,
                    "Emotional Focus": 3,
                    "Controversial": 2
                    }"""
            ),
            HumanMessage(
                content=f"""Scenario:
                    {scenario}

                    ###

                    Ratings:"""
            ),
        ]
    )

    output = CreativityScenarioItemEvaluationParser().parse(output.content)

    return output


def evaluate_scenarios(
    prompt_idx: int,
    output_file: str,
    model_name: str,
    llm,
    round: int,
    scenario_col: str,
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
                row[scenario_col],  # "creative_scenario_without_feedback"
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
                row[scenario_col],  # "creative_scenario_with_feedback"
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
        temperature = 0  # output should be as close to deterministic as possible
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
