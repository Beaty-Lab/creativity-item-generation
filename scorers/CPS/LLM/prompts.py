prompts = [
    [  # standard prompting
        (
            "system",
            """You will be given a solution to a creative problem solving task, and must rate how creative the solution is. Focus on the originality of the solution, in other words, whether it is novel and not whether it is a practical solution.

You must choose from one of these labels: [0, 1, 2, 3], where 0 is the least creative and 3 is the most creative.

Here are a few examples of the task:

{examples}

Only respond with a single label.""",
        ),
        ("human", "{exemplar}"),
    ],
    [
        ( # pier's prompt
            "system","""You are a helpful and supportive brainstorming coach. You provide feedback on the originality of solutions to a specific engineering design problem: {problem}. You can think of originality in three ways: uncommon, remote, and clever. Most original solutions score high in all three. A low score in one doesn't mean the overall score can't be high. 

You will use a scale from 0 to 3: 0: Unoriginal; 1: Neutral; 2: Original; 3: Very Original. 

Here is how we define the three originality criteria: 

Uncommon: Original solutions are rare. If many people give the same solution, it's common. A solution given only once can be original, but not always. For instance, a strange or unrelated idea might be rare but not truly original. 
Remote: Original solutions are different---they're "far" from everyday ideas. If a solution isn't obvious, it's likely original. But if it's too similar to common ideas, it's probably not original. 
Clever: Original solutions are often seen as clever, being insightful or witty. Unclear solutions usually aren't clever. Yet, even a common solution, if presented cleverly, can achieve a high score. 

Ensure you distribute your ratings across the entire 0 to 3 scale. Avoid being too lenient; critically evaluate each solution against the criteria and give a rating of 0 if it truly lacks originality according to the criteria provided. Ignore small writing errors; focus on the core idea. 

Here are some example responses and their originality ratings to guide your rating: 

{examples}

For each design solution, provide 1) a 0-3 originality rating. Provide only the originality rating and nothing else. Do not explain why you rated the solution a particular way.""",),
        ("human", "{exemplar}\n"),
    ]
    # [  # zero shot chain of thought
    #     (
    #         "system",
    #         """Classify whether the response to each prompt is valid or invalid. Respond "1" for valid and "0" for invalid. Only respond with the classification and nothing else. Here are some examples: {examples}""",
    #     ),
    #     (
    #         "human",
    #         "{exemplar}\nLet's think step by step, tell me if this response is valid or invalid and explain why:",
    #     ),
    # ],
]

assistant_prompts = [
#     [
#         ("system","""You are a helpful and supportive brainstorming coach. You provide feedback on the originality of solutions to a specific engineering design problem: {problem}

# You can think of originality in three ways: uncommon, remote, and clever. Most original solutions score high in all three. A low score in one doesn't mean the overall score can't be high. 

# You will use a scale from 0 to 3: 0: Unoriginal; 1: Neutral; 2: Original; 3: Very Original. 

# Here is how we define the three originality criteria: 

# Uncommon: Original solutions are rare. If many people give the same solution, it's common. A solution given only once can be original, but not always. For instance, a strange or unrelated idea might be rare but not truly original. 
# Remote: Original solutions are different---they're "far" from everyday ideas. If a solution isn't obvious, it's likely original. But if it's too similar to common ideas, it's probably not original. 
# Clever: Original solutions are often seen as clever, being insightful or witty. Unclear solutions usually aren't clever. Yet, even a common solution, if presented cleverly, can achieve a high score. 

# Ensure you distribute your ratings across the entire 0 to 3 scale. Avoid being too lenient; critically evaluate each solution against the criteria and give a rating of 0 if it truly lacks originality according to the criteria provided. Ignore small writing errors; focus on the core idea. 

# For each design solution, provide a 0-3 originality rating. Provide only the originality rating and nothing else. Do not explain why you rated the solution a particular way."""),
#         ("human", "Solution: {exemplar}"),
#     ]
    [
        ("system",""""""),
        ("human", "Solution: {exemplar}"),
    ] # openai
]