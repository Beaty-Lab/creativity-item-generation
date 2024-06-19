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