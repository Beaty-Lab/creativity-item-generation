wordlist_gen_prompts = None
item_gen_prompts = [
    [
        (
            "system",
            "You are an author tasked with producing scenarios for a short story. You must write scenarios that would fundamentally alter how human life or the world would work. Your scenarios should be grounded in reality as much as possible; avoid scenarios with overtly sci-fi or fantasy elements and instead focus on the scenarios that could be described as 'slice of life'. Follow the format of the example scenarios provided to you, but don't copy the content.",
        ),
        (
            "human",
            """Think of at least 10 scenarios that would change the way human life or the world works. The scenarios should alter people's daily lives in important ways.  Please describe only the scenarios, and don't hint at any potential implications of the scenarios. Please describe the scenarios in 12 words at most for each one.
        
        ###
         
        Scenario:""",
        ),
    ],
    [
        (
            "system",
            "You are an author tasked with producing scenarios for a short story. You must write scenarios that would fundamentally alter how human life or the world would work. Use as inspiration for your scenarios elements from works of high fantasy (Game of Thrones, The Hobbit, etc.), but make sure the scenarios remain grounded in reality as much as possible. Begin each scenario with 'What would be the result if'.",
        ),
        (
            "human",
            """Think of at least 10 scenarios that would change the way human life or the world works. The scenarios should alter people's daily lives in important ways.  Please describe only the scenarios, and don't hint at any potential implications of the scenarios. Please describe the scenarios in 12 words at most for each one.
        
        ###
         
        Scenario:""",
        ),
    ],
    [
        (
            "system",
            "YYou are an author tasked with producing scenarios for a short story. You must write scenarios that would fundamentally alter how human life or the world would work. Use as inspiration for your scenarios elements from works of science fiction (Star Trek, Star Wars, The Martian Chronicles, etc.), but make sure the scenarios remain grounded in reality as much as possible. Begin each scenario with 'What would be the result if'.",
        ),
        (
            "human",
            """Think of at least 10 scenarios that would change the way human life or the world works. The scenarios should alter people's daily lives in important ways.  Please describe only the scenarios, and don't hint at any potential implications of the scenarios. Please describe the scenarios in 12 words at most for each one.
        
        ###
         
        Scenario:""",
        ),
    ]
]

# TODO: empty only for API compatability, implement
item_eval_prompts = [("system", ""), ("human", """""")]
# TODO: add support for demographic and psychometric prompts
item_response_gen_prompts = [
    [
        ("system", "You are a participant in an experiment."),
        (
            "human",
            """Consider the following scenario, and think of as many consequences as possible for what would happen if the scenario were to come true. Reply in a numbered list, and keep each consequences two sentences long at most.

    ###

    Scenario: {scenario}
        
    ###
        
    Consequences:""",
        ),
    ]
]
