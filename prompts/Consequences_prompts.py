wordlist_gen_prompts = None
item_gen_prompts = [
    [
        ("system","You are an author tasked with producing scenarios for a short story. You must write scenarios that would fundamentally alter how human life or the world would work. Your scenarios should be grounded in reality as much as possible; avoid scenarios with overtly sci-fi or fantasy elements and instead focus on the scenarios that could be described as 'slice of life'. Follow the format of the example scenarios provided to you, but don't copy the content."),
        ("human","""Think of at least 10 scenarios that would change the way human life or the world works. The scenarios should alter people's daily lives in important ways.  Please describe only the scenarios, and don't hint at any potential implications of the scenarios. Please describe the scenarios in 12 words at most for each one. Here are a few example scenarios:

        1. What would be the result if people no longer needed or wanted sleep?
        2. What would be the results if everyone suddenly lost the ability to read and write?
        3. What would be the result if everyone suddenly lost the sense of balance and were unable to stay in the upright position for more than a moment?
        
        ###
         
        Scenario:""")
    ]
]
item_eval_prompts = None
item_response_gen_prompts = [
    ("system", "You are a participant in an experiment."),
    ("human", """Consider the following sceanrio, and think of as many consequences as possible for what would happen if the scenario were to come true. Reply in a numbered list, and keep each consequences two sentences long at most.

    ###

    Scenario: {scenario}
        
    ###
        
    Consequences:""")
]