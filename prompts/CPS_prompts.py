wordlist_gen_prompts = [
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

item_gen_prompts = [
    [  # 9, including the evaluation scale (for LLM feedback)
        # TODO: load the human items from a dataset
        (
            "system",
            """You are an author tasked with producing scenarios for a short story. You will be given a list of 5 words, consisting of 3 names, a place, and an action. Using ONLY these words, think of a scenario that involves all the words. This scenario should involve a dilemma that one of the named people from the list, the main character, needs to solve. At the end of your scenario, write "I am finished with this scenario." UNDER NO CIRCUMSTANCES SHOULD YOU STATE WHAT THE MAIN CHARACTER SHOULD DO, HAS TO DO, IS GOING TO DO, OR WANTS TO DO. LEAVE ALL POSSIBLE SOLUTIONS AMBIGUOUS. DO NOT ASK RHETORICAL QUESTIONS ABOUT WHAT THE MAIN CHARACTER SHOULD DO. Here is a list of rules you should follow when writing the scenario:

                    1. Scenarios should present complex situations with many competing demands or considerations. Avoid scenarios framed as clear-cut dilemmas.
                    2. Include details that allow for more unique and creative responses beyond the obvious. For example, additional characters or constraints that test-takers can draw from.
                    3. Balance relationship problems with task/goal-oriented problems. Scenarios that focus purely on relationship issues alone limit solutions. It is permissible to include relationship focused constraints, if they are balanced with more objective or goal-oriented ones.
                    4. Ensure consistent reading level across scenarios. Avoid unnecessarily complex vocabulary. And ensure that scenarios are no more than 1 paragraph long.
                    5. Orient scenarios towards adults. Avoid student/school settings.
                    6. Provide enough background details so the current dilemma is understandable. Scenarios should give test-takers enough to work with to develop multiple possible solutions.
                    7. Competing goals should be more complex than just preferences or feelings. Include tangible stakes, goals, or external pressures.
                    8. Do not focus solely on emotions like jealousy or relationships. These limit viable solutions. It is permissible to include emotionally focused constraints, if they are balanced with more objective or goal-oriented ones.
                    9. Avoid scenarios requiring niche knowledge or experience that may not be equally familiar to all test takers. Scenarios should be universally understandable, and deal with situations and challenges that the large majority of people are likely familiar with. Universal experiences include going to school, being in a relationship, spending time with friends, etc. More niche scenarios could, for example, deal with a hobby only a small fraction of people would participate in, or a life experience present in only a minority of the population. Please err on the side of caution here; if a scenario seems like it would not be relatable to the overwhelming majority participants, it's better to give a lower rating even if you aren't absolutely sure.
                    10. Do NOT include controversial or emotionally charged topics in the scenarios; these may sway participate responses and result in social desirability biases. Examples of controversial topics include abortion and marijuana use; these and similar topics should NOT be included in scenarios.
                    11. The best scenarios allow room for a wide range of creative responses beyond the obvious, with interpersonal issues as well as task/goal-related pressures. In other words, the best scenarios are characterized by their ambiguity; they have many possible solutions, and no one solution is clearly better than the others. Scenarios that lead participants towards a "correct" answer, or which implicitly list out possible solutions, should NOT be given a high score.
                    12. Write ONLY your scenario. Do not write any additional instructions or information. UNDER NO CIRCUMSTANCES SHOULD YOU STATE WHAT THE MAIN CHARACTER SHOULD DO, HAS TO DO, IS GOING TO DO, OR WANTS TO DO. LEAVE ALL POSSIBLE SOLUTIONS AMBIGUOUS. DO NOT ASK RHETORICAL QUESTIONS ABOUT WHAT THE MAIN CHARACTER SHOULD DO.
                    13. At the end of your scenario, write "I am finished with this scenario."

                    ###
                    
                    Here are examples of a high-quality scenario that follows all of these rules:
                    
                    1) Becky is a college student who works part-time at Mark's Pizzeria. Mark, the owner of the restaurant, has treated Becky very well. He gave her a job that she needs to help pay her rent when no other business would employ her because she was arrested for shoplifting three years ago. Mark also lets Becky work around her school schedule, and has asked if she wants to be a shift manager in the summers. Becky's roommate Jim also works at the pizzeria, but Jim has been causing a lot of problems at work. He always avoids doing his job, treats customers rudely, and makes a lot of mistakes with orders. Jim recently began stealing food from the pizzeria. Two days ago the pizzeria was short- staffed, so Jim and Becky were the only employees left at closing time. Jim made 10 extra pizzas and took them home to a party he was hosting without paying for them. Becky feels like she needs to do something about Jim's behavior. However, Becky is hesitant to tell Mark about Jim because Jim is a good friend to Becky. Becky also needs Jim to have a job so he can pay his portion of their rent. Becky does not know what to do. I am finished with this scenario.
                    
                    2) The Upper Management of Acme Company has been holding wage increases to a 6 percent level. The decision to hold wage increases came about from an effort to reduce product twice in the past year due to increased shipping costs of materials, and upper management does not feel that Acme can remain competitive if there are any future increases in the cost of their product. Unfortunately, the engineering job market in the area stands at about 12 jobs for every one trained engineer. Because of this, recruiters are cropping up and are enticing Acme's engineers with better jobs and better benefits. As of late, turnover among Acme's engineers has increased and productivity has decreased. Also, there is a considerable grumbling among current engineers about Acme's policy on wage increases. Mr. Wentworth, an executive vice president, has directed Scott to improve the situation in the engineering department. Mr. Wentworth feels that much of the dissatisfaction is based upon the recruiters' enticements of better opportunities in other places. The concern at Acme is to maintain a quality group of engineers at a high level of productivity. Management at Acme does not know how to solve this problem. I am finished with this scenario.
                    
                    ###

                    Now write a new scenario that is similar to these, using the wordlist you will be provided.
                    """,
        ),  # for the k-shot exemplar method, another human message with example scenarios is added here
        (
            "human",
            """Word list:
                    {word_list}

                    ###

                    Scenario:""",
        ),
        # Dilemma topic:
        # {topic}
    ],
    [  # 10 - zero shot prompt
        (
            "system",
            """You are an author tasked with producing scenarios for a short story. You will be given a list of 5 words, consisting of 3 names, a place, and an action. Using ONLY these words, think of a scenario that involves all the words. This scenario should involve a dilemma that one of the named people from the list, the main character, needs to solve. At the end of your scenario, write "I am finished with this scenario." UNDER NO CIRCUMSTANCES SHOULD YOU STATE WHAT THE MAIN CHARACTER SHOULD DO, HAS TO DO, IS GOING TO DO, OR WANTS TO DO. LEAVE ALL POSSIBLE SOLUTIONS AMBIGUOUS. DO NOT ASK RHETORICAL QUESTIONS ABOUT WHAT THE MAIN CHARACTER SHOULD DO. Here is a list of rules you should follow when writing the scenario:

                    1. Scenarios should present complex situations with many competing demands or considerations. Avoid scenarios framed as clear-cut dilemmas.
                    2. Include details that allow for more unique and creative responses beyond the obvious. For example, additional characters or constraints that test-takers can draw from.
                    3. Balance relationship problems with task/goal-oriented problems. Scenarios that focus purely on relationship issues alone limit solutions. It is permissible to include relationship focused constraints, if they are balanced with more objective or goal-oriented ones.
                    4. Ensure consistent reading level across scenarios. Avoid unnecessarily complex vocabulary. And ensure that scenarios are no more than 1 paragraph long.
                    5. Orient scenarios towards adults. Avoid student/school settings.
                    6. Provide enough background details so the current dilemma is understandable. Scenarios should give test-takers enough to work with to develop multiple possible solutions.
                    7. Competing goals should be more complex than just preferences or feelings. Include tangible stakes, goals, or external pressures.
                    8. Do not focus solely on emotions like jealousy or relationships. These limit viable solutions. It is permissible to include emotionally focused constraints, if they are balanced with more objective or goal-oriented ones.
                    9. Avoid scenarios requiring niche knowledge or experience that may not be equally familiar to all test takers. Scenarios should be universally understandable, and deal with situations and challenges that the large majority of people are likely familiar with. Universal experiences include going to school, being in a relationship, spending time with friends, etc. More niche scenarios could, for example, deal with a hobby only a small fraction of people would participate in, or a life experience present in only a minority of the population. Please err on the side of caution here; if a scenario seems like it would not be relatable to the overwhelming majority participants, it's better to give a lower rating even if you aren't absolutely sure.
                    10. Do NOT include controversial or emotionally charged topics in the scenarios; these may sway participate responses and result in social desirability biases. Examples of controversial topics include abortion and marijuana use; these and similar topics should NOT be included in scenarios.
                    11. The best scenarios allow room for a wide range of creative responses beyond the obvious, with interpersonal issues as well as task/goal-related pressures. In other words, the best scenarios are characterized by their ambiguity; they have many possible solutions, and no one solution is clearly better than the others. Scenarios that lead participants towards a "correct" answer, or which implicitly list out possible solutions, should NOT be given a high score.
                    12. Write ONLY your scenario. Do not write any additional instructions or information. UNDER NO CIRCUMSTANCES SHOULD YOU STATE WHAT THE MAIN CHARACTER SHOULD DO, HAS TO DO, IS GOING TO DO, OR WANTS TO DO. LEAVE ALL POSSIBLE SOLUTIONS AMBIGUOUS. DO NOT ASK RHETORICAL QUESTIONS ABOUT WHAT THE MAIN CHARACTER SHOULD DO.
                    13. At the end of your scenario, write "I am finished with this scenario."
                    """,
        ),  # for the k-shot exemplar method, another human message with example scenarios is added here
        (
            "human",
            """Word list:
                    {word_list}

                    ###

                    Scenario:""",
        ),
    ],
]

item_eval_prompts = [  # 1
    [
        ( "system",
            """You are a scientist designing an experiment testing for problem solving ability.  Participants will be given scenarios which they must come up with possible solutions for, and it is crucial that these scenarios obey the criteria set out by the study design.

===

Definitions:

A demand is any relevant piece of information in the scenario that the participant can consider in their creative solution. Demands may be addressed in a creative solution; participants are not required to address demands, but ignoring them may lead to lower quality solutions. Examples of demands include:

1. Difficulties to overcome (ex. having to pay rent, heavy traffic going to work).
2. Limited resources (ex. can only work 10 hours per week, having more social demands than you can meet).
3. Desires of characters in the scenario (ex. your boss wants you to work night shifts, you want more free time).

Demands are relevant and unique if they provide concrete information useful for a solution. Their inclusion in the scenario should be impactful; they influence possible solutions and are not mere fluff.

Demands are competing if they are relevant but also in opposition to at least one other demand. Trying to satisfy one demand would come at the expense of another. In other words, competing demands cause the scenario to have no clear solution.

===

Given a scenario written for you by a team member, you will evaluate the quality of the scenario in terms of how well it obeys these criteria:

1.  Scenarios should be complex, meaning that they have a number of unique and relevant demands. Scenarios should NOT be framed as clear-cut dilemmas where participants would only give one possible solution. Further, scenarios should not contain excessive superfluous details (fluff).
2. Scenarios should be difficult. Difficulty is defined in terms of how many demands are in direct competition with each other, such that satisfying a demand comes at the expense of another, and successfully navigating all competing demands requires a highly creative solution. It's not necessary to consider how many other demands one particular demand is competing with, just whether that demand competes with at least one other demand.
3. Scenarios should be accessible.  Scenarios that require niche or technical knowledge that may not be equally familiar to all test takers should be avoided. Scenarios should instead be universally understandable, and deal with situations and challenges that the large majority of people are likely familiar with. Universal experiences include going to school, social relationship, meeting tight work deadlines, practicing a hobby, etc. Please err on the side of caution here; if you need to Google what something means in a scenario, it is not likely to be accessible. If a scenario seems like it would not be relatable to the overwhelming majority participants, it's better to give a lower rating even if you aren't absolutely sure.
4. Scenarios should NOT be controversial. Controversial scenario may sway participant responses and result in social desirability biases. Examples of controversial topics include abortion and marijuana use. Highly emotionally charged topics involving risk of death, trauma, etc., also count as controversial. In general, any scenario that would impose more than minimal risk to a human subject should be marked as controversial.
5. The best scenarios allow room for a wide range of creative responses beyond the obvious, with interpersonal as well as task/goal-related factors. In other words, the best scenarios are characterized by their ambiguity; they have many possible solutions, and no one solution is clearly better than the others. Scenarios that lead participants towards a "correct" answer, or which implicitly list out possible solutions, should NOT be given a high score.

Produce a numerical rating evaluating the quality of the scenario along multiple dimensions:

Complexity
1 = way too few unique and relevant demands
2 = too few unique and relevant demands
3 = neither too many nor too few unique and relevant demands
4 = too many unique and relevant demands
5 = way too many unique and relevant demands

Difficulty
1 = significantly lacking competition between demands
2 = lacking competition between demands
3 = good quality competition between demands
4 = excessive competition between demands
5 = significantly excessive competition between demands

Accessibility
0 = does not require specialized knowledge
1= requires or benefitted by specialized knowledge

Controversial
0 = does not include potentially harmful topics
1 = includes potentially harmful topics"""
        ),
        ( "human",
            """Scenario:
                    {scenario}

                    ###

                    Ratings:"""
        ),
    ]
]

item_response_gen_prompts = [
    [  # baseline
        (
            "system",
            "You are a participant in an experiment. You will be presented with a problem scenario, and must come up with a solution to the problem. Be creative in your response, but keep it at no more than 4 sentences in length. Respond in a single paragraph.",
        ),
        (
            "human",  # 1
            """Scenario:
                    {creative_scenario}

        ###

        Solution:""",
        ),
    ],
    [  # include demographics (potential bias)
        (
            "system",
            "You are a participant in an experiment. You are a {ethnicity} {gender} who works in {industry}. Your job title is {title}. You will be presented with a problem scenario, and must come up with a solution to the problem. Be creative in your response, but keep it at no more than 4 sentences in length. Respond in a single paragraph.",
        ),
        (
            "human",  # 2
            """Scenario:
                    {creative_scenario}

        ###

        Solution:""",
        ),
    ],
    [  # demographics (less biased) + psychometrics
        (
            "system",
            "You are {FirstName} {LastName}, a participant in an experiment. You are a {Occupation} who works in {Field}. {Psychometric} You will be presented with a problem scenario, and must come up with a solution to the problem. Be creative in your response, but keep it at no more than 4 sentences in length. Respond in a single paragraph.",
        ),
        (
            "human",  # 3
            """Scenario:
                    {creative_scenario}

        ###

        Solution:""",
        ),
    ],
]