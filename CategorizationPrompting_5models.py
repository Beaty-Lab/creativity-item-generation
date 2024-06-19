if '/Users/simone/Desktop/ARIgrant/CategorizationProjects/PromptingARI/ItemCategorization' not in sys.path:
    sys.path.append('/Users/simone/Desktop/ARIgrant/CategorizationProjects/PromptingARI/ItemCategorization')

os.chdir('/Users/simone/Desktop/ARIgrant/CategorizationProjects/PromptingARI/ItemCategorization')

import os
from openai import OpenAI
import anthropic
import pandas as pd
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from OpenAI_APIkey import OPENAI_API_KEY
from OpenAI_APIkey import ANTHROPIC_API_KEY

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration





#######
#####

# Define the function to get the response from OpenAI gpt4
def get_category_gpt4(scenario, systemPrompt, tempvalueset):
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": f"{systemPrompt}"
            },
            {
                "role": "user",
                "content": f"Scenario: {scenario} Category:"
            }
        ],
        temperature=tempvalueset,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    choicepassage = response.choices[0]
    category = choicepassage.message.content

    if "Category: " in category:
        category = category.replace("Category: ", "")

    return category

#######
#####


# Define the function to get the response from OpenAI gpt3.5-turbo
def get_category_gpt3_5_turbo(scenario, systemPrompt, tempvalueset):
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"{systemPrompt}"
            },
            {
                "role": "user",
                "content": f"Scenario: {scenario} Category:"
            }
        ],
        temperature=tempvalueset,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    choicepassage = response.choices[0]
    category = choicepassage.message.content

    if "Category: " in category:
        category = category.replace("Category: ", "")

    return category

#######
#####


# Define the function to get the response from Claude 3 Haiku
def get_category_claude3_haiku(scenario, systemPrompt, tempvalueset):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        system=systemPrompt,
        max_tokens=256,
        temperature=tempvalueset,
        top_p=1,
        messages=[
            {
                "role": "user",
                "content": f"Scenario: {scenario} Category:"
            }
        ]
    )

    choicepassage = response.content[0]
    category = choicepassage.text

    return category

#######
#####


# Define the function to get the response from Claude 3 Haiku
def get_category_claude3_sonnet(scenario, systemPrompt, tempvalueset):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        system=systemPrompt,
        max_tokens=256,
        temperature=tempvalueset,
        top_p=1,
        messages=[
            {
                "role": "user",
                "content": f"Scenario: {scenario} Category:"
            }
        ]
    )


    choicepassage = response.content[0]
    category = choicepassage.text

    return category

#######
#####

def get_category_flant5_base(scenario, systemPrompt, tempvalueset):
    model_name = "google/flan-t5-base"  # depend on prompt res always isn't what expected
    tokenizer_flantfive = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model_flantfive = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float)

    prompt = systemPrompt
    input_prompt =  f"Scenario: {scenario} Category:"
    inputs = tokenizer_flantfive(prompt + input_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model_flantfive.generate(
        **inputs,
        do_sample=True,
        max_length=512,
        temperature=tempvalueset,
        decoder_start_token_id=tokenizer_flantfive.bos_token_id)

    decoded_text = tokenizer_flantfive.batch_decode(outputs, skip_special_tokens=True)[0]

    return decoded_text

#######
#####




file_path = 'Round2Data/round 2 consequences items to rate _ Prompting.csv'
itemstorate = pd.read_csv(file_path)

system_prompt_set = '''You will be given a scenario developed for the consequences task, and must categorize the scenario based on the general themes contained within it.

You must choose from one of these categories: [Fantasy, Sci-fi, Slice of Life]

Here are a few examples of the task:

1. Scenario: What would be the result if dragons roamed the skies? Category: Fantasy
2. Scenario: What would be the result if humans could upload their consciousness into machines? Category: Sci-fi
3. Scenario: What would be the result if humans could upload memories and replay them? Category: Sci-fi
4. Scenario: What would be the result if everyone gains 10 years? Category: Slice of life 
5. Scenario: What would be the result if everyone loses emotions? Category: Slice of life

Only respond with a single category.'''

temperature_value_set = 0.1



functions = {
    'flan_t5_base': get_category_flant5_base,
    'gpt4': get_category_gpt4,
    'gpt3_5_turbo': get_category_gpt3_5_turbo,
    'claude3_haiku': get_category_claude3_haiku,
    'claude3_sonnet': get_category_claude3_sonnet
}

for column_name, func in functions.items():
    itemstorate[column_name] = itemstorate['Scenario'].apply(lambda x: func(x, system_prompt_set, temperature_value_set))

#Save CSV
itemstorate.to_csv('Categorized_5judges_temp0_0_1_prompt2.csv')




itemstorate['flan_t5_base'] = itemstorate['Scenario'].apply(lambda x: get_category_flant5_base(x, system_prompt_set, temperature_value_set))
itemstorate['gpt4'] = itemstorate['Scenario'].apply(lambda x: get_category_gpt4(x, system_prompt_set, temperature_value_set))
itemstorate['gpt3_5_turbo'] = itemstorate['Scenario'].apply(lambda x: get_category_gpt3_5_turbo(x, system_prompt_set, temperature_value_set))
itemstorate['claude3_haiku'] = itemstorate['Scenario'].apply(lambda x: get_category_claude3_haiku(x, system_prompt_set, temperature_value_set))
itemstorate['claude3_sonnet'] = itemstorate['Scenario'].apply(lambda x: get_category_claude3_sonnet(x, system_prompt_set, temperature_value_set))


#GEMINI MODEL?
#TEST THIS MODEL ON THE CATEGORIZATION 
