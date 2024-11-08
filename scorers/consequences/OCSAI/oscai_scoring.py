import requests
import pandas as pd
import json
import imp
# from oscai_key import oscai_key
# from OscaiKey import oscai_key
oscai_key = imp.load_source("oscai_key", "/home/aml7990/Code/creativity-item-generation/scorers/consequences/OCSAI/OscaiKey.py")

headers = {
    "accept": "application/json",
    "X-API-KEY": oscai_key.oscai_key,
    "Content-Type": "application/x-www-form-urlencoded",
}


def predict_with_model(save_file_name: str, round: int):
    """
    Predict the originality of responses for a given round using the OCSAI model.

    Predictions are saved to a json file with the same name as the input file, but with the round number appended to the name.

    Args:
        save_file_name (str): The name of the file to save the output to.
        round (int): The round number for item generation.
    """
    save_file = pd.read_json(f"{save_file_name}_round_{round}.json")
    save_file[f"originality_round_{round}"] = None
    for index, row in save_file.iterrows():
        question = row[f"creative_scenario_round_{round}"]
        response = row[f"creative_response_round_{round}"]
        params = {
            "model": "ocsai-1.5",
            "input": f"{question},{response}",
            "elab_method": "none",
            "language": "English",
            "task": "consequences",
            "question_in_input": "false",
            "question": question,
        }

        response = requests.post(
            "https://openscoring.du.edu/llm", params=params, headers=headers
        )
        if response.status_code != 200:
            print(f"OSCAI API Failure: {response.status_code} {response.reason}")
            continue
            # save_file.to_json(f"{save_file_name}_round_{round}.json")
            # exit(-1)

        raw_output = dict(json.loads(response.content))
        originality_score = raw_output["scores"][0]["originality"]
        save_file.at[index, f"originality_round_{round}"] = originality_score

    save_file = save_file.dropna(subset=[f"originality_round_{round}"])
    if len(save_file) == 0:
        print(
            f"Scoring failure, please check OSCAI and confirm requests are formatted properly"
        )
        exit(-1)
    save_file.to_json(f"{save_file_name}_round_{round}.json")
