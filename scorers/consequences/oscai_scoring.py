import requests
import pandas as pd
import json

# from oscai_key import oscai_key
# TODO: DELETE!
oscai_key = "ocs-V0hmFRFqmFP52rdMg_SGn43k0NhHcTa158bm9FjNZTM"

headers = {
    'accept': 'application/json',
    'X-API-KEY': oscai_key,
    'Content-Type': 'application/x-www-form-urlencoded',
}

def predict_with_model(save_file_name: str, round: int):
    save_file = pd.read_json(f"{save_file_name}_round_{round}.json")
    save_file[f"originality_round_{round}"] = None
    for index, row in save_file.iterrows():
        question = row[f"creative_scenario_round_{round}"]
        response = row[f"creative_response_round_{round}"]
        params = {
            'model': 'ocsai-1.5',
            'input': f'{question},{response}',
            'elab_method': 'none',
            'language': 'English',
            'task': 'consequences',
            'question_in_input': 'false',
            'question': question,
        }

        response = requests.post('https://openscoring.du.edu/llm', params=params, headers=headers)
        # TODO: add error handling if API returns no response
        raw_output = dict(json.loads(response.content))
        originality_score = raw_output["scores"][0]['originality']
        save_file.at[index, f"originality_round_{round}"] = originality_score
    
    save_file.to_json(f"{save_file_name}_round_{round}.json")