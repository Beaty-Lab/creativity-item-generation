import requests
import pandas as pd

from oscai_key import oscai_key

headers = {
    'accept': 'application/json',
    'X-API-KEY': oscai_key,
    'Content-Type': 'application/x-www-form-urlencoded',
}

def predict_with_model(save_file: str, round: int):
    save_file = pd.read_json(f"{save_file}_round_{round}.json")
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
        originality_score = response["scores"]["originality"]
        save_file.at[index, f"originality_round_{round}"] = originality_score
    
    save_file.to_json(f"{save_file}_round_{round}.json")