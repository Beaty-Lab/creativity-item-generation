"""
    Executes training for a given scoring model
    Shares the same confid as CPIG
"""
from config import config, peft_config
from scorers.CPS import CPSFinetuneTLM, CPSPeftTLM, CPSPromptedLLM, CPSSetFit
from scorers.consequences import *

# define a mapping from tasks to possible scorers

# TODO: implement consequences
scorer_map = {
    "consequences": {},
    "CPS": {
        "Semantic Distance": None,
        "Finetuned": CPSFinetuneTLM,
        "PEFT": CPSPeftTLM.PeftModel,
        "Prompted LLM": CPSPromptedLLM,
        "SetFit": CPSSetFit
    }
}

if __name__ == "__main__":
    if config["task"] == "consequences":
        print("Not yet implemented for training!")
        exit(-1)
    elif config["task"] == "CPS":
        try:
            Scorer = scorer_map[config["task"]][config["scorerType"]](config, peft_config)
        except Exception as e:
            print(e)
            exit(-1)
    else:
        print("Not a valid task!")
        exit(-1)
    
    print("Starting training!")
    Scorer.fit()
    