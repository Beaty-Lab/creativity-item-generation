few_shot_config = {
    "random_seed": 3,
    "OutputFile": "/home/aml7990/Code/creativity-item-generation/scorers/CPS/LLM/outputs/claude-3/",
    "logFile": "/home/aml7990/Code/creativity-item-generation/scorers/CPS/LLM/outputs/claude-3/log.txt",
    "dataset": "/home/aml7990/Code/creativity-item-generation/datasets/cps/cleaned/CPSTfulldataset3-standarized-with-dev.csv",
    "ModelName": "claude-3",
    "TrainItem": "Becky",
    "TokenizerName": "claude-3",
    "use_sweep": True,
    "MaxTokens": 2048,
    "Temperature": 0,
    "TopP": 1.0,
    "NumShots": 30,
    "PromptIdx": 1, # TODO: Simone: update
    "use_test_set": False,
    "metric": "FacScoresO",
    "test_set": "heldout",
    # TODO: Simone: update with a key value pair so I can find it
    # See the sample below, but I don't care what you use.
    # "Simone?": True,
}