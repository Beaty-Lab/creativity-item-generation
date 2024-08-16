few_shot_config = {
    "random_seed": 1,
    "OutputFile": "/home/aml7990/Code/creativity-item-generation/scorers/CPS/LLM/outputs/claude-3/",
    "logFile": "/home/aml7990/Code/creativity-item-generation/scorers/CPS/LLM/outputs/claude-3/log.txt",
    "dataset": "/home/aml7990/Code/creativity-item-generation/datasets/cps/cleaned/CPSTfulldataset3-standarized-with-dev.csv",
    "ModelName": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "TrainItem": "Becky",
    "TokenizerName": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "use_sweep": False,
    "MaxTokens": 2048,
    "Temperature": 0.00000000001,
    "TopP": 1.0,
    "NumShots": 20,
    "PromptIdx": 1,
    "use_test_set": False,
    "metric": "FacScoresO",
    "test_set": "test",
    "use_quintiles": True,
    # TODO: Simone: update with a key value pair so I can find it
    # See the sample below, but I don't care what you use.
    # "Simone?": True,
}