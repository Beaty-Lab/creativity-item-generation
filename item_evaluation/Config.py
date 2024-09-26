few_shot_config = {
    "random_seed": 2,
    "TrainSet": "/home/aml7990/Code/creativity-item-generation/item_evaluation/cleaned_datasets/CPS_train.csv",
    "TestSet": "/home/aml7990/Code/creativity-item-generation/item_evaluation/cleaned_datasets/CPS_test.csv",
    # "ModelName": "/home/aml7990/Code/creative-judgeLM/model_training/multitask-LLM-checkpoint-2446",
    "ModelName": "meta-llama/Llama-3.1-8B-Instruct",
    "TokenizerName": "meta-llama/Llama-3.1-8B-Instruct",
    "MaxTokens": 4096,
    "Temperature": 0.0,
    "TopP": 1.0,
    "NumShots": 5,
    "PromptIdx": 1,
    "use_human_test_set": False,  # false = dev set
    "label": "complexity_aggregrate", # ignored for multitask if test_task != CPS
    "human_test_set": "test",
}