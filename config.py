import torch
from peft import LoraConfig, TaskType, PromptEncoderConfig

TASK = "consequences"
config = {
    # must be one of CPS or consequences
    "task": TASK,
    # numeric params
    "random_seed": 333,
    "numIter": 3,
    "itemGenFrequencyPenalty": 0.0,
    "itemEvalFrequencyPenalty": 0.0,
    "itemResponseGenFrequencyPenalty": 0.0,
    "itemGenPresencePenalty": 0.0,
    "itemEvalPresencePenalty": 0.0,
    "itemResponseGenPresencePenalty": 0.0,
    "itemGenTemperature": 1.0,
    "itemEvalTemperature": 0.0,
    "itemResponseGenTemperature": 1.0,
    "itemGenTopP": 1.0,
    "itemEvalTopP": 1.0,
    "itemResponseGenTopP": 1.0,
    "itemGenPromptIdx": 0,
    "itemEvalPromptIdx": 0,
    "itemResponseGenPromptIdx": 0,
    "itemGenMaxTokens": 70,
    "itemEvalMaxTokens": 2048,
    "itemResponseGenMaxTokens": 100,  # will be the same as the max for item gen if using the same model
    "numItemsPerList": 3,
    "numItemGenerationAttempts": 4,
    "itemGenNumShots": 3,
    "numResponsesPerItem": 1,
    # shot selection params
    "EmbeddingModel": "all-mpnet-base-v2",
    # non-numeric params
    "shotSelectionMetric": "originality",
    "shotSelectionSort": "max",
    "shotSelectionAggregate": "mean",
    "shotSelectionAlgorithm": "constraint satisfaction",
    # scoring models dirs
    "itemResponseOriginalityModelDir": "/home/aml7990/Code/creative-judgeLM/model_training/flan-t5-xl-multitask/flan-t5-xl-multitask-originality-scoring/",
    "itemResponseQualityModelDir": "/home/aml7990/Code/creative-judgeLM/model_training/flan-t5-xl-multitask/flan-t5-xl-multitask-originality-scoring/",
    # model dirs and flags
    "itemGenModelName": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "itemEvalModelName": "meta-llama/Llama-3.2-3B-Instruct",
    "itemResponseGenModelName": "meta-llama/Llama-3.2-3B-Instruct",
    "useItemEvalModel": False,
    "useItemScoring": True,  # only set to false if you want to generate a bunch of items without prompt optimization
    "useItemScoringModelPeft": False,
    # dataset dirs
    "wordlistFile": None,  # "/home/aml7990/Code/creativity-item-generation/outputs/creative_wordlist_5_words_small.tsv",
    # if not using a wordlist, must specify the below to dicate how many items to generate on the first pass
    "NumSeedItems": 5,
    # "demographicsFile": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/data/PsychometricData.csv",
    "demographicsFile": None,
    ## prompt config file ##
    "promptConfig": f"/home/aml7990/Code/creativity-item-generation/prompts/{TASK}_prompts.py",
    ## scorer training params ##
    "doEval": False,
    "scorerBaseModel": "meta-llama/Meta-Llama-3-8B-Instruct",
    "pTunedModel": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model/CPS/P-tuning/feasible-sun-764",
    "scorerDataPath": "/home/aml7990/Code/creativity-item-generation/datasets/cps/cleaned/CPSTfulldataset3-standarized.csv",
    "scorerInputColumn": "SolutionsWithProblem",
    "scorerLabelColumn": "FacScoresO",
    "scorerOutputDir": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model/CPS/",
    "scorerType": "P-tuning",
    "use_sweep": False,
    "WandbProjectName": "ARI-year2-scorers",
    "lr": 0.00005,
    "epochs": 10,  # 25
    "batchSize": 1,
    "quantization": "16-bit",  # logging ONLY, not passed to transformers
    # Lora params
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
    # P-tuning params
    "numVirtualTokens": 8,
    "encoderHiddenSize": 512,
    # quantization
    "use_fp16": False,
    # TODO: implment bitsandbytes config, these are ignored right now
    "load_in_8bit": False,
    "load_in_4bit": False,
    "bnb_4bit_quant_type": None,
    "bnb_4bit_use_double_quant": False,
    # "bnb_4bit_compute_dtype": torch.bfloat16,
}
# output dirs
item_gen_model_name = config["itemGenModelName"].split("/")[1]
item_response_gen_model_name = config["itemResponseGenModelName"].split("/")[1]
config["itemGenOutputFile"] = (
    f"/home/aml7990/Code/creativity-item-generation/outputs/{TASK}/{item_gen_model_name}_seed={config['random_seed']}_prompt={config['itemGenPromptIdx']}/items.json"
)
config["itemResponseGenOutputFile"] = (
    f"/home/aml7990/Code/creativity-item-generation/outputs/{TASK}/{item_gen_model_name}_seed={config['random_seed']}_prompt={config['itemGenPromptIdx']}/item_responses"
)
config["itemEvalOutputFile"] = config["itemGenOutputFile"]
config["logFile"] = "log.txt"

if config["scorerType"] == "LoRA":
    peft_config = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=config["lora_target_modules"],
    )
elif config["scorerType"] == "P-tuning":
    peft_config = PromptEncoderConfig(
        task_type="SEQ_CLS",
        num_virtual_tokens=config["numVirtualTokens"],
        encoder_hidden_size=config["encoderHiddenSize"],
    )
# TODO: implement other configs, add quantization config
