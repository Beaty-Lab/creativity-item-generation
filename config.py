import torch
from peft import LoraConfig, TaskType

TASK = "CPS"
config = {
    # must be one of CPS or consequences
    "task": TASK,
    
    # numeric params
    "random_seed": 666,
    "numIter": 1,
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
    "itemGenPromptIdx": 2,
    "itemEvalPromptIdx": 0,
    "itemResponseGenPromptIdx": 0,
    "itemGenMaxTokens": 300,
    "itemEvalMaxTokens": 2048,
    "itemResponseGenMaxTokens": 50, # will be the same as the max for item gen if using the same model
    "numItemsPerList": 3,
    "numItemGenerationAttempts": 3,
    "itemGenNumShots": 4,
    "numResponsesPerItem": 0,

    # shot selection params
    "EmbeddingModel": "all-MiniLM-L6-v2",

    # non-numeric params
    "shotSelectionMetric": "originality",
    "shotSelectionSort": "max",
    "shotSelectionAggregate": "mean",
    "shotSelectionAlgorithm": "constraint satisfaction",

    # scoring models dirs (ignored for consequences which uses OCS)
    "itemResponseOriginalityModelDir": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model/originality_model_factor_score/",
    "itemResponseQualityModelDir": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model/quality_model_factor_score/",

    # model dirs and flags
    "itemGenModelName": "gpt-3.5-turbo",
    "itemEvalModelName": "gpt-3.5-turbo",
    "itemResponseGenModelName": "gpt-3.5-turbo",
    "useItemEvalModel": False,
    "useItemScoring": False, # only set to false if you want to generate a bunch of items without prompt optimization
    "useItemScoringModelPeft": False,

    # dataset dirs
    "wordlistFile": None, #"/home/aml7990/Code/creativity-item-generation/outputs/creative_wordlist_5_words_small.tsv",
    # if not using a wordlist, must specify the below to dicate how many items to generate on the first pass
    "NumSeedItems": 20, 
    # "demographicsFile": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/data/PsychometricData.csv",
    "demographicsFile": None,

    # output dirs
    "itemGenOutputFile": f"/home/aml7990/Code/creativity-item-generation/outputs/{TASK}/gpt-3.5-turbo_seed=666_prompt=2/items.json",
    "itemEvalOutputFile": f"/home/aml7990/Code/creativity-item-generation/outputs/{TASK}/gpt-3.5-turbo_seed=666_prompt=2/items.json",
    "itemResponseGenOutputFile": f"/home/aml7990/Code/creativity-item-generation/outputs/{TASK}/gpt-3.5-turbo_seed=666_prompt=2/item_responses",
    "logFile": f"/home/aml7990/Code/creativity-item-generation/outputs/{TASK}/gpt-3.5-turbo_seed=666_prompt=2/log.txt",

    ## CTransformers ##
    "useCTransformers": False,
    "CTransformersNumGPULayers": 50,
    "CTransformersItemGenTokenizer": "meta-llama/Llama-2-7b-chat-hf",
    "CTransformersItemEvalTokenizer": "meta-llama/Llama-2-7b-chat-hf",
    "CTransformersItemResponseGenTokenizer": "meta-llama/Llama-2-7b-chat-hf",

    ## prompt config file ##
    "promptConfig": f"/home/aml7990/Code/creativity-item-generation/prompts/{TASK}_prompts.py",

    ## scorer training params ##
    "scorerBaseModel": "meta-llama/Meta-Llama-3-8B-Instruct",
    "scorerDataPath": "/home/aml7990/Code/creativity-item-generation/datasets/cps/cleaned/CPSTfulldataset3-standarized.csv",
    "scorerInputColumn": "SolutionsWithProblem",
    "scorerLabelColumn": "FacScoresO",
    "scorerOutputDir": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model/CPS/",
    "scorerType": "PEFT",
    "use_sweep": False,
    "WandbProjectName": "ARI-year2-scorers",
    "lr": 0.00005,
    "epochs": 25, # 25
    "batchSize": 16,

    # Lora params
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q_proj", "v_proj"],

    # quantization
    "use_fp16": True,
    # TODO: implment bitsandbytes config, these are ignored right now
    "load_in_8bit": True,
    "load_in_4bit": False,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    # "bnb_4bit_compute_dtype": torch.bfloat16,


}

peft_config = LoraConfig(
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    bias="none",
    task_type=TaskType.SEQ_CLS,
    target_modules=config["lora_target_modules"],
)
# TODO: implement other configs, add quantization config