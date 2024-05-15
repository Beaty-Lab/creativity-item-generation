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
    "itemGenTemperature": 0.7,
    "itemEvalTemperature": 0.0,
    "itemResponseGenTemperature": 1.0,
    "itemGenTopP": 1.0,
    "itemEvalTopP": 1.0,
    "itemResponseGenTopP": 1.0,
    "itemGenPromptIdx": 8,
    "itemEvalPromptIdx": 0,
    "itemResponseGenPromptIdx": 0,
    "itemGenMaxTokens": 768,
    "itemEvalMaxTokens": 2048,
    "itemResponseGenMaxTokens": 350,
    "numItemsPerList": 3,
    "numItemGenerationAttempts": 3,
    "itemGenNumShots": 1,
    "numResponsesPerItem": 1,

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
    "itemGenModelName": "Locutusque/llama-3-neural-chat-v2.2-8B",
    "itemEvalModelName": "Locutusque/llama-3-neural-chat-v2.2-8B",
    "itemResponseGenModelName": "Locutusque/llama-3-neural-chat-v2.2-8B",
    "useItemEvalModel": False,
    "useItemScoringModelPeft": False,

    # dataset dirs
    "wordlistFile": "/home/aml7990/Code/creativity-item-generation/outputs/creative_wordlist_5_words_small.tsv",
    "demographicsFile": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/data/PsychometricData.csv",

    # output dirs
    "itemGenOutputFile": "/home/aml7990/Code/creativity-item-generation/outputs/consequences_test/items.json",
    "itemEvalOutputFile": "/home/aml7990/Code/creativity-item-generation/outputs/consequences_test/items.json",
    "itemResponseGenOutputFile": "/home/aml7990/Code/creativity-item-generation/outputs/consequences_test/item_responses",
    "logFile": "/home/aml7990/Code/creativity-item-generation/outputs/consequences_test/log.txt",

    # CTransformers
    "useCTransformers": False,
    "CTransformersNumGPULayers": 50,
    "CTransformersItemGenTokenizer": "meta-llama/Llama-2-7b-chat-hf",
    "CTransformersItemEvalTokenizer": "meta-llama/Llama-2-7b-chat-hf",
    "CTransformersItemResponseGenTokenizer": "meta-llama/Llama-2-7b-chat-hf",

    # prompt config file
    "promptConfig": f"/home/aml7990/Code/creativity-item-generation/prompts/{TASK}_prompts.py"
}