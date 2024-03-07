config = {

    "random_seed": 777,
    "numIter": 5,
    "itemGenModelName": "meta-llama/Llama-2-70b-chat-hf",
    "useItemEvalModel": False,
    "useItemResponseEvalModel": True,
    "itemEvalModelName": "meta-llama/Llama-2-70b-chat-hf",
    "itemResponseGenModelName": "meta-llama/Llama-2-7b-chat-hf",
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
    "itemGenPromptIdx": 8,
    "itemEvalPromptIdx": 0,
    "itemResponseGenPromptIdx": 0,
    "itemGenMaxTokens": 768,
    "itemEvalMaxTokens": 2048,
    "itemResponseGenMaxTokens": 350,
    "wordlistFile": "/home/aml7990/Code/creativity-item-generation/outputs/creative_wordlist_5_words.tsv",
    "demographicsFile": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/data/DemographicData.csv",
    "itemGenOutputFile": "/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/with_controversial_filter_few_shot/llama_70b_item_gen_llama_70b_eval_max_gen_tokens_350_4_exemplars_mean_seed_777/items.json",
    "itemEvalOutputFile": "/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/with_controversial_filter_few_shot/llama_70b_item_gen_llama_70b_eval_max_gen_tokens_350_4_exemplars_mean_seed_777/items.json",
    "itemResponseGenOutputFile": "/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/with_controversial_filter_few_shot/llama_70b_item_gen_llama_70b_eval_max_gen_tokens_350_4_exemplars_mean_seed_777/item_responses", # should NOT be the same as the top 2
    "numItemsPerList": 3, # TODO: no longer used
    "numItemGenerationAttempts": 3, # this value times 3
    "itemResponseOriginalityModelDir": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model/originality_model_factor_score/",
    "itemResponseQualityModelDir": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model/quality_model_factor_score/",
    "shotSelectionMetric": "originality", # the metric used to select which items lead to consistently good responses
    "itemGenNumShots": 4,                 # how many shots from the i-th trial to include in the i+1th trial
    "shotSelectionSort": "max",           # how to select the shots to use based on the metric, either "max" or "min" 
    "shotSelectionAggregate": "mean",
    "logFile": "/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/with_controversial_filter_few_shot/llama_70b_item_gen_llama_70b_eval_max_gen_tokens_350_4_exemplars_mean_seed_777/log.txt"
}