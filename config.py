# TODO: add experiment config as a dictionary
#         itemResponseEvalModelDir: str, # TODO: is this needed?
#         inputFile: str
config = {
    "numIter": 2,
    "itemGenModelName": "google",
    "useItemEvalModel": True,
    "itemEvalModelName": "google",
    "itemResponseGenModelName": "google",
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
    "itemGenMaxTokens": 500,
    "itemEvalMaxTokens": 2048,
    "itemResponseGenMaxTokens": 500,
    "wordlistFile": "/home/aml7990/Code/creativity-item-generation/outputs/creative_wordlist_5_words_small.tsv",
    "demographicsFile": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/data/DemographicData.csv",
    "itemGenOutputFile": "/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/with_controversial_filter_few_shot/debug_driver_script/debug_test.json",
    "itemEvalOutputFile": "/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/with_controversial_filter_few_shot/debug_driver_script/debug_test.json",
    "itemResponseGenOutputFile": "/home/aml7990/Code/creativity-item-generation/outputs/without_eval_scores/with_controversial_filter_few_shot/debug_driver_script/debug_response_test.json", # should NOT be the same as the top 2
    "numItemsPerList": 3, # TODO: no longer used, delete
    "itemResponseOriginalityModelDir": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model/originality_model_factor_score/",
    "itemResponseQualityModelDir": "/home/aml7990/Code/creativity-item-generation/optimize_item_gen_prompt/scoring_model/quality_model_factor_score/",
    "shotSelectionMetric": "originality", # the metric used to select which items lead to consistently good responses
    "itemGenNumShots": 2,                 # how many shots from the i-th trial to include in the i+1th trial
    "shotSelectionSort": "max",           # how to select the shots to use based on the metric, either "max" or "min" 
}