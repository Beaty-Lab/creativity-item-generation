# prompts for item generation and evaluation are kept separate from other scripts.
# New prompts can be added to the appropriate list, and used by passing the index to the prompt in config.py
import importlib
from config import config
from typing import List

spec = importlib.util.spec_from_file_location(config["promptConfig"].split("/")[-1],config["promptConfig"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def load_prompts() -> List[str]:
    item_gen_prompt = module.item_gen_prompts[config["itemGenPromptIdx"]]
    item_eval_prompt = module.item_eval_prompts[config["itemEvalPromptIdx"]]
    item_response_gen_prompt = module.item_response_gen_prompts[config["itemResponseGenPromptIdx"]]
    wordlist_gen_prompts = module.wordlist_gen_prompts # TODO: generalize
    return item_gen_prompt, item_eval_prompt, item_response_gen_prompt, wordlist_gen_prompts