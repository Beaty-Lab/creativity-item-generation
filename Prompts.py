# prompts for item generation and evaluation are kept separate from other scripts.
# New prompts can be added to the appropriate list, and used by passing the index to the prompt in config.py
import importlib
from config import config

spec = importlib.util.spec_from_file_location(config["promptConfig"].split("/")[-1],config["promptConfig"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

item_gen_prompts = module.item_gen_prompts
item_eval_prompts = module.item_eval_prompts
item_response_gen_prompts = module.item_response_gen_prompts
wordlist_gen_prompts = module.wordlist_gen_prompts # TODO: generalize