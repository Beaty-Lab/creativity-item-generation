"""
    Use LLMs (LLama-3-70b, Vicuna-70b, Qwen-2, etc.) and prompt engineering to score.
    Try many (100+) shot with API models.
    For all other models, try vanilla prompting, RAG, and structure prompting (CoT, Z-CoT, Self Consistency, Tree / Graph of thoughts, etc.)
    Optionally, explore methods for prompt optimization.
    NOTE: unlike Peft, these are all hard prompts.
"""
# TODO: finish, separateme methods for tuning and inference using sklearn-like API