import json
import os
from typing import Tuple

from fire import Fire

from .dataloader import *
from .utils import (
    load_vllm_model,
    process_prompts_instruct,
    str_to_dict,
)


def main(
    tasks: str,
    model_path: str,
    sampling_params: str,
    output_path: str,
) -> None:
    """
    Run the full answer generation pipeline for one or more tasks.

    Parameters
    ----------
    tasks : str
        Semicolon-separated string of task names. Each task is expected to be callable
        and return a tuple where the last element contains the list of prompts.
    model_path : str
        Path to the model to load.
    sampling_params : str
        Sampling parameters as a comma-separated 'key=value' string (e.g., "top_k=10,temperature=0.7").
    output_path : str
        Base directory to save generated answers.

    Returns
    -------
    None
        Generated answers are saved to JSON files under `output_path`.
    """
    print("""
===================================
========== LOADING MODEL ==========
===================================
""")

    sampling_params = str_to_dict(sampling_params)
    model, sampling_params = load_vllm_model(model_path, sampling_params)

    print("""
===============================================
========== MODEL LOADED SUCCESSFULLY ==========
===============================================
""")

    print("""
========================================
========== GENERATING ANSWERS ==========
========================================
""")

    tasks = tasks.split(";")
    model_name = model_path.split("/")[-1]

    for task in tasks:
        save_path = f"{output_path}/{task}/{model_name}"

        if os.path.exists(f"{save_path}/answers.json"):
            print(f"Generated answers already stored at {save_path}/answers.json")
            continue

        print(f"\n*** Generating answers for task {task} ***\n")

        try:
            prompts = eval(task)()[-2]
        except Exception as e:
            print(f"Error: {e}")
            continue

        processed_prompts = process_prompts_instruct(
            prompts, model_path, model.llm_engine.model_config.max_model_len
        )
        print(f"\n---\nPrompt example:\n\n{processed_prompts[0]}\n---\n")

        try:
            outputs = model.generate(processed_prompts, sampling_params)
        except Exception as e:
            print(f"Error: {e}")
            continue

        generated_answers = [output.outputs[0].text for output in outputs]
        os.makedirs(save_path, exist_ok=True)

        with open(f"{save_path}/answers.json", "w") as f:
            json.dump(generated_answers, f, ensure_ascii=False, indent=4)

        print(f"Generated answers saved at {save_path}/answers.json")

    print("""
=====================================
========== GENERATION DONE ==========
=====================================
""")


if __name__ == "__main__":
    Fire(main)
