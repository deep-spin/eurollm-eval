import json
import os
import random
from typing import Any, Dict, List

from fire import Fire
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from .dataloader import *
from .utils import (
    load_vllm_model_for_loglik,
    process_prompts_base,
)


def evaluate(
    questions: List[str],
    options: List[List[str]],
    answers: List[int],
    model: LLM,
    tokenizer_path: str,
    sampling_params: SamplingParams,
) -> List[Dict[str, Any]]:
    """
    Evaluate model predictions using log-likelihood over candidate options.

    Parameters
    ----------
    questions : list of str
        List of input questions/prompts.
    options : list of list of str
        Candidate answer options for each question.
    answers : list of int
        Ground-truth answer indices for each question.
    model : LLM
        vLLM model used for evaluation.
    tokenizer_path : str
        Path to the tokenizer used for prompt processing.
    sampling_params : SamplingParams
        Sampling parameters for model generation.

    Returns
    -------
    list of dict
        Assessment for each question containing:
        - "question": question text
        - "options": list of options
        - "ground_truth": ground-truth index
        - "log_likelihoods": list of log-likelihoods per option
        - "model_answer": predicted index
        - "match": 1 if model answer matches ground truth, else 0
    """
    print(f"---\nPrompt example:\n\n{questions[0]}\n---\n")
    prompts = [
        question + option
        for question, _options in zip(questions, options)
        for option in _options
    ]
    prompts = process_prompts_base(
        prompts, tokenizer_path, model.llm_engine.model_config.max_model_len
    )

    prompt_ids, counter = [], 0
    for _options in options:
        prompt_ids.append(list(range(counter, counter + len(_options))))
        counter += len(_options)

    try:
        outputs = model.generate(prompts, sampling_params)
    except Exception as e1:
        print(f"An error ocurred: {e1}; retrying with flexible generation")
        outputs = []
        for prompt in tqdm(prompts, desc="Generating outputs"):
            try:
                outputs.append(
                    model.generate([prompt], sampling_params, use_tqdm=False)[0]
                )
            except Exception as e2:
                print(f"Sample error: {e2}; skipping")
                outputs.append(None)

    logliks = []
    for output in outputs:
        if output is not None:
            logprobs = []
            for token_id, logprob_dict in zip(
                output.prompt_token_ids, output.prompt_logprobs
            ):
                if logprob_dict is not None:
                    logprobs.append(logprob_dict[token_id].logprob)
            logliks.append(np.mean(logprobs))
        else:
            logliks.append(-np.inf)

    logliks = [[logliks[i] for i in _prompt_ids] for _prompt_ids in prompt_ids]
    model_answers = [np.argmax(_logliks).item() for _logliks in logliks]
    assessments = [
        {
            "question": questions[i],
            "options": options[i],
            "ground_truth": answers[i],
            "log_likelihoods": logliks[i],
            "model_answer": model_answers[i],
            "match": (model_answers[i] == answers[i]) * 1,
        }
        for i in range(len(questions))
    ]

    return assessments


def main(
    tasks: str,
    model_path: str,
    output_path: str,
) -> None:
    """
    Run log-likelihood evaluation for one or more tasks and save results.

    Parameters
    ----------
    tasks : str
        Semicolon-separated string of task names. Each task should be callable and
        return a tuple: (questions, options, answers).
    model_path : str
        Path to the model to load.
    output_path : str
        Base directory to save evaluation results and samples.

    Returns
    -------
    None
        Results and sample assessments are saved as JSON files under `output_path`.
    """
    print("""
===================================
========== LOADING MODEL ==========
===================================
""")

    model, sampling_params = load_vllm_model_for_loglik(model_path)
    model_name = model_path.split("/")[-1]

    print("""
===============================================
========== MODEL LOADED SUCCESSFULLY ==========
===============================================
""")

    print("""
========================================
========== EVALUATING ANSWERS ==========
========================================
""")

    tasks = tasks.split(";")

    for task in tasks:
        save_path = f"{output_path}/{task}/{model_name}"

        if os.path.exists(f"{save_path}/results.json"):
            print(f"Assessment file already stored at {save_path}/results.json")
            continue

        print(f"\n*** Evaluating with log-likelihood for task {task} ***\n")

        try:
            questions, options, answers = eval(task)()
        except Exception as e:
            print(f"Error: {e}")
            continue

        try:
            assessments = evaluate(
                questions, options, answers, model, model_path, sampling_params
            ) 
        except Exception as e:
            print(f"Error: {e}")
            continue

        scores = [assessment["match"] for assessment in assessments]
        random.seed(0)
        sampled_ids = random.sample(range(len(questions)), min(100, len(questions)))
        samples = [
            {
                "id": i,
                "question": questions[i],
                "options": options[i],
                "ground_truth": answers[i],
                **assessments[i],
            }
            for i in sampled_ids
        ]
        os.makedirs(save_path, exist_ok=True)

        with open(f"{save_path}/samples.json", "w") as f:
            json.dump(samples, f, ensure_ascii=False, indent=4)

        with open(f"{save_path}/results.json", "w") as f:
            json.dump({"score": np.mean(scores)}, f, ensure_ascii=False, indent=4)

        print(f"Assessment files saved at {save_path}")

    print("""
=====================================
========== EVALUATION DONE ==========
=====================================
""")


if __name__ == "__main__":
    Fire(main)
