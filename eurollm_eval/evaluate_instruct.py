import json
import os
import random
from typing import Any, Dict, List, Union

from comet import load_from_checkpoint
from fire import Fire
from vllm import LLM, SamplingParams

from .dataloader import *
from .utils import (
    load_vllm_model,
    process_prompts_instruct,
    str_to_dict,
)

EVAL_PROMPT_TEMPLATES = {
    "default": """You are an evaluator. Your task is to determine whether the GENERATED ANSWER is equivalent in meaning to the GROUND TRUTH answer, given the QUESTION.
If the GENERATED ANSWER is empty, consider it not equivalent.
End your response with "Answer: True" if the GENERATED ANSWER and GROUND TRUTH convey the same meaning, and "Answer: False" otherwise.

QUESTION:
{question}

GENERATED ANSWER:
{generated_answer}

GROUND TRUTH:
{ground_truth}""",
    "ifeval": """You are an evaluator. Your task is to determine whether the GENERATED ANSWER fully complies with the given INSTRUCTION.
End your reponse with "Answer: True" if the GENERATED ANSWER strictly follows the INSTRUCTION, and "Answer: False" otherwise.

INSTRUCTION:
{question}

GENERATED ANSWER:
{generated_answer}{ground_truth}""",
}


def evaluate(
    task: str,
    questions: List[str],
    ground_truths: List[str],
    answers: List[str],
    judge: LLM,
    tokenizer_path: str,
    sampling_params: SamplingParams,
) -> List[Dict[str, Union[str, int]]]:
    """
    Evaluate generated answers using an instruction-based judge model.

    Parameters
    ----------
    task : str
        Task name to select prompt template.
    questions : list of str
        Original questions/prompts.
    ground_truths : list of str
        Ground-truth answers.
    answers : list of str
        Generated answers to be evaluated.
    judge : LLM
        vLLM judge model.
    tokenizer_path : str
        Path to the tokenizer for prompt processing.
    sampling_params : SamplingParams
        Sampling parameters for the judge model.

    Returns
    -------
    list of dict
        Assessments containing:
        - "assessment": model-generated evaluation text
        - "match": 1 if assessment contains "True", else 0
    """
    prompt_template = EVAL_PROMPT_TEMPLATES[
        task if task in EVAL_PROMPT_TEMPLATES else "default"
    ]
    prompts = [
        prompt_template.format(
            question=question,
            generated_answer=answer,
            ground_truth=ground_truth,
        )
        for question, answer, ground_truth in zip(questions, answers, ground_truths)
    ]
    processed_prompts = process_prompts_instruct(
        prompts, tokenizer_path, judge.llm_engine.model_config.max_model_len
    )
    print(f"\n---\nPrompt example:\n\n{processed_prompts[0]}\n---\n")
    outputs = judge.generate(processed_prompts, sampling_params)
    assessments = [
        {
            "assessment": output.outputs[0].text,
            "match": ("True" in output.outputs[0].text) * 1,
        }
        for output in outputs
    ]
    return assessments


def evaluate_mt(
    sources: List[str],
    references: List[str],
    candidates: List[str],
    judge: Any,
    sampling_params: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    Evaluate machine translation outputs using a judge model (e.g., COMET).

    Parameters
    ----------
    sources : list of str
        Source sentences.
    references : list of str
        Reference translations.
    candidates : list of str
        Model-generated candidate translations.
    judge : Any
        MT evaluation model with a .predict() method.
    sampling_params : dict
        Additional parameters for judge.predict().

    Returns
    -------
    list of dict
        Assessments containing:
        - "assessment": model-generated evaluation score/text
    """
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(sources, candidates, references)
    ]
    print(f"\n---\nData example:\n\n{data[0]}\n---\n")
    outputs = judge.predict(data, gpus=1, **sampling_params)[0]
    assessments = [{"assessment": output} for output in outputs]
    return assessments


def main(
    tasks: str,
    model: str,
    judge_path: str,
    sampling_params: str,
    answers_path: str,
    output_path: str,
) -> None:
    """
    Run evaluation of generated answers using a judge model (instruction-based or MT).

    Parameters
    ----------
    tasks : str
        Semicolon-separated string of task names. Each task should be callable and return
        appropriate data (questions, ground_truths, sources if MT).
    model : str
        Name or path of the model that generated answers.
    judge_path : str
        Path to the judge model or checkpoint.
    sampling_params : str
        Sampling parameters as comma-separated 'key=value' string.
    answers_path : str
        Base directory where generated answers are stored.
    output_path : str
        Directory to save evaluation results and sample assessments.

    Returns
    -------
    None
        Assessment results are saved as JSON files under `output_path`.
    """
    print("""
===================================
========== LOADING MODEL ==========
===================================
""")

    judge_name = judge_path.split("/")[-1]
    sampling_params = str_to_dict(sampling_params)

    if judge_name == "wmt22-comet-da":
        judge = load_from_checkpoint(
            f"{judge_path}/checkpoints/model.ckpt", local_files_only=True
        )
        is_mt = True
    else:
        judge, sampling_params = load_vllm_model(judge_path, sampling_params)
        is_mt = False

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
        load_path = f"{answers_path}/{task}/{model}"
        save_path = f"{output_path}/{task}/{model}/{judge_name}"

        if not os.path.exists(f"{load_path}/answers.json"):
            print(f"No generated answers found at {load_path}/answers.json")
            continue

        if os.path.exists(f"{save_path}/results.json"):
            print(f"Assessment file already stored at {save_path}/results.json")
            continue

        print(f"\n*** Evaluating answers for task {task} ***\n")

        with open(f"{load_path}/answers.json", "r") as f:
            answers = json.load(f)

        try:
            if is_mt:
                sources, questions, ground_truths = eval(task)()
            else:
                questions, ground_truths = eval(task)()

        except Exception as e:
            print(f"Error: {e}")
            continue

        try:
            if is_mt:
                assessments = evaluate_mt(
                    sources, ground_truths, answers, judge, sampling_params
                )
            else:
                assessments = evaluate(
                    task,
                    questions,
                    ground_truths,
                    answers,
                    judge,
                    judge_path,
                    sampling_params,
                )
        except Exception as e:
            print(f"Error: {e}")
            continue

        scores = [
            assessment["assessment"] if is_mt else assessment["match"]
            for assessment in assessments
        ]
        random.seed(0)
        sampled_ids = random.sample(range(len(questions)), min(100, len(questions)))
        samples = [
            {
                "id": i,
                "question": questions[i],
                "generated_answer": answers[i],
                "ground_truth": ground_truths[i],
                **assessments[i],
            }
            for i in sampled_ids
        ]
        os.makedirs(save_path, exist_ok=True)

        with open(f"{save_path}/samples.json", "w") as f:
            json.dump(samples, f, ensure_ascii=False, indent=4)

        with open(f"{save_path}/results.json", "w") as f:
            json.dump(
                {"score": sum(scores) / len(scores)}, f, ensure_ascii=False, indent=4
            )

        print(f"Assessment files saved at {save_path}")

    print("""
=====================================
========== EVALUATION DONE ==========
=====================================
""")


if __name__ == "__main__":
    Fire(main)
