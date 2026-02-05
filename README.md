# EuroLLMEval

This repository contains the codebase for evaluating **EuroLLM-22B** and **EuroMOE-2.6B-0.6B**, in both base and instruction-tuned variants, as well as the corresponding baseline models used for comparison.

## Prerequisite

This repository is built on top of the [vLLM](https://github.com/vllm-project/vllm) inference engine. To run the code successfully, you must install `vllm` (recommended version: 0.10.2):

```bash
pip install vllm==0.10.2
```

## Tasks

The supported tasks are defined in [`eurollm_eval/dataloader.py`](eurollm_eval/dataloader.py). We distinguish between two types of tasks: **log-likelihood tasks**, which are used to evaluate base models (with the `_loglik` suffix), and **generative tasks**, which are used to evaluate instruction-tuned models. All tasks are implemented as Python functions.

### Log-likelihood tasks

Log-likelihood tasks are few-shot multiple-choice tasks. They return a list of few-shot prompts, a list of candidate answer choices, and the corresponding ground-truth answers as indices.

Example of a log-likelihood task:

```python
def hellaswag_loglik(num_shots=3):
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    questions = [f"{question} " for question in dataset["ctx"]]
    options = dataset["endings"]
    answers = [int(label) for label in dataset["label"]]
    fs_prefix = "\n\n".join(
        [questions[i] + options[i][answers[i]] for i in range(num_shots)]
    ) + "\n\n"
    questions, options, answers = (
        questions[num_shots:],
        options[num_shots:],
        answers[num_shots:],
    )
    questions = [fs_prefix + question for question in questions]
    return questions, options, answers
```

### Generative tasks

Generative tasks return a list of prompts and their corresponding ground-truth answers. Translation tasks additionally return the raw source sentences alongside the generation prompts. Some generative tasks use the `_constrained` suffix, indicating that the prompt explicitly instructs the model to produce the answer in a specific format.

Example of a generative task:

```python
def mmlu():
    dataset = load_dataset("cais/mmlu", "all", split="test")
    prompts, answers = [], []
    for example in dataset:
        choices_string = "\n" + "\n".join(
            [
                f"{letter}) {choice}"
                for letter, choice in zip(
                    ["A", "B", "C", "D"], example["choices"]
                )
            ]
        )
        prompts.append(example["question"] + choices_string)
        answers.append(["A", "B", "C", "D"][example["answer"]])
    return prompts, answers
```

## Evaluation

### Base models

Base model evaluation is performed using [`eurollm_eval/evaluate_base.py`](eurollm_eval/evaluate_base.py). Base models are evaluated on log-likelihood (multiple-choice) tasks. For each example, the question is concatenated with each candidate answer, and the log-likelihood of the resulting sequence is computed. The predicted answer is the candidate with the highest log-likelihood, which is then compared against the ground-truth label.

For example, to evaluate `Qwen/Qwen3-14B-Base` on `hellaswag_loglik` and `mmlu_loglik`, run:

```bash
python -m eurollm_eval.evaluate_base \
  --tasks "hellaswag_loglik,mmlu_loglik" \
  --model_path "Qwen/Qwen3-14B-Base" \
  --output_path "./results/base"
```

### Instruction-tuned models

Instruction-tuned models are evaluated in a generative setting, where models are prompted to produce answers directly. The evaluation follows a two-step process: first, answers are generated using [`eurollm_eval/generate.py`](eurollm_eval/generate.py); second, the generated answers are assessed using LLM-based judges via [`eurollm_eval/evaluate_instruct.py`](eurollm_eval/evaluate_instruct.py).

For example, to generate answers with `Qwen/Qwen3-14B` on the `arc_challenge` and `bbh` tasks, run:

```bash
python -m eurollm_eval.generate \
  --tasks "arc_challenge,bbh" \
  --model_path "Qwen/Qwen3-14B" \
  --sampling_params "temperature=0.7,top_p=0.8,top_k=20,min_p=0,presence_penalty=1.5" \
  --output_path "./answers/instruct"
```

The generated outputs are then evaluated using an LLM-as-a-judge setup, in which a judge model is provided with the original questions, the model-generated answers, and the corresponding ground truths.

For example, to assess the generated answers using `nvidia/Llama-3_3-Nemotron-Super-49B-v1_5` as the LLM judge, run:

```bash
python -m eurollm_eval.evaluate_instruct \
  --tasks "arc_challenge,bbh" \
  --model "Qwen3-14B" \
  --judge_path "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5" \
  --sampling_params "temperature=0" \
  --answers_path "./answers/instruct" \
  --output_path "./results/instruct"
```

## Analysis

The [`analysis/`](analysis/) directory contains analyses based on the evaluation results stored in [`results/`](results/).

- [`analysis/main_results.ipynb`](analysis/main_results.ipynb) includes the main results and model comparisons across the relevant tasks.
- [`analysis/assessment_analysis.ipynb`](analysis/assessment_analysis.ipynb) presents an analysis comparing regex-based answer extraction and LLM-as-a-judge evaluation, with reference to human judgments.
