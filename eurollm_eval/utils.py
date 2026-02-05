import os
from typing import Any, Dict, List, Tuple, Union

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams


NUM_AVAILABLE_GPUS = int(os.environ.get("NUM_AVAILABLE_GPUS", torch.cuda.device_count()))
NUM_AVAILABLE_NODES = int(os.environ.get("NUM_AVAILABLE_NODES", 1))


def str_to_dict(_str: str) -> Dict[str, Union[int, float]]:
    """
    Convert a comma-separated key=value string into a dictionary.

    Parameters
    ----------
    _str : str
        String containing comma-separated key=value pairs.

    Returns
    -------
    dict
        Dictionary with keys as strings and values converted to int (for 'top_k' and 'batch_size') or float otherwise.
    """
    _dict = [elt.split("=") for elt in _str.split(";")]
    _dict = {
        k: (int(v) if k in ["top_k", "batch_size"] else float(v)) for k, v in _dict
    }
    return _dict


def get_configs_to_run(
    tasks: List[str], model_path: str, judge_paths: List[str], output_path: str
) -> Dict[str, List[str]]:
    """
    Determine which configurations still need to be run based on existing outputs.

    Parameters
    ----------
    tasks : list of str
        List of task names.
    model_path : str
        Path to the model.
    judge_paths : list of str
        Paths to judge configurations.
    output_path : str
        Base directory to save outputs.

    Returns
    -------
    dict
        Dictionary mapping each task to a list of judge paths that need to be run.
    """
    model_name = model_path.split("/")[-1]
    configs_to_run = {}

    for task in tasks:
        configs_to_run[task] = []

        for judge_path in judge_paths:
            judge_name = judge_path.split("/")[-1]
            save_path = os.path.join(output_path, task, model_name, judge_name)

            if not (os.path.exists(save_path) and len(os.listdir(save_path)) == 2):
                configs_to_run[task].append(judge_path)

    return configs_to_run


def load_vllm_model(
    path: str,
    sampling_params: Dict[str, Any],
    max_model_len: int = 32768,
    max_tokens: int = 2048,
    seed: int = 0,
) -> Tuple[LLM, SamplingParams]:
    """
    Load a vLLM model and prepare its sampling parameters.

    Parameters
    ----------
    path : str
        Path to the model.
    sampling_params : dict
        Sampling parameters to pass to SamplingParams.
    max_model_len : int, default=32768
        Maximum model sequence length.
    max_tokens : int, default=2048
        Maximum number of tokens to generate.
    seed : int, default=0
        Random seed for sampling.

    Returns
    -------
    model : LLM
        The loaded vLLM model.
    sampling_params : SamplingParams
        Sampling parameters object.
    """
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model_kwargs = {
        "model": path,
        "dtype": "bfloat16",
        "tensor_parallel_size": NUM_AVAILABLE_GPUS,
        "enforce_eager": NUM_AVAILABLE_NODES > 1,
        "max_model_len": min(
            max_model_len, getattr(config, "max_position_embeddings", float("inf"))
        ),
        "trust_remote_code": True,
    }
    sampling_kwargs = {
        "max_tokens": max_tokens,
        "seed": seed,
        **sampling_params,
    }

    if "Mistral-Small-3.2-24B-Instruct-2506" in path:
        for k in ["tokenizer_mode", "load_format", "config_format"]:
            model_kwargs[k] = "mistral"

    model = LLM(**model_kwargs)
    sampling_params = SamplingParams(**sampling_kwargs)
    return model, sampling_params


def load_vllm_model_for_loglik(
    path: str, max_model_len: int = 32768
) -> Tuple[LLM, SamplingParams]:
    """
    Load a vLLM model for log-likelihood evaluation (single-token sampling).

    Parameters
    ----------
    path : str
        Path to the model.
    max_model_len : int, default=32768
        Maximum model sequence length.

    Returns
    -------
    model : LLM
        The loaded vLLM model.
    sampling_params : SamplingParams
        Sampling parameters object configured for log-likelihood computation.
    """
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    model_kwargs = {
        "model": path,
        "dtype": "bfloat16",
        "tensor_parallel_size": torch.cuda.device_count(),
        "max_model_len": min(
            max_model_len, getattr(config, "max_position_embeddings", float("inf"))
        ),
        "trust_remote_code": True,
    }
    sampling_kwargs = {
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": 1,
        "prompt_logprobs": 1,
    }

    if "Mistral-Small-3.2-24B-Instruct-2506" in path:
        for k in ["tokenizer_mode", "load_format", "config_format"]:
            model_kwargs[k] = "mistral"

    model = LLM(**model_kwargs)
    sampling_params = SamplingParams(**sampling_kwargs)
    return model, sampling_params


def process_prompts_instruct(
    prompts: List[str], tokenizer_path: str, max_model_len: int
) -> List[str]:
    """
    Process prompts for instruction-tuned models using the appropriate tokenizer.

    Parameters
    ----------
    prompts : list of str
        List of input prompts.
    tokenizer_path : str
        Path or identifier of the tokenizer/model.
    max_model_len : int
        Maximum sequence length for the model.

    Returns
    -------
    list of str
        Processed prompts ready for model input.
    """
    if "Mistral-Small-3.2-24B-Instruct-2506" in tokenizer_path:
        tokenizer = MistralTokenizer.from_file(f"{tokenizer_path}/tekken.json")
        prompts = [
            tokenizer.decode(
                tokenizer.encode_chat_completion(
                    ChatCompletionRequest(
                        messages=[{"role": "user", "content": prompt}]
                    )
                ).tokens[: max_model_len - 3],
                special_token_policy=SpecialTokenPolicy.KEEP,
            )
            for prompt in tqdm(prompts, desc="Processing prompts")
        ]

    elif "Llama-3_3-Nemotron-Super-49B-v1_5" in tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        prompts = [
            tokenizer.decode(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "/no_think"},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )[: max_model_len - 3]
            )
            for prompt in tqdm(prompts, desc="Processing prompts")
        ]

    elif "gpt-oss" in tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        prompts = [
            tokenizer.decode(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "Reasoning: low"},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )[: max_model_len - 3]
            )
            for prompt in tqdm(prompts, desc="Processing prompts")
        ]

    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        prompts = [
            tokenizer.decode(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )[: max_model_len - 3]
            )
            for prompt in tqdm(prompts, desc="Processing prompts")
        ]

    return prompts


def process_prompts_base(
    prompts: List[str], tokenizer_path: str, max_model_len: int
) -> List[str]:
    """
    Process prompts for base models using a standard tokenizer.

    Parameters
    ----------
    prompts : list of str
        List of input prompts.
    tokenizer_path : str
        Path or identifier of the tokenizer/model.
    max_model_len : int
        Maximum sequence length for the model.

    Returns
    -------
    list of str
        Processed prompts ready for model input.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    prompts = [
        tokenizer.decode(
            tokenizer.encode(prompt)[-(max_model_len - 3) :],
            skip_special_tokens=True,
        )
        for prompt in tqdm(prompts, desc="Processing prompts")
    ]
    return prompts
