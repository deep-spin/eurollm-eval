import os
import random
from typing import Any, List, Union

from datasets import Dataset, DatasetDict, concatenate_datasets
from datasets import get_dataset_config_names as _get_dataset_config_names
from datasets import load_dataset as _load_dataset

EUROLLM_LANGS = [
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "ga",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
    "ar",
    "ca",
    "zh",
    "gl",
    "hi",
    "ja",
    "ko",
    "no",
    "ru",
    "tr",
    "uk",
]

EUROLLM_LPS = [
    f"{src_lang}{tgt_lang}"
    for src_lang in EUROLLM_LANGS
    for tgt_lang in EUROLLM_LANGS
    if src_lang != tgt_lang
]

ISO2_TO_NATURAL = {
    "bg": "Bulgarian",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hu": "Hungarian",
    "ga": "Irish",
    "it": "Italian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mt": "Maltese",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sv": "Swedish",
    "ar": "Arabic",
    "ca": "Catalan",
    "zh": "Chinese",
    "gl": "Galician",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "no": "Norwegian",
    "ru": "Russian",
    "tr": "Turkish",
    "uk": "Ukrainian",
}

ISO3_TO_ISO2 = {
    "bul": "bg",
    "hrv": "hr",
    "ces": "cs",
    "dan": "da",
    "nld": "nl",
    "eng": "en",
    "est": "et",
    "fin": "fi",
    "fra": "fr",
    "deu": "de",
    "ell": "el",
    "hun": "hu",
    "gle": "ga",
    "ita": "it",
    "lav": "lv",
    "lit": "lt",
    "mlt": "mt",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "slk": "sk",
    "slv": "sl",
    "spa": "es",
    "swe": "sv",
    "ara": "ar",
    "cat": "ca",
    "zho": "zh",
    "glg": "gl",
    "hin": "hi",
    "jpn": "ja",
    "kor": "ko",
    "nor": "no",
    "rus": "ru",
    "tur": "tr",
    "ukr": "uk",
}

MT_TASKS = [
    "flores",
    "wmt24pp",
    "wmt25",
    "flores_eurollm",
    "wmt24pp_eurollm",
    "wmt25_eurollm",
]


def load_dataset(
    hf_path: str, *args: Any, **kwargs: Any
) -> Union[Dataset, DatasetDict]:
    """
    Load a HuggingFace dataset, optionally from a local datasets directory.

    Parameters
    ----------
    hf_path : str
        HuggingFace dataset identifier or path.
    *args : Any
        Positional arguments forwarded to the underlying dataset loader.
    **kwargs : Any
        Keyword arguments forwarded to the underlying dataset loader.

    Returns
    -------
    Dataset or DatasetDict
        The loaded dataset object.
    """
    if "LOCAL_DATASETS_DIR" in os.environ:
        path = os.path.join(os.environ["LOCAL_DATASETS_DIR"], hf_path.split("/")[-1])
    else:
        path = hf_path
    dataset = _load_dataset(path, *args, **kwargs)
    return dataset


def get_dataset_config_names(hf_path: str, *args: Any, **kwargs: Any) -> List[str]:
    """
    Return available configuration names for a HuggingFace dataset, optionally
    resolved from a local datasets directory.

    Parameters
    ----------
    hf_path : str
        HuggingFace dataset identifier or path.
    *args : Any
        Positional arguments forwarded to the underlying config lookup function.
    **kwargs : Any
        Keyword arguments forwarded to the underlying config lookup function.

    Returns
    -------
    list of str
        Available configuration names for the dataset.
    """
    if "LOCAL_DATASETS_DIR" in os.environ:
        path = os.path.join(os.environ["LOCAL_DATASETS_DIR"], hf_path.split("/")[-1])
    else:
        path = hf_path
    config_names = _get_dataset_config_names(path, *args, **kwargs)
    return config_names


##################
### Generative ###
##################


def hellaswag():
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    prompt_template = "Given the following context, provide the most likely continuation.\n{context}\n{options}"
    prompts, answers = [], []
    for example in dataset:
        choices_string = "\n".join(
            [
                f"{letter}) {ending}"
                for letter, ending in zip(["A", "B", "C", "D"], example["endings"])
            ]
        )
        prompt = prompt_template.format(context=example["ctx"], options=choices_string)
        prompts.append(prompt)
        answers.append(["A", "B", "C", "D"][int(example["label"])])
    return prompts, answers


def mmlu():
    dataset = load_dataset("cais/mmlu", "all", split="test")
    prompts, answers = [], []
    for example in dataset:
        choices_string = "\n" + "\n".join(
            [
                f"{letter}) {choice}"
                for letter, choice in zip(["A", "B", "C", "D"], example["choices"])
            ]
        )
        prompts.append(example["question"] + choices_string)
        answers.append(["A", "B", "C", "D"][example["answer"]])
    return prompts, answers


def mmlu_pro():
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    prompts = []
    for example in dataset:
        choices_string = "\n" + "\n".join(
            [
                f"{letter}) {choice}"
                for letter, choice in zip(
                    ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                    example["options"],
                )
            ]
        )
        prompts.append(example["question"] + choices_string)
    answers = dataset["answer"]
    return prompts, answers


def bbh():
    subset_names = [
        "boolean_expressions",
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "dyck_languages",
        "formal_fallacies",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ]
    dataset = []
    for subset_name in subset_names:
        dataset.append(load_dataset("SaylorTwift/bbh", subset_name, split="test"))
    dataset = concatenate_datasets(dataset)
    prompts = dataset["input"]
    answers = dataset["target"]
    return prompts, answers


def arc_challenge():
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    prompts = []
    for example in dataset:
        choices_string = "\n" + "\n".join(
            [
                f"{label}) {text}"
                for label, text in zip(
                    example["choices"]["label"], example["choices"]["text"]
                )
            ]
        )
        prompts.append(example["question"] + choices_string)
    answers = dataset["answerKey"]
    return prompts, answers


def gpqa_diamond():
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    random.seed(0)
    prompts, answers = [], []
    for example in dataset:
        choices = [
            example["Correct Answer"],
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        random.shuffle(choices)
        answer_idx = choices.index(example["Correct Answer"])
        choices_string = "\n" + "\n".join(
            [
                f"{letter}) {choice}"
                for letter, choice in zip(["A", "B", "C", "D"], choices)
            ]
        )
        prompts.append(example["Question"] + choices_string)
        answers.append(["A", "B", "C", "D"][answer_idx])
    return prompts, answers


def gsm8k():
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    prompts = dataset["question"]
    answers = [answer.split("#### ")[-1] for answer in dataset["answer"]]
    return prompts, answers


def math_500():
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    prompts = dataset["problem"]
    answers = dataset["answer"]
    return prompts, answers


def humaneval():
    dataset = load_dataset("openai/openai_humaneval", split="test")
    prompt_template = "Complete the following Python function accordingly.\n{prompt}"
    prompts = [prompt_template.format(prompt=prompt) for prompt in dataset["prompt"]]
    answers = dataset["canonical_solution"]
    return prompts, answers


def ifeval():
    dataset = load_dataset("google/IFEval", split="train")
    prompts = dataset["prompt"]
    answers = [""] * len(dataset)
    return prompts, answers


def m_hellaswag(lang):
    def fn():
        dataset = load_dataset("alexandrainst/m_hellaswag", lang, split="val")
        prompt_template = "Given the following context, provide the most likely continuation.\n{context}\n{options}\nAnswer in English."
        prompts, answers = [], []
        for example in dataset:
            choices_string = "\n".join(
                [
                    f"{letter}) {ending}"
                    for letter, ending in zip(["A", "B", "C", "D"], example["endings"])
                ]
            )
            prompt = prompt_template.format(
                context=example["ctx"], options=choices_string
            )
            prompts.append(prompt)
            answers.append(["A", "B", "C", "D"][int(example["label"])])
        return prompts, answers

    return fn


m_hellaswag_langs = get_dataset_config_names("alexandrainst/m_hellaswag")
for lang in m_hellaswag_langs:
    if lang in EUROLLM_LANGS and lang not in ["en", "zh"]:
        globals()[f"m_hellaswag_{lang}"] = m_hellaswag(lang)


def m_mmlu(lang):
    def fn():
        dataset = load_dataset("alexandrainst/m_mmlu", lang, split="test")
        prompts = []
        for example in dataset:
            choices = [
                example["option_a"],
                example["option_b"],
                example["option_c"],
                example["option_d"],
            ]
            choices_string = "\n".join(
                [
                    f"{letter}) {choice}"
                    for letter, choice in zip(["A", "B", "C", "D"], choices)
                ]
            )
            prompts.append(
                example["instruction"] + "\n" + choices_string + "\nAnswer in English."
            )
        answers = dataset["answer"]
        return prompts, answers

    return fn


m_mmlu_langs = get_dataset_config_names("alexandrainst/m_mmlu")
for lang in m_mmlu_langs:
    if lang in EUROLLM_LANGS and lang != "en":
        globals()[f"m_mmlu_{lang}"] = m_mmlu(lang)


def mmlu_prox(lang):
    def fn():
        dataset = load_dataset("li-lab/MMLU-ProX", lang, split="test")
        prompts = []
        for example in dataset:
            choices = [
                example["option_0"],
                example["option_1"],
                example["option_2"],
                example["option_3"],
                example["option_4"],
                example["option_5"],
                example["option_6"],
                example["option_7"],
                example["option_8"],
                example["option_9"],
            ]
            choices_string = "\n" + "\n".join(
                [
                    f"{letter}) {choice}"
                    for letter, choice in zip(
                        ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"], choices
                    )
                ]
            )
            prompts.append(
                example["question"] + choices_string + "\nAnswer in English."
            )
        answers = dataset["answer"]
        return prompts, answers

    return fn


mmlu_prox_langs = get_dataset_config_names("li-lab/MMLU-ProX")
for lang in mmlu_prox_langs:
    if lang in EUROLLM_LANGS and lang != "en":
        globals()[f"mmlu_prox_{lang}"] = mmlu_prox(lang)


def m_arc_challenge(lang):
    def fn():
        dataset = load_dataset("alexandrainst/m_arc", lang, split="test")
        prompts = []
        for example in dataset:
            choices = [
                example["option_a"],
                example["option_b"],
                example["option_c"],
                example["option_d"],
            ]
            choices_string = "\n" + "\n".join(
                [
                    f"{letter}) {choice}"
                    for letter, choice in zip(["A", "B", "C", "D"], choices)
                ]
            )
            prompts.append(
                example["instruction"] + choices_string + "\nAnswer in English."
            )
        answers = dataset["answer"]
        return prompts, answers

    return fn


m_arc_challenge_langs = get_dataset_config_names("alexandrainst/m_arc")
for lang in m_arc_challenge_langs:
    if lang in EUROLLM_LANGS and lang != "en":
        globals()[f"m_arc_challenge_{lang}"] = m_arc_challenge(lang)


def mgsm(lang):
    def fn():
        dataset = load_dataset(
            "juletxara/mgsm", lang, split="test", trust_remote_code=True
        )
        prompts = [
            question + "\nAnswer in English." for question in dataset["question"]
        ]
        answers = dataset["answer_number"]
        return prompts, answers

    return fn


mgsm_langs = get_dataset_config_names("juletxara/mgsm", trust_remote_code=True)
for lang in mgsm_langs:
    if lang in EUROLLM_LANGS and lang != "en":
        globals()[f"mgsm_{lang}"] = mgsm(lang)


def flores(lp):
    def fn():
        src_lang, tgt_lang = lp.split("-")
        dataset_src = load_dataset("hgissbkh/flores", src_lang, split="test")
        dataset_tgt = load_dataset("hgissbkh/flores", tgt_lang, split="test")
        src_lang = (
            "English"
            if src_lang == "en_Latn"
            else ISO2_TO_NATURAL[ISO3_TO_ISO2[src_lang.split("_")[0]]]
        )
        tgt_lang = (
            "English"
            if tgt_lang == "en_Latn"
            else ISO2_TO_NATURAL[ISO3_TO_ISO2[tgt_lang.split("_")[0]]]
        )
        sources = dataset_src["text"]
        prompts = [
            "Translate the following {src_lang} source text to {tgt_lang}. Output only the translation.\nSource: {src}\nTranslation: ".format(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src=src.strip(),
            )
            for src in sources
        ]
        answers = [tgt.strip() for tgt in dataset_tgt["text"]]
        return sources, prompts, answers

    return fn


def flores_eurollm(lp):
    def fn():
        src_lang, tgt_lang = lp.split("-")
        dataset_src = load_dataset("hgissbkh/flores", src_lang, split="test")
        dataset_tgt = load_dataset("hgissbkh/flores", tgt_lang, split="test")
        src_lang = (
            "English"
            if src_lang == "en_Latn"
            else ISO2_TO_NATURAL[ISO3_TO_ISO2[src_lang.split("_")[0]]]
        )
        tgt_lang = (
            "English"
            if tgt_lang == "en_Latn"
            else ISO2_TO_NATURAL[ISO3_TO_ISO2[tgt_lang.split("_")[0]]]
        )
        sources = dataset_src["text"]
        prompts = [
            "Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {src}\n{tgt_lang}: ".format(
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src=src.strip(),
            )
            for src in sources
        ]
        answers = [tgt.strip() for tgt in dataset_tgt["text"]]
        return sources, prompts, answers

    return fn


flores_langs = get_dataset_config_names("hgissbkh/flores")
for lang in flores_langs:
    xxx = lang.split("_")[0]
    if lang not in ["en_Latn", "arb_Latn", "zho_Hant"] and xxx in ISO3_TO_ISO2:
        xx = ISO3_TO_ISO2[xxx]
        if f"en{xx}" in EUROLLM_LPS:
            globals()[f"flores_en{xx}"] = flores(f"en_Latn-{lang}")
            globals()[f"flores_{xx}en"] = flores(f"{lang}-en_Latn")            
            globals()[f"flores_eurollm_en{xx}"] = flores_eurollm(f"en_Latn-{lang}")
            globals()[f"flores_eurollm_{xx}en"] = flores_eurollm(f"{lang}-en_Latn")


def wmt24pp(lp, from_en=True):
    def fn():
        dataset = load_dataset("google/wmt24pp", lp, split="train")
        dataset = dataset.filter(lambda x: not x["is_bad_source"])
        xx = lp.split("-")[1][:2]
        if from_en:
            src_lang, tgt_lang = "en", xx
            sources, targets = dataset["source"], dataset["target"]
        else:
            src_lang, tgt_lang = xx, "en"
            sources, targets = dataset["target"], dataset["source"]
        prompts = [
            "Translate the following {src_lang} source text to {tgt_lang}. Output only the translation.\nSource: {src}\nTranslation: ".format(
                src_lang=ISO2_TO_NATURAL[src_lang],
                tgt_lang=ISO2_TO_NATURAL[tgt_lang],
                src=src.strip(),
            )
            for src in sources
        ]
        answers = [tgt.strip() for tgt in targets]
        return sources, prompts, answers

    return fn


def wmt24pp_eurollm(lp, from_en=True):
    def fn():
        dataset = load_dataset("google/wmt24pp", lp, split="train")
        dataset = dataset.filter(lambda x: not x["is_bad_source"])
        xx = lp.split("-")[1][:2]
        if from_en:
            src_lang, tgt_lang = "en", xx
            sources, targets = dataset["source"], dataset["target"]
        else:
            src_lang, tgt_lang = xx, "en"
            sources, targets = dataset["target"], dataset["source"]
        prompts = [
            "Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {src}\n{tgt_lang}: ".format(
                src_lang=ISO2_TO_NATURAL[src_lang],
                tgt_lang=ISO2_TO_NATURAL[tgt_lang],
                src=src.strip(),
            )
            for src in sources
        ]
        answers = [tgt.strip() for tgt in targets]
        return sources, prompts, answers

    return fn


wmt24pp_lps = get_dataset_config_names("google/wmt24pp")
for lp in wmt24pp_lps:
    xx_XX = lp.split("-")[1]
    xx = xx_XX[:2]
    if f"en{xx}" in EUROLLM_LPS and xx_XX not in [
        "ar_EG",
        "zh_TW",
        "fr_CA",
        "sw_KE",
        "pt_PT",
    ]:
        globals()[f"wmt24pp_en{xx}"] = wmt24pp(lp, from_en=True)
        globals()[f"wmt24pp_{xx}en"] = wmt24pp(lp, from_en=False)        
        globals()[f"wmt24pp_eurollm_en{xx}"] = wmt24pp_eurollm(lp, from_en=True)
        globals()[f"wmt24pp_eurollm_{xx}en"] = wmt24pp_eurollm(lp, from_en=False)


def wmt25(lp):
    def fn():
        dataset = load_dataset("hgissbkh/wmt25", lp, split="test")
        src_lang, tgt_lang = lp.split("-")
        sources = dataset["src"]
        prompts = [
            "Translate the following {src_lang} source text to {tgt_lang}. Output only the translation.\nSource: {src}\nTranslation: ".format(
                src_lang=ISO2_TO_NATURAL[src_lang],
                tgt_lang=ISO2_TO_NATURAL[tgt_lang],
                src=src.strip(),
            )
            for src in sources
        ]
        answers = [tgt.strip() for tgt in dataset["tgt"]]
        return sources, prompts, answers

    return fn


def wmt25_eurollm(lp):
    def fn():
        dataset = load_dataset("hgissbkh/wmt25", lp, split="test")
        src_lang, tgt_lang = lp.split("-")
        sources = dataset["src"]
        prompts = [
            "Translate the following {src_lang} source text to {tgt_lang}:\n{src_lang}: {src}\n{tgt_lang}: ".format(
                src_lang=ISO2_TO_NATURAL[src_lang],
                tgt_lang=ISO2_TO_NATURAL[tgt_lang],
                src=src.strip(),
            )
            for src in sources
        ]
        answers = [tgt.strip() for tgt in dataset["tgt"]]
        return sources, prompts, answers

    return fn


wmt25_lps = get_dataset_config_names("hgissbkh/wmt25")
for lp in wmt25_lps:
    src_lang, tgt_lang = lp.split("-")
    if f"{src_lang}{tgt_lang}" in EUROLLM_LPS:
        globals()[f"wmt25_{src_lang}{tgt_lang}"] = wmt25(lp)
        globals()[f"wmt25_eurollm_{src_lang}{tgt_lang}"] = wmt25_eurollm(lp)


######################
### Log-likelihood ###
######################


def hellaswag_loglik(num_shots=3):
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    questions = [f"{question} " for question in dataset["ctx"]]
    options = dataset["endings"]
    answers = [int(label) for label in dataset["label"]]
    fs_prefix = (
        "\n\n".join([questions[i] + options[i][answers[i]] for i in range(num_shots)])
        + "\n\n"
    )
    questions, options, answers = (
        questions[num_shots:],
        options[num_shots:],
        answers[num_shots:],
    )
    questions = [fs_prefix + question for question in questions]
    return questions, options, answers


def mmlu_loglik(num_shots=3):
    dataset = load_dataset("cais/mmlu", "all", split="test")
    questions = [f"Question: {question}\nAnswer: " for question in dataset["question"]]
    options = dataset["choices"]
    answers = dataset["answer"]
    fs_prefix = (
        "\n\n".join([questions[i] + options[i][answers[i]] for i in range(num_shots)])
        + "\n\n"
    )
    questions, options, answers = (
        questions[num_shots:],
        options[num_shots:],
        answers[num_shots:],
    )
    questions = [fs_prefix + question for question in questions]
    return questions, options, answers


def arc_challenge_loglik(num_shots=3):
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    questions = [f"Question: {question}\nAnswer: " for question in dataset["question"]]
    options = [example["choices"]["text"] for example in dataset]
    answers = [
        example["choices"]["label"].index(example["answerKey"]) for example in dataset
    ]
    fs_prefix = (
        "\n\n".join([questions[i] + options[i][answers[i]] for i in range(num_shots)])
        + "\n\n"
    )
    questions, options, answers = (
        questions[num_shots:],
        options[num_shots:],
        answers[num_shots:],
    )
    questions = [fs_prefix + question for question in questions]
    return questions, options, answers


def m_hellaswag_loglik(lang):
    def fn(num_shots=3):
        dataset = load_dataset("alexandrainst/m_hellaswag", lang, split="val")
        questions = [f"{question} " for question in dataset["ctx"]]
        options = dataset["endings"]
        answers = [int(label) for label in dataset["label"]]
        fs_prefix = (
            "\n\n".join(
                [questions[i] + options[i][answers[i]] for i in range(num_shots)]
            )
            + "\n\n"
        )
        questions, options, answers = (
            questions[num_shots:],
            options[num_shots:],
            answers[num_shots:],
        )
        questions = [fs_prefix + question for question in questions]
        return questions, options, answers

    return fn


for lang in m_hellaswag_langs:
    if lang in EUROLLM_LANGS and lang not in ["en", "zh"]:
        globals()[f"m_hellaswag_loglik_{lang}"] = m_hellaswag_loglik(lang)


def m_mmlu_loglik(lang):
    def fn(num_shots=3):
        dataset = load_dataset("alexandrainst/m_mmlu", lang, split="test")
        questions = [
            f"Question: {question}\nAnswer: " for question in dataset["instruction"]
        ]
        options = [
            [example[f"option_{x}"] for x in ["a", "b", "c", "d"]]
            for example in dataset
        ]
        answers = [["A", "B", "C", "D"].index(answer) for answer in dataset["answer"]]
        fs_prefix = (
            "\n\n".join(
                [questions[i] + options[i][answers[i]] for i in range(num_shots)]
            )
            + "\n\n"
        )
        questions, options, answers = (
            questions[num_shots:],
            options[num_shots:],
            answers[num_shots:],
        )
        questions = [fs_prefix + question for question in questions]
        return questions, options, answers

    return fn


for lang in m_mmlu_langs:
    if lang in EUROLLM_LANGS and lang != "en":
        globals()[f"m_mmlu_loglik_{lang}"] = m_mmlu_loglik(lang)


def m_arc_challenge_loglik(lang):
    def fn(num_shots=3):
        dataset = load_dataset("alexandrainst/m_arc", lang, split="test")
        questions = [
            f"Question: {question}\nAnswer: " for question in dataset["instruction"]
        ]
        options = [
            [
                example[f"option_{x}"]
                for x in ["a", "b", "c", "d"]
                if isinstance(example[f"option_{x}"], str)
            ]
            for example in dataset
        ]
        answers = [["A", "B", "C", "D"].index(answer) for answer in dataset["answer"]]
        fs_prefix = (
            "\n\n".join(
                [questions[i] + options[i][answers[i]] for i in range(num_shots)]
            )
            + "\n\n"
        )
        questions, options, answers = (
            questions[num_shots:],
            options[num_shots:],
            answers[num_shots:],
        )
        questions = [fs_prefix + question for question in questions]
        return questions, options, answers

    return fn


for lang in m_arc_challenge_langs:
    if lang in EUROLLM_LANGS and lang != "en":
        globals()[f"m_arc_challenge_loglik_{lang}"] = m_arc_challenge_loglik(lang)


###################
### Constrained ###
###################


def gpqa_diamond_constrained():
    prompts, answers = gpqa_diamond()
    prompts = [
        prompt
        + '\nAnswer with "the answer is X)", where X denotes the correct choice letter.'
        for prompt in prompts
    ]
    return prompts, answers


def gsm8k_constrained():
    prompts, answers = gsm8k()
    prompts = [
        prompt + '\nAnswer with "the answer is X", where X is the correct answer.'
        for prompt in prompts
    ]
    return prompts, answers


def mmlu_constrained():
    prompts, answers = mmlu()
    prompts = [
        prompt
        + '\nAnswer with "the answer is X)", where X denotes the correct choice letter.'
        for prompt in prompts
    ]
    return prompts, answers


def mmlu_pro_constrained():
    prompts, answers = mmlu_pro()
    prompts = [
        prompt
        + '\nAnswer with "the answer is X)", where X denotes the correct choice letter.'
        for prompt in prompts
    ]
    return prompts, answers


MT_TASKS_WITH_LPS = [
    glob for glob in globals() if any(glob.startswith(task) for task in MT_TASKS)
]
