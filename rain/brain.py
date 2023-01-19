import torch
from config import AUTO_MODEL_FOR_SEQ_2_SEQ_LM_MODEL, AUTO_TOKENIZER_MODEL

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(AUTO_TOKENIZER_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(AUTO_MODEL_FOR_SEQ_2_SEQ_LM_MODEL)

device = torch.device("cuda")


def preprocess(text):
    return text.replace("\n", "_")


def postprocess(text):
    return text.replace("_", "\n")


def answer(text, sample=False, top_p=0.6):
    """

    sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样、
    """
    text = preprocess(text)
    encoding = tokenizer(
        text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt"
    )
    if not sample:  # 不进行采样
        out = model.generate(
            **encoding,
            return_dict_in_generate=True,
            output_scores=False,
            max_length=128,
            num_beams=4,
            length_penalty=0.6,
        )
    else:  # 采样（生成）
        out = model.generate(
            **encoding,
            return_dict_in_generate=True,
            output_scores=False,
            max_length=128,
            do_sample=True,
            top_p=top_p,
        )
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])


def auto_prompt(query: str, origin_text: str) -> str:
    if "翻译" in query:
        if "英文" in query:
            prompt = f"翻译成英文：\n{origin_text}\n答案："
        else:
            prompt = f"翻译成中文：\n{origin_text}\n答案："
    elif "文本纠错" in query:
        prompt = f"文本纠错：\n{origin_text}\n答案："
    else:
        prompt = origin_text

    print("prompt: " + prompt)
    return prompt
