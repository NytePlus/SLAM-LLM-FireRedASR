import torch
import math
import argparse
import logging
import sys
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from whisper_normalizer.english import EnglishTextNormalizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

english_normalizer = EnglishTextNormalizer()

MAPPING_FILE = "/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech/test_oracle_v1/id2id"
id_map = {}
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        a, b = line.split()
        id_map[a] = b

class Paragraph():
    def __init__(self, video):
        utt = id_map[video["youtube_channel"]]
        self.idx = {}
        self.segments = []
        for s in video["segments"]:
            self.segments.append(s["txt_raw"])
            idx = int(s["uttid"].split('-')[-1])

            uttid = f'{utt}-{idx :04d}'
            self.idx[uttid] = len(self.segments) - 1

    def part(self, uttid, range = 40):
        end = self.idx[uttid]
        begin = max(0, end - range)
        return " ".join(self.segments[begin : end]), self.segments[end]

paragraph_dict = {}

class PPL():
    def __init__(self, model_name = "/models/Qwen2.5-7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.device="npu:0"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()

    def compute(self, sentence):
        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            add_special_tokens=False
        )

        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)
            
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        ppl = math.exp(loss.item())
        return ppl

    def compute_rep_ppl(self, context, rep):
        sentence = context + " " + rep

        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)

        input_ids = inputs["input_ids"].clone()
        labels = input_ids.clone()

        context_ids = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        context_len = len(context_ids)

        labels[:, :context_len] = -100
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

        ppl = math.exp(loss.item())
        return ppl
    
    def compute_cross_ppl(self, context, ref, hyp):
        sentence = context + " " + ref

        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            add_special_tokens=False
        )
        
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)

        labels = self.tokenizer(
            context + " " + hyp,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"]

        context_ids = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        context_len = len(context_ids)

        labels[:, :context_len] = -100
        with torch.no_grad():
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

        ppl = math.exp(loss.item())
        return ppl


def main(args):
    if args.context is not None:
        with open(args.context, "r") as f:
            data = json.load(f)
            for v in data["videos"]:
                utt = id_map[v["youtube_channel"]]
                paragraph_dict[utt] = Paragraph(v)

    refs = {}
    with open(args.multitask, "r") as f:
        for line in f:
            data = json.loads(line)
            uttid, ref = data['key'], data['target'].upper()
            biasing_words = set(item.strip() for item in data['hotword'].strip().split(","))
            if args.norm:
                ref = english_normalizer(ref).upper()
            refs[uttid] = {"text": ref, "biasing_words": biasing_words}

    logger.info("Loaded %d reference utts from %s", len(refs), args.multitask)

    hyps = {}
    with open(args.pred, "r") as f:
        for line in f:
            ary = line.strip().split(" ", maxsplit=1)
            # May have empty hypo
            if len(ary) >= 2:
                uttid, hyp = ary[0], ary[1].upper()
            else:
                uttid, hyp = ary[0], ""
            
            if args.norm:
                hyp = english_normalizer(hyp).upper()
            hyps[uttid] = hyp
    logger.info("Loaded %d hypothesis utts from %s", len(hyps), args.pred)

    if not args.lenient:
        for uttid in refs:
            if uttid in hyps:
                continue
            raise ValueError(
                f"{uttid} missing in pred! Set `--lenient` flag to ignore this error."
            )

    ppl = PPL()

    ignore_uttid = []
    for uttid in tqdm(refs, file=sys.stderr):
        if uttid not in hyps:
            ignore_uttid.append(uttid)
            continue
        ref_text = refs[uttid]["text"].lower()
        hyp_text = hyps[uttid].lower()

        if len(ref_text) == 0 or len(hyp_text) == 0:
            ignore_uttid.append(uttid)
            continue
        
        utt = '-'.join(uttid.split('-')[:-1])
        if args.context is not None:
            p = paragraph_dict[utt]
            context, gt = p.part(uttid)

        ref_ppl = ppl.compute_rep_ppl(context, ref_text)
        hyp_ppl = ppl.compute_rep_ppl(context, hyp_text)
        # print('context: ', context)
        # print('gt: ', gt)
        # print('ref: ', ref_text)

        if math.isnan(ref_ppl) or math.isnan(hyp_ppl):
            ignore_uttid.append(uttid)
            continue
        print(uttid, ref_ppl, hyp_ppl, hyp_ppl / ref_ppl)

    # print(f'ignored uttids: {ignore_uttid}')

if __name__ == '__main__':
    desc = "Compute ppl with qwen2-7b. Results are output to stdout."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--pred",
        required=True,
        help="Path to space-separated _pred file. First column is utterance ID. "
        "Second column is reference text.",
    )
    parser.add_argument(
        "--multitask",
        required=True,
        help="Path to multitask files. It is a jsonl file, each line a train/test data."
        "Has key, task, target, path, hotword. Use key, target and hotword.",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="If set, hyps doesn't have to cover all of refs.",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="If set, use whisper_normalizer to normalize refs and hyps",
    )
    parser.add_argument(
        "--context",
    )
    args = parser.parse_args()

    main(args)