from nltk import pos_tag, word_tokenize
from score import *

def load_reverse_mapping(path: str) -> dict:
    """
    读取文件，建立 {second_col: first_col} 的映射
    """
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            first, second = line.split()
            mapping[second] = first
    return mapping

def uttid2ytb(mapping, uttid):
    idx = int(uttid[-4:])
    uttid_prefix = uttid[:-5]
    ytb = mapping.get(uttid_prefix)
    if ytb is None:
        return uttid
    return f"YTB+{ytb}+{idx:05d}"

def is_proper_noun(word, context_sentence=None):
    """
    判断一个词是否是专有名词（NNP 或 NNPS）
    """
    if context_sentence is None:
        tokens = [word]
    else:
        tokens = word_tokenize(context_sentence)
    
    tags = pos_tag(tokens)
    for w, tag in tags:
        if w.lower() == word.lower() and tag in ('NNP', 'NNPS'):
            return True
    return False

def main(args):
    id_mapping = load_reverse_mapping('/aistor/sjtu/hpc_stor01/home/wangchencheng/data/slidespeech/test_oracle_v1/id2id')

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

    refs2 = {}
    with open(args.multitask2, "r") as f:
        for line in f:
            data = json.loads(line)
            uttid, ref = data['key'], data['target']
            refs2[uttid] = {"text": ref}

    logger.info("Loaded %d reference utts from %s", len(refs2), args.multitask2)

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

    # Calculate WER, U-WER, and B-WER
    wer = WordError()
    u_wer = WordError()
    b_wer = WordError()

    ignore_uttid = []
    for uttid in refs:
        if uttid not in hyps:
            ignore_uttid.append(uttid)
            continue
        ref_tokens = refs[uttid]["text"].split()

        id = uttid2ytb(id_mapping, uttid)
        ref2_tokens = refs2[id]["text"].split()
        nnps = [t.upper() for t in ref2_tokens if is_proper_noun(t, refs2[id]["text"])]

        biasing_words = refs[uttid]["biasing_words"]
        hyp_tokens = hyps[uttid].split()
        ed = EditDistance()
        try:
            result = ed.align(ref_tokens, hyp_tokens)
        except ValueError as e:
            if "Doesn't support empty ref AND hyp!" == str(e):
                ignore_uttid.append(uttid)
                continue
            else:
                raise e
        for code, ref_idx, hyp_idx in zip(result.codes, result.refs, result.hyps):
            if code in [Code.substitution, Code.deletion] and ref_tokens[ref_idx] in nnps:
                code = Code.match # 假设所有的专有名词都正确识别了
            # if code in [Code.substitution, Code.deletion] and ref_tokens[ref_idx].upper() in biasing_words:
            #     code = Code.match # 假设所有的kw都正确识别了

            if code == Code.match:
                wer.ref_words += 1
                if ref_tokens[ref_idx] in biasing_words:
                    b_wer.ref_words += 1
                else:
                    u_wer.ref_words += 1
            elif code == Code.substitution:
                wer.ref_words += 1
                wer.errors[Code.substitution] += 1
                if ref_tokens[ref_idx] in biasing_words:
                    b_wer.ref_words += 1
                    b_wer.errors[Code.substitution] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.substitution] += 1
            elif code == Code.deletion:
                wer.ref_words += 1
                wer.errors[Code.deletion] += 1
                if ref_tokens[ref_idx] in biasing_words:
                    b_wer.ref_words += 1
                    b_wer.errors[Code.deletion] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.deletion] += 1
            elif code == Code.insertion:
                wer.errors[Code.insertion] += 1
                if hyp_tokens[hyp_idx] in biasing_words:
                    b_wer.errors[Code.insertion] += 1
                else:
                    u_wer.errors[Code.insertion] += 1

    # Report results
    print(f"WER: {wer.get_result_string()}")
    print(f"U-WER: {u_wer.get_result_string()}")
    print(f"B-WER: {b_wer.get_result_string()}")

    print(f'ignored uttids: {ignore_uttid}')


if __name__ ==  "__main__":
    desc = "Compute WER, U-WER, and B-WER. Results are output to stdout."
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
        "--multitask2",
        required=True,
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
    args = parser.parse_args()
    main(args)