from score import WordError, EditDistance, Code

wer = WordError()
u_wer = WordError()
b_wer = WordError()

ref_tokens = ['The', 'cat', 'sat', 'down']
biasing_words = ['down']
hyp_tokens = ['The', 'big', 'cat', 'sat']
ed = EditDistance()
result = ed.align(ref_tokens, hyp_tokens)

# print(len(ref_tokens), len(hyp_tokens), len(result.codes))
for code, ref_idx, hyp_idx in zip(result.codes, result.refs, result.hyps):
    print(code, ref_idx, hyp_idx)
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
    print(u_wer.ref_words)

# Report results
print(f"WER: {wer.get_result_string()}")
print(f"U-WER: {u_wer.get_result_string()}")
print(f"B-WER: {b_wer.get_result_string()}")