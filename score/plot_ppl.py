import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def read_wer(path):
    wer_dict = {}
    current_utt = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("utt:"):
                current_utt = line.split("utt:")[1].strip()
            elif line.startswith("WER:") and current_utt is not None:
                wer = float(line.split()[1])

                if wer < 100:
                    wer_dict[current_utt] = wer
                    current_utt = None

    return wer_dict



def read_ppl(path):
    ppl_dict = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            utt = parts[0]
            ppl = float(parts[-1])
            if ppl <= 10:
                ppl_dict[utt] = ppl

    return ppl_dict



wer_dict = read_wer("exp/20251110-1416-slidespeech-hotenc_asr/aispeech_asr_epoch_32_total_step_400000/decode_slidespeech_asr_test_norm_cer")
ppl_dict = read_ppl("exp/20251110-1416-slidespeech-hotenc_asr/aispeech_asr_epoch_32_total_step_400000/decode_slidespeech_asr_test_norm_pplcmg10")

# 取交集 utt
common_utts = sorted(set(wer_dict) & set(ppl_dict))
print(f"Matched utts: {len(common_utts)}")

wers = np.array([wer_dict[u] for u in common_utts]).reshape(-1, 1)
ppls = np.array([ppl_dict[u] for u in common_utts])

# 线性回归
model = LinearRegression()
model.fit(wers, ppls)

x = np.linspace(wers.min(), wers.max(), 200).reshape(-1, 1)
y = model.predict(x)

# 画图
plt.figure(figsize=(6, 5))
plt.scatter(wers, ppls, alpha=0.7)
plt.plot(x, y)
plt.xlabel("WER (%)")
plt.ylabel("PPL")
plt.title("PPL vs WER (utt-aligned)")
plt.grid(True)
plt.show()
plt.savefig("ppl_vs_wer_contextmarkgt10.png", dpi=300, bbox_inches='tight')

print("Regression:")
print(f"  slope = {model.coef_[0]:.4f}")
print(f"  intercept = {model.intercept_:.4f}")
print(f"  R^2 = {model.score(wers, ppls):.4f}")
