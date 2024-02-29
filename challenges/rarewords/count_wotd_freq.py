import sacrebleu
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
t13a = Tokenizer13a()

# 295805439
srcfile="/apdcephfs/share_733425/vinnylywang/jianhuipang/datasets/wmt23/wmt23-deen/forfairseq/tmp_shuf/train.de"
# srcfile="./10.tmp"

dicttofreq={}
with open(srcfile, "r") as f:
    for line in f:
        ws = t13a(line.strip().lower()).split()
        for w in ws:
            if w not in dicttofreq:
                dicttofreq[w] = 1
            else:
                dicttofreq[w] += 1


r = sorted(dicttofreq.items(), key=lambda x: x[1])

for w, f in r:
    print("{} {}".format(w, f))