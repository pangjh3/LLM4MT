from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
t13a = Tokenizer13a()

file="/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23/test.de2en.de"
file="/apdcephfs/share_733425/vinnylywang/jianhuipang/datasets/wmt23/wmt23-deen/forfairseq/tmp_shuf/test.de"

lines = open(file, 'r').readlines()

word2freq={}
with open('./tokword2freq2.txt', 'r') as f:
    for line in f:
        w, c = line.strip().split()
        word2freq[w]=int(c)

# wordtype=set()
freqblock2num={0:[],1:[],2:[],4:[],8:[],16:[],32:[],64:[],128:[],256:[],512:[],999:[],1999:[],3999:[],7999:[],15999:[],31999:[],63999:[],64000:[]}

for l in lines:
    # l = l.replace("-", " ").replace("“","").replace("„","").replace("30€","30 €").replace("10€","10 €").replace("”","").replace("®","").replace("'s"," 's").replace("test'","test")
    ws = t13a(l.strip().lower()).split()
    # ws = l.strip().split()
    for w in ws:
        if w in word2freq:
            wfreq = word2freq[w]
            for k in freqblock2num:
                if wfreq <= k:
                    freqblock2num[k].append(w)
                    break
            if wfreq >= 64000:
                freqblock2num[64000].append(w)
        else:
            if w != "'s" and w != "[" and w != "|" and w != "„" and w != "“" and w!="”":
                freqblock2num[0].append(w)
                print(w)

for k in freqblock2num:
    freqblock2num[k] = list(set(freqblock2num[k]))
    print(k, len(freqblock2num[k]))

    