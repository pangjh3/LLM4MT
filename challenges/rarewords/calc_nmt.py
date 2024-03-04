from string import punctuation
from sacremoses import MosesTruecaser, MosesTokenizer
import numpy as np
mtok = MosesTokenizer(lang='en')

srcfile="jianhuipang/datasets/wmt23/wmt23-deen/forfairseq/tmp_shuf/test.de"
tgtfile="jianhuipang/datasets/wmt23/wmt23-deen/forfairseq/tmp_shuf/test.en"
refile="jianhuipang/fairseq/wmt23models2/results/de2en.10000000.5.hyp.out"
alignfile="./wmt23test.align2.out"
testdatafile="./test.data2.out"


srclines=open(srcfile, 'r').readlines()
tgtlines=open(tgtfile, 'r').readlines()

# alignfile="./wmt23test.align2.out"
alignlines=open(alignfile, 'r').readlines()
refalignlines=alignlines[:549]
llmsalignlines=alignlines[549:549+549]
nmtalignlines=alignlines[549+549:]

puns = list(punctuation)

# testdatafile="./test.data2.out"
datalines=open(testdatafile, 'r').readlines()
refdatalines = datalines[:549]
llmsdatalines = datalines[549:549+549]
nmtdatalines = datalines[549+549:]

# refile="jianhuipang/LLMs4MT/test/WMT23/newptmodel/test.de2en.model.50000.out.hyp"
# refile="jianhuipang/fairseq/wmt23models2/results/de2en.10000000.5.hyp.out"
reflines = ["" for i in range(len(tgtlines))]
outlines = open(refile, 'r').readlines()
i=0
for l in outlines:
    if l.startswith("H-"):
        id=int(l.strip().split('\t')[0][2:])
        reflines[id] = l.strip().split('\t')[-1]
# def rare_score(freq, )
for i, l in enumerate(reflines):
    if l == "":
        nmtalignlines.insert(i, "")
        nmtdatalines.insert(i,"")

# print(len(refalignlines))
# print(len(nmtalignlines), len(nmtdatalines))


word2freq={}
with open('./wordfreq.txt', 'r') as f:
    for line in f:
        w, c = line.strip().split()
        word2freq[w]=int(c)

score={}
deletew={}
swfreq={}
ii=0
for refdata, refalign, mtdata, mtalign in zip(refdatalines, refalignlines, nmtdatalines, nmtalignlines):
    ii+=1
    if mtdata == "":
        continue
    refs, reft = refdata.strip().split(" ||| ")
    refsws = refs.strip().split()
    reftws = reft.strip().split()
    mts, mtt = mtdata.strip().split(" ||| ")
    mtsws = mts.strip().split()
    mttws = mtt.strip().split()
    # print(refsws)
    # print(reftws)
    # print(mtsws)
    # print(mttws)

    refalign = refalign.strip().split()
    refid2id={}
    for al in refalign:
        x,y=al.strip().split('-')
        x,y=int(x),int(y)
        if x in refid2id:
            refid2id[x].append(y)
        else:
            refid2id[x] = [y]

    mtalign = mtalign.strip().split()
    mtid2id={}
    for al in mtalign:
        x,y=al.strip().split('-')
        x,y=int(x),int(y)
        if x in mtid2id:
            mtid2id[x].append(y)
        else:
            mtid2id[x] = [y]

    # print(id2id)
    for i, w in enumerate(refsws):
        if w in puns:
            continue
        # w = w.lower()
        if w not in score:
            score[w] = 0.0
        if w not in deletew:
            deletew[w] = 0.0
        if w in swfreq:
            swfreq[w] += 1
        else:
            swfreq[w] = 1
        dflag=1
        if i in refid2id:
            if w in mtsws:
                mtsindex = mtsws.index(w)
                if mtsindex in mtid2id:
                    refalignedtwids = refid2id[i]
                    mtalignedtwids = mtid2id[mtsindex]

                    scorei = 0.0
                    for rtid in refalignedtwids:
                        reftarw = reftws[rtid]
                        rwcounttimes = 0.0
                        mtwcounttimes = 0.0
                        for www in reftws:
                            if www == reftarw:
                                rwcounttimes+=1.0
                        for mttid in mtalignedtwids:
                            mttidw = mttws[mttid]
                            for www in mttws:
                                if www == mttidw:
                                    mtwcounttimes += 1.0
                        if mtwcounttimes > rwcounttimes:
                            scorei += rwcounttimes/mtwcounttimes
                        elif mtwcounttimes == 0 and rwcounttimes == 0:
                            scorei += 0.0
                        elif mtwcounttimes == rwcounttimes:
                            scorei += 1.0
                        else:
                            scorei += mtwcounttimes/rwcounttimes

                    ss = scorei/len(refalignedtwids)
    
                    # ss = min(float(len(mtalignedtwids)/len(refalignedtwids)), 1.0)

                    if w in word2freq:
                        if 2<=word2freq[w]<4:
                            print('--', ii, w, mtalignedtwids, refalignedtwids)

                    score[w] += ss
    
                else:
                    deletew[w] += 1.0
                    

freqblock2score={0:[],1:[],2:[],4:[],8:[],16:[],32:[],64:[],128:[],256:[],512:[],999:[],1999:[],3999:[],7999:[],15999:[],31999:[],63999:[],64000:[]}
deleteblock2score={0:[],1:[],2:[],4:[],8:[],16:[],32:[],64:[],128:[],256:[],512:[],999:[],1999:[],3999:[],7999:[],15999:[],31999:[],63999:[],64000:[]}
blocklist=list(freqblock2score.keys())
# print(blocklist)
freq2acc={}
# print(score)
score2={}
delete2={}
for w in score:
    if w not in word2freq:
        word2freq[w]=0
    # else:
    #     word2freq[w]+=1

for w in score:

    score2[w] = score[w]/swfreq[w]
    if w not in deletew:
        deletew[w] = 0.0
    delete2[w] = float(deletew[w]/swfreq[w])
    # print("{} {} {} {}".format(w, score2[w], score[w], swfreq[w]))
    freq=word2freq[w]
    # print(freq)
    if freq in freq2acc:
        freq2acc[freq].append(score2[w])
    else:
        freq2acc[freq] = [score2[w]]

    bi = 0
    for bs in blocklist:
        if freq <= bs:
            bi = bs
            break
        bi = 64000
    # print(bi)
    freqblock2score[bi].append(score2[w])
    deleteblock2score[bi].append(delete2[w])

for bi in freqblock2score:
    sl = freqblock2score[bi]
    # print(sl)
    freqblock2score[bi] = np.mean(sl)

    sdl=deleteblock2score[bi]
    deleteblock2score[bi] = np.mean(sdl)

    print("{} {} {}".format(bi, freqblock2score[bi], deleteblock2score[bi]))
