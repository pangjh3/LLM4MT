src=de
tgt=en

size=10000

# for size in 10000 50000 100000 500000 1000000 5000000 10000000;do
for size in 10000;do


modelpath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/model/llms4mt-de2en-32a100/llama2-sfton-${size}-bitexts-and-alpacagpt4-and-newstests17to20
modelpath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/model/newptmodel-llms4mt-de2en-32a100/llama2-sfton-100000-bitexts-and-alpacagpt4-and-newstests17to20
#wmt23 test
datapath=/apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/WMT23
rpath=$datapath/c5alignment
mkdir -p $rpath


srcfile=./de2
tgtfile=./en2


outfile=./out.ac.txt
hypfile=$outfile.hyp

python3 ../train/attention_alignment_llama2.py --model-name-or-path $modelpath \
    -lp ${src}-${tgt} \
    -t 0.1 \
    -sa 'beam' \
    -ins /apdcephfs/share_733425/vinnylywang/jianhuipang/LLMs4MT/test/instruct_inf.txt \
    -i $srcfile \
    -rf $tgtfile \
    -o $outfile


done