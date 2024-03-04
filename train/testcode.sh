src=de
tgt=en

size=10000

# for size in 10000 50000 100000 500000 1000000 5000000 10000000;do
for size in 10000;do

# for nb in 1 2 3 8 12 20 30 50 100 200 500 1000;do


modelpath=jianhuipang/LLMs4MT/model/llms4mt-de2en-32a100/llama2-sfton-${size}-bitexts-and-alpacagpt4-and-newstests17to20
modelpath=jianhuipang/LLMs4MT/model/newptmodel-llms4mt-de2en-32a100/llama2-sfton-100000-bitexts-and-alpacagpt4-and-newstests17to20

srcfile=jianhuipang/LLMs4MT/alignment/de2
tgtfile=jianhuipang/LLMs4MT/alignment/en2


outfile=./out.testgogo.txt
hypfile=$outfile.hyp

python3 ./testcode.py --model-name-or-path $modelpath \
    -lp ${src}-${tgt} \
    -t 0.1 \
    -sa 'beam' \
    -ins jianhuipang/LLMs4MT/test/instruct_inf.txt \
    -i $srcfile \
    -rf $tgtfile \
    -o $outfile



done