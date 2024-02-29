##############################
# Function: convert bilingual sentence pairs to alpaca data format
# Author: Wenxiang Jiao
# Last modified: 2023/04/15
##############################

import argparse
import time
import json
from tqdm import tqdm
import random
import numpy as np
import csv, json

random.seed(520)
flags=[0,1]
# Instrauct language
lang_instruction = {
    'de': {'de': "Deutsch", 'en': "Englisch", 'ja': "Japanisch", 'zh': "Chinesisch"},
    'en': {'de': "German", 'en': "English", 'ja': "Japanese", 'zh': "Chinese"},
    'ja': {'de': "ドイツ語", 'en': "英語", 'ja': "日本語", 'zh': "中国語"},
    'zh': {'de': "德语", 'en': "英语", 'ja': "日语", 'zh': "中文"},
}


def write_json(src, tgt, in_file, out_file):
    inlines = open(in_file, 'r', encoding='utf-8').readlines()
    with open(out_file, 'w', encoding='utf-8') as fo:
        # data = dict()
        for line in inlines:
            # if rand.choice(flags) == 0:
            sl, tl = line.strip().split('\t')
            text = {"text":"[{}]: {}\n\n[{}]: {}".format(src, sl, tgt, tl)}
            # text = {"text":"{}\n {}".format(sl, tl)}
            jsoned = json.dumps(text, ensure_ascii=False)
            fo.write(jsoned)
            fo.write('\n')



if __name__ == "__main__":
    """
    python3 ../create_s2t_alpaca.py -sf train.en-de.en -tf train.en-de.de -s en -t de -if ../instruct_follow.txt -of data_pair_alp.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-s', type=str, required=True, help='src language, en, de, ja, zh')
    parser.add_argument('--tgt', '-t', type=str, required=True, help='tgt language, en, de, ja, zh')
    # parser.add_argument('--src-file','-sf', type=str, required=True, help='src file')
    # parser.add_argument('--tgt-file','-tf', type=str, required=True, help='tgt file')
    parser.add_argument('--in-file','-if', type=str, required=True, help='in file')
    parser.add_argument('--out-file','-of', type=str, required=True, help='out file')
    args = parser.parse_args()
    src, tgt = args.src, args.tgt
    in_file=args.in_file
    out_file = args.out_file

    # Start
    write_json(src, tgt, in_file, out_file)
