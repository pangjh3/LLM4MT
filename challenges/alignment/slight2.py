import argparse
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
import torch
import random
import json
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def draw_heat_map(df, mask_data, rx_tick, sz_tick, sz_tick_num, rx_tick_num, x_label, z_label, map_title):
    # 用于画图
    # c_map = sns.cubehelix_palette(start=1.6, light=0.8, as_cmap=True, reverse=True)
    plt.subplots(figsize=(6, 6))
    # ax = sns.heatmap(df, vmax=600, vmin=500, mask=mask_data, cmap=c_map,
    #                  square=True, linewidths=0.005, xticklabels=rx_tick, yticklabels=sz_tick)

    ax = sns.heatmap(df,mask=mask_data, xticklabels=rx_tick, yticklabels=sz_tick,cmap="coolwarm")

    # ax = sns.heatmap(df,mask=mask_data, square=True, linewidths=0.005)

    # ax.set_xticks(rx_tick_num)
    # ax.set_yticks(sz_tick_num)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    # Set the font size for xticklabels
    ax.xaxis.set_tick_params(labelsize=15)

    # Set the font size for yticklabels
    ax.yaxis.set_tick_params(labelsize=15)
    # ax.set_xlabel(x_label)
    # ax.set_ylabel(z_label)
    # ax.set_title(map_title)
    plt.savefig(map_title + '.pdf')
    plt.show()
    plt.close()

model_outputs_attentions=torch.load("./attnm.today.pt")["attn_matrix"]

i="avg"
fo = open('./senttoday/layers.{}.out'.format(i), 'w')

attm = torch.mean(model_outputs_attentions[0], dim=0)



# tgt2srcattm = attm[85:, 56:79]


tgten="T od ay , ▁I ' m ▁going ▁to ▁the ▁park .".split()
srcde="He ute ▁ge he ▁ich ▁in ▁den ▁Park .".split()

prefix="<s> ▁Below ▁is ▁an ▁instruction ▁that ▁describes ▁a ▁task , ▁pa ired ▁with ▁an ▁input ▁that ▁provides ▁further ▁context . ▁Write ▁a ▁response ▁that ▁appropri ately ▁comple tes ▁the ▁request . <0x0A> <0x0A> ## # ▁Inst ruction : <0x0A> Trans late ▁the ▁following ▁sentences ▁from ▁German ▁to ▁English . <0x0A> <0x0A> ## # ▁Input : <0x0A> <0x0A> I ch ▁mag ▁Ä pf el . <0x0A> ## # ▁Response : I ▁like ▁app les . <0x0A> <0x0A> ## # ▁Input : <0x0A> B itte ▁ent sp ann en ▁Sie ▁sich ▁und ▁gen ie ßen ▁Sie ▁den ▁Park . <0x0A> <0x0A> ## # ▁Response : Please ▁relax ▁and ▁enjoy ▁the ▁park . <0x0A> <0x0A> ## # ▁Input : <0x0A> He ute ▁ge he ▁ich ▁in ▁den ▁Park . <0x0A> <0x0A> ## # ▁Response".split()

tgt2srcattm = attm[71:, 56:65]
print(model_outputs_attentions.size())
draw_heat_map(tgt2srcattm.detach().to(torch.float).cpu().numpy(), None, srcde, tgten, 0,0,0,0,'./senttoday/{}'.format(i))
print(tgt2srcattm.size(), file=fo, flush=True)
t2sscores=tgt2srcattm.detach().to(torch.float).cpu().numpy()
indexes = t2sscores.argmax(axis=1)
print(indexes, file=fo, flush=True)
for i,x in enumerate(indexes):
    print(tgten[i], srcde[x], file=fo, flush=True)
print(len(indexes), file=fo, flush=True)



layers = [i for i in range(12,15)]
for i in layers:

    fo = open('./senttoday/layers.{}.out'.format(i), 'w')

    attm = model_outputs_attentions[0][i]



    tgt2srcattm = attm[71:, 56:65]
    print(tgt2srcattm)
    print(tgt2srcattm.size())
    # minv, _ = torch.min(x, )
    # break
    # print(attm)
    # print(attm.detach().to(torch.float).cpu().numpy())
    # plt.figure()
    # plot = sns.heatmap(tgt2srcattm.detach().to(torch.float).cpu().numpy())
    # plt.savefig("./tgt2srcattn.{}.jpg".format(i))
    # plt.close()
    # if i==0 or i==31:
    draw_heat_map(tgt2srcattm.detach().to(torch.float).cpu().numpy(), None, srcde, tgten, 0,0,0,0,'./senttoday/{}'.format(i))


    print(tgt2srcattm.size(), file=fo, flush=True)
    t2sscores=tgt2srcattm.detach().to(torch.float).cpu().numpy()
    indexes = t2sscores.argmax(axis=1)
    print(indexes, file=fo, flush=True)
    for i,x in enumerate(indexes):
        print(tgten[i], srcde[x], file=fo, flush=True)
    print(len(indexes), file=fo, flush=True)

