# import matplotlib.pyplot as plt
import soundfile as sf
# import os
import json
# import math
import torch
from torch import nn
from torch.nn import functional as F
# from torch.utils.data import DataLoader

import commons
import utils
# from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

# hps = utils.get_hparams_from_file("C:/Users/y2657/sungil/vits/vits/configs/korean_base.json")

# net_g = SynthesizerTrn(
#     len(symbols),
#     hps.data.filter_length // 2 + 1,
#     hps.train.segment_size // hps.data.hop_length,
#     **hps.model).cpu()#cuda()
# _ = net_g.eval()

# _ = utils.load_checkpoint("C:/Users/y2657/sungil/vits/vits_model/23.02.17G_147000.pth", net_g, None)
hps_ms = utils.get_hparams_from_file("C:/Users/y2657/sungil/vits/vits_model/korms_config.json")
net_g_ms = SynthesizerTrn(
    len(symbols),
    hps_ms.data.filter_length // 2 + 1,
    hps_ms.train.segment_size // hps_ms.data.hop_length,
    n_speakers=hps_ms.data.n_speakers,
    **hps_ms.model)
_ = net_g_ms.eval()

_ = utils.load_checkpoint("C:/Users/y2657/sungil/vits/vits_model/2023_06_09_G_614000.pth", net_g_ms, None)
text = """
kss에 22000개가량 데이터를 추가해서,
a백 한대로 7일정도 돌린거같습니다.
"""
sid= torch.LongTensor([35])#107 105
stn_tst = get_text(text, hps_ms)
with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
    sf.write(f'C:/Users/y2657/sungil/vits/vits_model/test2.wav', audio, 22050, 'PCM_16')
# text = "A씨는 2020년 9월 어머니의 사망으로 어머니 소유 주택을 상속받았다."
# text = """A씨는 2020년 9월 어머니의 사망으로 어머니 소유 주택을 상속받았다.이를 물려받기 전 A씨는 한 채의 주택을 이미 보유, 2014년 8월 취득, 한 상태였다.
# 이후 상속주택을 2021년 5월 3일에 양도, 같은 달 13일에는 6년 넘게 보유하던 기존 집까지 처분한 뒤에 '1세대 1주택 비과세'로 판단해 양도소득세를 신고했다.
# 그런데 국세청은 상속주택을 먼저 팔고 기존 집을 처분했기 때문에, 기존 집 보유 기간을 '재기산,최종 1주택이 된 날인 2021년 5월 3일부터,' 해야 한다며 이를 거부했다."""
# text = ["안녕하세요! 저희는 SKT AI FellowShip 4기에서 Language-Image Multi-Modal AI 기술 연구를 맡았던 Team KEANU입니다!"]
# count = 0 
# for i in range(len(text)):
#     count += 1
#     stn_tst = get_text(text[i], hps)
#     with torch.no_grad():
#         x_tst = stn_tst.cpu().unsqueeze(0)
#         x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
#         audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
#     sf.write(f'C:/Users/y2657/sungil/vits/vits_model/a_{count}.wav', audio, 22050, 'PCM_16')
# ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
# stn_tst = get_text("data", hps_ms)
# for i in range(5,7):
#   sid = torch.LongTensor([i])
#   count += 1
    # with torch.no_grad():
    #     x_tst = stn_tst.unsqueeze(0)
    #     x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
    #     audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
    #     sf.write(f'C:/Users/y2657/sungil/project/movie/bible/genesis_{count}.wav', audio, 16000, 'PCM_16')
#   ipd.display(ipd.Audio(audio, rate=hps_ms.data.sampling_rate))
# sid = torch.LongTensor([21])
# with open("C:/Users/y2657/sungil/project/movie/sample.json", 'r', encoding="utf8") as jf:
#     data = json.load(jf)
# for keys in data:
#     if keys == "title":
#         stn_tst = get_text(data[keys], hps_ms)
#         with torch.no_grad():
#             x_tst = stn_tst.unsqueeze(0)
#             x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
#             audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
#             sf.write(f'C:/Users/y2657/sungil/project/movie/bible/title.wav', audio, 16000, 'PCM_16')
#     else:
#         summary = data.get(keys)
#         for sent in summary.values():
#             count += 1
#             stn_tst = get_text(sent, hps_ms)
#             with torch.no_grad():
#                 x_tst = stn_tst.unsqueeze(0)
#                 x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
#                 audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.float().numpy()
#                 sf.write(f'C:/Users/y2657/sungil/project/movie/bible/genesis_{count}.wav', audio, 16000, 'PCM_16')
