import torch
from model import CNN_MoE
import torchaudio
from torchaudio.transforms import MFCC
from config import CFG
import json
import os

def predict(wav_path, model_path='cnn_moe_sprsound_best.pth'):
    # 加载模型
    model = CNN_MoE()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # MFCC 特征提取器
    transform = MFCC(
        sample_rate=CFG.sr,
        n_mfcc=CFG.n_mfcc,
        melkwargs={
            'n_fft': CFG.n_fft,
            'hop_length': CFG.hop,
            'n_mels': CFG.n_mfcc
        }
    )

    # 加载音频
    wav, sr = torchaudio.load(wav_path)
    if sr != CFG.sr:
        wav = torchaudio.functional.resample(wav, sr, CFG.sr)
    wav = wav.mean(0)

    # 提取 MFCC 特征
    spec = transform(wav)
    patch_frames = int(CFG.patch_sec * CFG.sr / CFG.hop)

    if spec.size(1) < patch_frames:
        spec = torch.nn.functional.pad(spec, (0, patch_frames - spec.size(1)))
    patch = spec[:, :patch_frames].unsqueeze(0).unsqueeze(0)  # (1,1,n_mfcc,frames)

    # 推理
    logits = model(patch)
    pred = logits.argmax(1).item()
    labels = ['normal', 'crackle', 'wheeze', 'both', 'other']
    print(f'预测标签: {labels[pred]}')

# 用法示例：
# predict('SPRSound/audio/example.wav')
