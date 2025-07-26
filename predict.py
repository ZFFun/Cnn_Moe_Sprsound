import torch
from model import CNN_MoE
import torchaudio
from torchaudio.transforms import MFCC
from config import CFG
import json
import os

# 主函数定义。wav_path:输入音频路径，model_path:训练好的模型参数路径
def predict(wav_path, model_path='cnn_moe_sprsound_best.pth'):
    # 加载模型
    model = CNN_MoE()
    # 加载训练好的模型参数文件，并映射到CPU上（即使你没有GPU也能运行）
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    # 设置模型为“评估模式”，关闭Dropout、BatchNorm的训练行为
    model.eval()

    # MFCC 特征提取器
    transform = MFCC(
        sample_rate=CFG.sr,             # 设置采样率
        n_mfcc=CFG.n_mfcc,              # 设置提取的 MFCC 维度（如40维）
        melkwargs={                     # 内部 mel 频谱参数
            'n_fft': CFG.n_fft,         # 短时傅里叶变换窗口大小（如512）
            'hop_length': CFG.hop,      # 帧移（如128）
            'n_mels': CFG.n_mfcc        # Mel滤波器数量 = n_mfcc
        }
    )

    # 加载音频
    wav, sr = torchaudio.load(wav_path)
    # 如果音频采样率不是目标采样率（如8000），就进行重采样
    if sr != CFG.sr:
        wav = torchaudio.functional.resample(wav, sr, CFG.sr)
    # 如果是立体声（双通道），则转为单通道（取均值）
    wav = wav.mean(0)

    #  将waveform转换为MFCC特征图，形状：[n_mfcc, 帧数]
    spec = transform(wav)
    # 计算每个 patch 应包含的帧数，例：1.3s * 8000 / 128 ≈ 81帧
    patch_frames = int(CFG.patch_sec * CFG.sr / CFG.hop)

    # 如果帧数不足，则右侧补零
    if spec.size(1) < patch_frames:
        spec = torch.nn.functional.pad(spec, (0, patch_frames - spec.size(1)))

    # 截取前 patch_frames 帧，并添加 batch 和 channel 维度
    # 输出形状变为 [1, 1, n_mfcc, patch_frames]，比如 [1, 1, 40, 81]
    patch = spec[:, :patch_frames].unsqueeze(0).unsqueeze(0)  # (1,1,n_mfcc,frames)

    # 将patch输入模型，得到每个类别的logits（输出分数）
    logits = model(patch)
    # 取最大值对应的类别索引（例如预测为类别2）
    pred = logits.argmax(1).item()
    # 标签索引映射表，对应索引顺序：[0, 1, 2, 3, 4]
    labels = ['normal', 'crackle', 'wheeze', 'both', 'other']
    # 输出预测结果，比如“预测标签: crackle”
    print(f'预测标签: {labels[pred]}')

# 用法示例：
# predict('SPRSound/audio/example.wav')
