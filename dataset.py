import os, json, random
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import MFCC

class SPRSoundDataset(Dataset):
    """
    SPRSoundDataset：用于加载 SPRSound 数据集，输出 MCFF 特征块和标签
    root	数据集所在的根目录。指向包含音频文件或标签文件的文件夹路径。
    split='train'	数据集的划分方式，指定当前加载的是训练集（'train'）、验证集（'val'）还是测试集（'test'）。
    patch_sec=1.3	每个音频样本的时间长度，单位是秒（s）。根据文献，平均每个事件持续时间约为 1.3秒。
    sr=8000	采样率（sampling rate），表示每秒钟采样的点数，单位是 Hz。论文明确指出录音使用 8kHz 采样率。
    n_mfcc=40	提取 MFCC（Mel 频率倒谱系数）特征时，保留了前 40 个系数，每一帧的音频会被压缩成 40 维的向量。。
    hop=128	hop length，表示在短时傅里叶变换（STFT）中，每次窗函数滑动的步长（单位：采样点数）。为 8000Hz 设置较合适的 hop size，较细粒度的时间分辨率（比如 10ms）: hop = sr * 0.016 = 128
    n_fft=512	FFT 的窗口大小，表示进行快速傅里叶变换（FFT）时使用多少个点来计算频谱。这个值越大，频率分辨率越高，但时间分辨率会降低。用于 8000Hz 采样率时的典型窗口长度（64ms），论文中也有参考 Log-Mel 的 frame_length=2048 与 hop=512 是基于16kHz，相应缩小

    """

    def __init__(self, root, split='train', patch_sec=1.3,
                 sr=8000, n_mfcc=40, hop=128, n_fft=512):
        self.root = root
        self.split = split
    #拼接路径
        with open(os.path.join(root, f'list_{split}.txt')) as f:
            #.strip() 去除行首行尾的空格、\n 换行符，最后得到字符串列表
            self.file_list = [l.strip() for l in f]

        # MFCC 特征提取器
        # waveform
        #  → STFT（短时傅里叶变换）
        #  → Mel 滤波器组（MelSpectrogram）
        #  → 对数取值（AmplitudeToDB）
        #  → 离散余弦变换（DCT）
        #  → 得到 MFCC（只保留前 n_mfcc 个系数）
        self.mfcc = MFCC(
            sample_rate=sr,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop,
                'n_mels': n_mfcc,
                'center': True
            }
        )

        #计算给定音频片段（patch）的时长对应多少帧（frame）MFCC 特征
        #原始音频的长度可能不一致，或提取出的 MFCC 特征帧数不同。从特征图中“随机截取”一个 patch（小块），作为训练数据
        self.patch_frames = int(patch_sec * sr / hop)  # 例：1.3s * 8000 / 128 ≈ 81帧

        # 标签映射表
        self.lbl2idx = {'normal': 0, 'crackle': 1, 'wheeze': 2, 'both': 3, 'other': 4}

    def __len__(self):
        return len(self.file_list)

    #wav_path 是你要加载的音频文件路径
    def _load_audio(self, wav_path):
        #导入PyTorch官方提供的torchaudio库，用于加载、处理音频数据
        import torchaudio
        #wav是音频的Tensor，形状通常是[通道数, 采样点数]。sr是音频的实际采样率。
        wav, sr = torchaudio.load(wav_path)
        if sr != 8000:
            #将音频从原始采样率sr转为8000Hz
            wav = torchaudio.functional.resample(wav, sr, 8000)
        return wav.mean(0)  # 单通道。如多通道，对多个通道求平均

    def __getitem__(self, idx):
        #取出第idx个音频样本的ID
        rec = self.file_list[idx]
        #构造该样本对应的元数据路径和音频文件路径
        meta_path = os.path.join(self.root, 'meta', rec + '.json')
        wav_path = os.path.join(self.root, 'audio', rec + '.wav')
        meta = json.load(open(meta_path))
        label = self.lbl2idx[meta['label']]
        wav = self._load_audio(wav_path)

        # 调用 MFCC 提取器，将 waveform 转换为 MFCC 特征图
        spec = self.mfcc(wav)
        # 裁切为 patch，如果帧数不足（如帧数 < 81），用 0 在右侧补齐
        if spec.size(1) < self.patch_frames:
            spec = torch.nn.functional.pad(spec, (0, self.patch_frames - spec.size(1)))
        #从完整MFCC特征图中随机截取一个patch，random.randint(...) 使模型每次训练看到不同片段（数据增强）
        start = random.randint(0, spec.size(1) - self.patch_frames)
        patch = spec[:, start:start + self.patch_frames]

        #patch.unsqueeze(0)：给 patch 加一个维度，变成 [1, n_mfcc, frames]，适配 CNN 的输入格式（1个通道）
        #最终返回：patch: [1, 40, 81]，label: 数字形式的分类标签（如 0、1、2）
        return patch.unsqueeze(0), label
