import os, json, random, glob
import torch
from torch.utils.data import Dataset
from torchaudio.transforms import MFCC
import soundfile as sf   # 用 pysoundfile 读取 wav

class SPRSoundDataset(Dataset):
    """
    SPRSoundDataset：用于加载 SPRSound 数据集，输出 MFCC 特征块和标签
    root	数据集所在的根目录。指向包含音频文件或标签文件的文件夹路径。
    split='train'	数据集的划分方式，指定当前加载的是训练集（'train'）、验证集（'val'）还是测试集（'test'）。
    patch_sec=1.3	每个音频样本的时间长度，单位是秒（s）。根据文献，平均每个事件持续时间约为 1.3秒。
    sr=8000	采样率（sampling rate），表示每秒钟采样的点数，单位是 Hz。论文明确指出录音使用 8kHz 采样率。
    n_mfcc=40	提取 MFCC（Mel 频率倒谱系数）特征时，保留了前 40 个系数，每一帧的音频会被压缩成 40 维的向量。
    hop=128	hop length，表示在短时傅里叶变换（STFT）中，每次窗函数滑动的步长（单位：采样点）。为 8kHz 设置 hop=128 (~16ms)。
    n_fft=512	FFT 的窗口大小，表示进行快速傅里叶变换时使用多少点来计算频谱。512 对应 64ms 窗口。
    """

    def __init__(self, root, split='train', patch_sec=2.6,
                 sr=8000, n_mfcc=40, hop=128, n_fft=512):
        self.root = root
        self.split = split

        # 根据 split 类型确定要使用的目录
        if split in ('train', 'val'):
            json_dir = os.path.join(root, 'train_json')
            wav_dir  = os.path.join(root, 'train_wav')
        else:  # test
            json_dir = os.path.join(root, 'test_json')
            wav_dir  = os.path.join(root, 'test_wav')

        # 获取所有 JSON 文件列表（去掉扩展名）
        json_files    = glob.glob(os.path.join(json_dir, '*.json'))
        self.file_list = [os.path.splitext(os.path.basename(f))[0] for f in json_files]

        # train/val 划分（20% 验证）
        random.seed(42)
        random.shuffle(self.file_list)
        split_idx = int(len(self.file_list) * 0.2)
        if split == 'val':
            self.file_list = self.file_list[:split_idx]
        elif split == 'train':
            self.file_list = self.file_list[split_idx:]

        print(f"Loaded {split} dataset with {len(self.file_list)} files")

        # MFCC 特征提取器
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

        # 计算 patch 对应多少帧
        self.patch_frames = int(patch_sec * sr / hop)  # 1.3s * 8000 / 128 ≈ 81 帧

        # **只映射 record_annotation 的 5 个类别**
        self.classes = ['Normal', 'CAS', 'DAS', 'CAS & DAS', 'Poor Quality']
        self.lbl2idx  = {lab: i for i, lab in enumerate(self.classes)}

    def __len__(self):
        return len(self.file_list)

    # wav_path 是你要加载的音频文件路径
    def _load_audio(self, wav_path):
        # 使用 soundfile 读取 wav 并转为单通道 Tensor
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Audio file {wav_path} not found!")
        data, sr = sf.read(wav_path, dtype='float32')  # data: np.ndarray, sr: int
        wav = torch.from_numpy(data).float()            # 转成 FloatTensor
        if wav.ndim > 1:
            wav = wav.mean(dim=1)                       # 多通道取平均
        # 如果不是 8kHz，做线性插值重采样
        if sr != 8000:
            wav = torch.nn.functional.interpolate(
                wav.unsqueeze(0).unsqueeze(0),
                scale_factor=8000/sr,
                mode='linear',
                align_corners=False
            ).squeeze()
        return wav

    def __getitem__(self, idx):
        rec = self.file_list[idx]
        # 构造 json / wav 路径
        if self.split in ('train', 'val'):
            json_dir = os.path.join(self.root, 'train_json')
            wav_dir  = os.path.join(self.root, 'train_wav')
        else:
            json_dir = os.path.join(self.root, 'test_json')
            wav_dir  = os.path.join(self.root, 'test_wav')

        meta = json.load(open(os.path.join(json_dir, rec + '.json')))
        # 只读取 record_annotation
        label_str = meta.get('record_annotation')
        if label_str not in self.lbl2idx:
            raise KeyError(f"Unexpected record_annotation {label_str} in {rec}")
        label = self.lbl2idx[label_str]

        wav  = self._load_audio(os.path.join(wav_dir, rec + '.wav'))
        spec = self.mfcc(wav)
        # pad to patch_frames
        if spec.size(1) < self.patch_frames:
            spec = torch.nn.functional.pad(
                spec, (0, self.patch_frames - spec.size(1))
            )
        # 随机截取 patch
        start = random.randint(0, spec.size(1) - self.patch_frames)
        patch = spec[:, start:start + self.patch_frames]
        # 返回 [1, n_mfcc, frames], label
        return patch.unsqueeze(0), label

