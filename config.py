import os
import torch


class CFG:
    # 数据集根目录路径
    data_dir = 'SPRSound'

    # 训练超参数
    batch_size = 32
    epochs = 50
    lr = 1e-4

    # MFCC 特征参数
    sr = 8000  # 采样率
    n_mfcc = 40  # MFCC 维数
    hop = 128  # hop length
    n_fft = 512  # FFT window size
    patch_sec = 1.3  # 每个音频片段的时长（秒）

    # 专家网络配置
    num_experts = 4  # 专家网络数量
    expert_dim = 256  # 专家网络隐藏层维度

    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据集路径配置
    train_wav_dir = os.path.join(data_dir, 'train_wav')
    train_json_dir = os.path.join(data_dir, 'train_json')
    test_wav_dir = os.path.join(data_dir, 'test_wav')
    test_json_dir = os.path.join(data_dir, 'test_json')

    # 数据集划分文件
    train_list = os.path.join(data_dir, 'list_train.txt')
    val_list = os.path.join(data_dir, 'list_val.txt')
    test_list = os.path.join(data_dir, 'list_test.txt')