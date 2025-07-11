class CFG:
    data_dir = 'SPRSound'
    batch_size = 32
    epochs = 50
    lr = 1e-4

    # MFCC 特征参数
    sr = 8000            # 采样率
    n_mfcc = 40          # MFCC 维数
    hop = 128            # hop length
    n_fft = 512          # FFT window size
    patch_sec = 1.3      # 每个音频片段的时长（秒）
