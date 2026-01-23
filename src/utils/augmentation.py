"""
数据增强工具 - 用于提升模型泛化能力
"""
import numpy as np
import torch
from scipy.interpolate import interp1d


def add_gaussian_noise(sequences, std=0.01):
    """
    添加高斯噪声
    Args:
        sequences: (N, T, F) numpy array
        std: 噪声标准差
    Returns:
        带噪声的序列
    """
    noise = np.random.normal(0, std, sequences.shape)
    return sequences + noise


def time_stretch(sequences, ratio=0.1):
    """
    时间拉伸/压缩
    Args:
        sequences: (N, T, F) numpy array
        ratio: 拉伸比例，例如0.1表示±10%
    Returns:
        时间拉伸后的序列
    """
    N, T, F = sequences.shape
    stretched = np.zeros_like(sequences)
    
    for i in range(N):
        # 随机选择拉伸因子
        factor = 1.0 + np.random.uniform(-ratio, ratio)
        new_length = int(T * factor)
        
        if new_length < 2:
            stretched[i] = sequences[i]
            continue
        
        # 对每个特征进行插值
        for f in range(F):
            old_x = np.linspace(0, 1, T)
            new_x = np.linspace(0, 1, new_length)
            
            # 插值
            interp_func = interp1d(old_x, sequences[i, :, f], kind='linear', fill_value='extrapolate')
            stretched_seq = interp_func(new_x)
            
            # 采样回原始长度
            sample_idx = np.linspace(0, new_length - 1, T).astype(int)
            stretched[i, :, f] = stretched_seq[sample_idx]
    
    return stretched


def random_rotation_2d(sequences, max_angle=15):
    """
    随机2D旋转（仅对xy坐标，不对velocity）
    Args:
        sequences: (N, T, F) numpy array，假设前34维是xy坐标
        max_angle: 最大旋转角度（度）
    Returns:
        旋转后的序列
    """
    N, T, F = sequences.shape
    rotated = sequences.copy()
    
    # 假设前34维是17个关键点的xy坐标
    if F < 34:
        return sequences
    
    for i in range(N):
        # 随机旋转角度
        angle = np.random.uniform(-max_angle, max_angle) * np.pi / 180
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        # 旋转矩阵
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # 对每个时间步
        for t in range(T):
            # 重塑为(17, 2)
            xy = sequences[i, t, :34].reshape(17, 2)
            
            # 应用旋转
            xy_rotated = xy @ rotation_matrix.T
            
            # 放回
            rotated[i, t, :34] = xy_rotated.flatten()
    
    return rotated


def augment_batch(X, y, aug_config):
    """
    批量数据增强
    Args:
        X: (N, T, F) 特征序列
        y: (N,) 标签
        aug_config: 增强配置字典
            - use_augmentation: bool
            - aug_noise_std: float
            - aug_time_stretch: float
    Returns:
        增强后的 (X_aug, y_aug)
    """
    if not aug_config.get('use_augmentation', False):
        return X, y
    
    X_aug = X.copy()
    
    # 高斯噪声
    noise_std = aug_config.get('aug_noise_std', 0.01)
    if noise_std > 0:
        X_aug = add_gaussian_noise(X_aug, std=noise_std)
    
    # 时间拉伸
    time_ratio = aug_config.get('aug_time_stretch', 0.1)
    if time_ratio > 0:
        X_aug = time_stretch(X_aug, ratio=time_ratio)
    
    return X_aug, y


class AugmentedDataset(torch.utils.data.Dataset):
    """
    支持数据增强的PyTorch Dataset
    """
    def __init__(self, X, y, aug_config=None):
        """
        Args:
            X: numpy array (N, T, F)
            y: numpy array (N,)
            aug_config: 增强配置
        """
        self.X = X
        self.y = y
        self.aug_config = aug_config or {}
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx:idx+1]  # (1, T, F)
        y = self.y[idx:idx+1]  # (1,)
        
        # 应用增强
        if self.aug_config.get('use_augmentation', False):
            x, y = augment_batch(x, y, self.aug_config)
        
        x = torch.tensor(x[0], dtype=torch.float32)
        y = torch.tensor(y[0], dtype=torch.long)
        
        return x, y
