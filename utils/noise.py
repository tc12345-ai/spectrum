"""
AWGN信道模型
添加加性高斯白噪声到信号
"""

import numpy as np
from typing import Tuple, Optional


class AWGNChannel:
    """AWGN信道类"""
    
    def __init__(self, snr_db: float = 10.0):
        """
        初始化AWGN信道
        
        Args:
            snr_db: 信噪比 (dB)
        """
        self.snr_db = snr_db
        
    def set_snr(self, snr_db: float):
        """设置信噪比"""
        self.snr_db = snr_db
        
    def add_noise(self, signal: np.ndarray, 
                  snr_db: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """
        向信号添加AWGN噪声
        
        Args:
            signal: 输入信号（复数或实数）
            snr_db: 信噪比 (dB)，如果不指定则使用实例设置
            
        Returns:
            noisy_signal: 加噪后的信号
            noise_power: 噪声功率
        """
        if snr_db is None:
            snr_db = self.snr_db
            
        # 计算信号功率
        signal_power = np.mean(np.abs(signal)**2)
        
        # 计算噪声功率
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # 生成噪声
        if np.iscomplexobj(signal):
            # 复数噪声
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
            )
        else:
            # 实数噪声
            noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        
        noisy_signal = signal + noise
        
        return noisy_signal, noise_power
    
    @staticmethod
    def generate_noise_only(num_samples: int, 
                           noise_power: float = 1.0,
                           complex_noise: bool = True) -> np.ndarray:
        """
        仅生成噪声信号（H0假设：无信号）
        
        Args:
            num_samples: 样本数
            noise_power: 噪声功率
            complex_noise: 是否生成复数噪声
            
        Returns:
            noise: 噪声信号
        """
        if complex_noise:
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(num_samples) + 1j * np.random.randn(num_samples)
            )
        else:
            noise = np.sqrt(noise_power) * np.random.randn(num_samples)
            
        return noise
    
    @staticmethod
    def estimate_noise_power(signal: np.ndarray, 
                            method: str = 'mean') -> float:
        """
        估计信号中的噪声功率
        
        Args:
            signal: 输入信号
            method: 估计方法 ('mean', 'median')
            
        Returns:
            estimated_power: 估计的功率
        """
        power_samples = np.abs(signal)**2
        
        if method == 'mean':
            return np.mean(power_samples)
        elif method == 'median':
            # 中值估计更鲁棒
            return np.median(power_samples) / 0.6931  # ln(2) 校正因子
        else:
            return np.mean(power_samples)
