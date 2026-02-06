"""
能量检测器
实现基于能量的频谱感知算法
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional


class EnergyDetector:
    """能量检测器类"""
    
    def __init__(self, num_samples: int = 1000, 
                 noise_power: float = 1.0):
        """
        初始化能量检测器
        
        Args:
            num_samples: 检测使用的样本数
            noise_power: 已知或估计的噪声功率
        """
        self.num_samples = num_samples
        self.noise_power = noise_power
        self.threshold = None
        
    def set_noise_power(self, noise_power: float):
        """设置噪声功率"""
        self.noise_power = noise_power
        
    def compute_test_statistic(self, signal: np.ndarray) -> float:
        """
        计算检测统计量
        
        T = (1/N) * sum(|y[n]|^2)
        
        Args:
            signal: 接收信号
            
        Returns:
            test_stat: 检测统计量
        """
        # 使用实际样本数或指定的样本数
        samples = signal[:self.num_samples] if len(signal) >= self.num_samples else signal
        test_stat = np.mean(np.abs(samples)**2)
        return test_stat
    
    def calibrate_threshold(self, target_pf: float, 
                           method: str = 'theoretical') -> float:
        """
        自动门限校准：给定目标虚警概率Pf，计算检测门限
        
        Args:
            target_pf: 目标虚警概率
            method: 校准方法 ('theoretical', 'simulation')
            
        Returns:
            threshold: 检测门限
        """
        N = self.num_samples
        sigma2 = self.noise_power
        
        if method == 'theoretical':
            # 理论方法：利用卡方分布
            # 在H0下，2*N*T/sigma^2 ~ chi2(2N) (对于复数信号)
            # T/sigma^2 ~ chi2(2N)/(2N)
            # 因此 T ~ sigma^2 * chi2(2N)/(2N)
            
            # P(T > gamma | H0) = Pf
            # P(chi2(2N)/(2N) > gamma/sigma^2) = Pf
            # 1 - F_chi2(2N * gamma / sigma^2) = Pf
            # gamma = sigma^2 * F_chi2_inv(1-Pf) / (2N)
            
            # 简化：直接使用Gamma分布
            # T ~ Gamma(N, sigma^2/N) under H0
            shape = N
            scale = sigma2 / N
            threshold = stats.gamma.ppf(1 - target_pf, shape, scale=scale)
            
        elif method == 'simulation':
            # 仿真方法：Monte Carlo
            num_trials = 10000
            test_stats = []
            
            for _ in range(num_trials):
                # 生成纯噪声（H0假设）
                noise = np.sqrt(sigma2 / 2) * (
                    np.random.randn(N) + 1j * np.random.randn(N)
                )
                test_stats.append(np.mean(np.abs(noise)**2))
            
            # 找到对应Pf的门限
            test_stats = np.sort(test_stats)[::-1]  # 降序排列
            idx = int(target_pf * num_trials)
            threshold = test_stats[idx] if idx < num_trials else test_stats[-1]
        
        else:
            raise ValueError(f"不支持的校准方法: {method}")
        
        self.threshold = threshold
        return threshold
    
    def detect(self, signal: np.ndarray, 
               threshold: Optional[float] = None) -> Tuple[bool, float]:
        """
        执行能量检测
        
        Args:
            signal: 接收信号
            threshold: 检测门限（如果不指定则使用校准的门限）
            
        Returns:
            decision: 检测决策 (True = 信号存在, False = 仅噪声)
            test_stat: 检测统计量
        """
        if threshold is None:
            threshold = self.threshold
            
        if threshold is None:
            raise ValueError("需要先校准门限或指定门限值")
        
        test_stat = self.compute_test_statistic(signal)
        decision = test_stat > threshold
        
        return decision, test_stat
    
    def compute_theoretical_pd(self, snr_db: float, 
                               threshold: Optional[float] = None) -> float:
        """
        计算理论检测概率Pd
        
        Args:
            snr_db: 信噪比 (dB)
            threshold: 检测门限
            
        Returns:
            pd: 检测概率
        """
        if threshold is None:
            threshold = self.threshold
            
        if threshold is None:
            raise ValueError("需要先校准门限或指定门限值")
        
        N = self.num_samples
        sigma2 = self.noise_power
        snr = 10 ** (snr_db / 10)
        
        # 在H1下，信号+噪声
        # 平均功率 = sigma^2 * (1 + SNR)
        signal_plus_noise_power = sigma2 * (1 + snr)
        
        # T ~ Gamma(N, (sigma^2 + P_s)/N) under H1
        shape = N
        scale = signal_plus_noise_power / N
        
        # Pd = P(T > threshold | H1)
        pd = 1 - stats.gamma.cdf(threshold, shape, scale=scale)
        
        return pd
    
    def compute_theoretical_pf(self, threshold: float) -> float:
        """
        计算理论虚警概率Pf
        
        Args:
            threshold: 检测门限
            
        Returns:
            pf: 虚警概率
        """
        N = self.num_samples
        sigma2 = self.noise_power
        
        # T ~ Gamma(N, sigma^2/N) under H0
        shape = N
        scale = sigma2 / N
        
        # Pf = P(T > threshold | H0)
        pf = 1 - stats.gamma.cdf(threshold, shape, scale=scale)
        
        return pf
    
    def monte_carlo_performance(self, signal_generator, 
                                snr_db: float,
                                threshold: float,
                                num_trials: int = 1000) -> Tuple[float, float]:
        """
        Monte Carlo仿真评估检测性能
        
        Args:
            signal_generator: 信号生成函数
            snr_db: 信噪比 (dB)
            threshold: 检测门限
            num_trials: 仿真次数
            
        Returns:
            pd: 检测概率
            pf: 虚警概率
        """
        from ..utils.noise import AWGNChannel
        
        N = self.num_samples
        sigma2 = self.noise_power
        channel = AWGNChannel(snr_db)
        
        detections_h1 = 0  # H1下的检测次数
        false_alarms_h0 = 0  # H0下的虚警次数
        
        for _ in range(num_trials):
            # H0: 仅噪声
            noise_only = np.sqrt(sigma2 / 2) * (
                np.random.randn(N) + 1j * np.random.randn(N)
            )
            _, test_stat_h0 = self.detect(noise_only, threshold)
            if test_stat_h0 > threshold:
                false_alarms_h0 += 1
            
            # H1: 信号+噪声
            signal = signal_generator(N)
            noisy_signal, _ = channel.add_noise(signal)
            _, test_stat_h1 = self.detect(noisy_signal, threshold)
            if test_stat_h1 > threshold:
                detections_h1 += 1
        
        pd = detections_h1 / num_trials
        pf = false_alarms_h0 / num_trials
        
        return pd, pf
    
    def get_info(self) -> dict:
        """获取检测器信息"""
        return {
            'detector_type': 'Energy Detector',
            'num_samples': self.num_samples,
            'noise_power': self.noise_power,
            'threshold': self.threshold
        }
