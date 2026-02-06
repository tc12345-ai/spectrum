"""
循环平稳检测器
实现基于循环平稳特性的频谱感知算法
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Tuple, Optional


class CyclostationaryDetector:
    """循环平稳检测器类"""
    
    def __init__(self, num_samples: int = 1024,
                 sample_rate: float = 1.0,
                 alpha_resolution: int = 64):
        """
        初始化循环平稳检测器
        
        Args:
            num_samples: 检测使用的样本数
            sample_rate: 采样率 (Hz)
            alpha_resolution: 循环频率分辨率
        """
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.alpha_resolution = alpha_resolution
        self.threshold = None
        
    def compute_cyclic_autocorrelation(self, signal: np.ndarray,
                                       alpha: float,
                                       max_tau: int = 32) -> np.ndarray:
        """
        计算循环自相关函数 (CAF)
        
        R_x^alpha(tau) = E[x(t+tau/2) * x*(t-tau/2) * exp(-j*2*pi*alpha*t)]
        
        Args:
            signal: 输入信号
            alpha: 循环频率
            max_tau: 最大时延
            
        Returns:
            caf: 循环自相关函数
        """
        N = len(signal)
        tau_range = np.arange(-max_tau, max_tau + 1)
        caf = np.zeros(len(tau_range), dtype=complex)
        
        # 时间向量
        t = np.arange(N) / self.sample_rate
        
        for i, tau in enumerate(tau_range):
            if tau >= 0:
                x1 = signal[tau:]
                x2 = np.conj(signal[:N-tau])
                t_valid = t[tau:]
            else:
                x1 = signal[:N+tau]
                x2 = np.conj(signal[-tau:])
                t_valid = t[:N+tau]
            
            # 计算循环自相关
            exp_term = np.exp(-1j * 2 * np.pi * alpha * t_valid)
            caf[i] = np.mean(x1 * x2 * exp_term)
        
        return caf
    
    def compute_spectral_correlation(self, signal: np.ndarray,
                                     alpha: float,
                                     nfft: int = 256) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算谱相关函数 (SCF)
        
        S_x^alpha(f) = F{R_x^alpha(tau)}
        
        Args:
            signal: 输入信号
            alpha: 循环频率
            nfft: FFT大小
            
        Returns:
            freq: 频率向量
            scf: 谱相关函数
        """
        max_tau = nfft // 2
        caf = self.compute_cyclic_autocorrelation(signal, alpha, max_tau)
        
        # 填充到nfft大小
        caf_padded = np.zeros(nfft, dtype=complex)
        start = (nfft - len(caf)) // 2
        caf_padded[start:start+len(caf)] = caf
        
        # FFT
        scf = np.fft.fftshift(np.fft.fft(caf_padded))
        freq = np.fft.fftshift(np.fft.fftfreq(nfft, 1/self.sample_rate))
        
        return freq, scf
    
    def compute_cyclic_spectrum(self, signal: np.ndarray,
                                alpha_range: Optional[np.ndarray] = None,
                                nfft: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算完整的循环谱
        
        Args:
            signal: 输入信号
            alpha_range: 循环频率范围
            nfft: FFT大小
            
        Returns:
            alpha: 循环频率向量
            freq: 频率向量
            cyclic_spectrum: 循环谱 (2D矩阵)
        """
        if alpha_range is None:
            # 默认循环频率范围
            alpha_max = self.sample_rate / 2
            alpha_range = np.linspace(-alpha_max, alpha_max, self.alpha_resolution)
        
        cyclic_spectrum = np.zeros((len(alpha_range), nfft), dtype=complex)
        
        for i, alpha in enumerate(alpha_range):
            freq, scf = self.compute_spectral_correlation(signal, alpha, nfft)
            cyclic_spectrum[i, :] = scf
        
        return alpha_range, freq, cyclic_spectrum
    
    def compute_test_statistic(self, signal: np.ndarray,
                               known_alpha: Optional[float] = None) -> float:
        """
        计算检测统计量
        
        Args:
            signal: 接收信号
            known_alpha: 已知的循环频率（如果不指定则搜索峰值）
            
        Returns:
            test_stat: 检测统计量
        """
        samples = signal[:self.num_samples] if len(signal) >= self.num_samples else signal
        
        if known_alpha is not None:
            # 使用已知循环频率
            caf = self.compute_cyclic_autocorrelation(samples, known_alpha)
            # 检测统计量：循环频率处的能量
            test_stat = np.sum(np.abs(caf)**2)
        else:
            # 搜索最大循环相关
            alpha_max = self.sample_rate / 4
            alpha_range = np.linspace(-alpha_max, alpha_max, self.alpha_resolution)
            
            max_energy = 0
            for alpha in alpha_range:
                if abs(alpha) < 1e-6:  # 跳过alpha=0（普通相关）
                    continue
                caf = self.compute_cyclic_autocorrelation(samples, alpha)
                energy = np.sum(np.abs(caf)**2)
                max_energy = max(max_energy, energy)
            
            test_stat = max_energy
        
        return test_stat
    
    def calibrate_threshold(self, target_pf: float,
                            num_trials: int = 1000,
                            noise_power: float = 1.0) -> float:
        """
        通过仿真校准检测门限
        
        Args:
            target_pf: 目标虚警概率
            num_trials: 仿真次数
            noise_power: 噪声功率
            
        Returns:
            threshold: 检测门限
        """
        test_stats = []
        
        for _ in range(num_trials):
            # 生成纯噪声（H0假设）
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(self.num_samples) + 
                1j * np.random.randn(self.num_samples)
            )
            test_stats.append(self.compute_test_statistic(noise))
        
        # 找到对应Pf的门限
        test_stats = np.sort(test_stats)[::-1]
        idx = int(target_pf * num_trials)
        threshold = test_stats[idx] if idx < num_trials else test_stats[-1]
        
        self.threshold = threshold
        return threshold
    
    def detect(self, signal: np.ndarray,
               threshold: Optional[float] = None,
               known_alpha: Optional[float] = None) -> Tuple[bool, float]:
        """
        执行循环平稳检测
        
        Args:
            signal: 接收信号
            threshold: 检测门限
            known_alpha: 已知的循环频率
            
        Returns:
            decision: 检测决策
            test_stat: 检测统计量
        """
        if threshold is None:
            threshold = self.threshold
            
        if threshold is None:
            raise ValueError("需要先校准门限或指定门限值")
        
        test_stat = self.compute_test_statistic(signal, known_alpha)
        decision = test_stat > threshold
        
        return decision, test_stat
    
    def estimate_cyclic_frequency(self, signal: np.ndarray,
                                  search_range: Optional[Tuple[float, float]] = None) -> float:
        """
        估计信号的主循环频率
        
        Args:
            signal: 输入信号
            search_range: 搜索范围 (alpha_min, alpha_max)
            
        Returns:
            estimated_alpha: 估计的循环频率
        """
        if search_range is None:
            alpha_max = self.sample_rate / 4
            search_range = (-alpha_max, alpha_max)
        
        alpha_range = np.linspace(search_range[0], search_range[1], 
                                  self.alpha_resolution * 2)
        
        max_energy = 0
        estimated_alpha = 0
        
        for alpha in alpha_range:
            if abs(alpha) < 1e-6:
                continue
            caf = self.compute_cyclic_autocorrelation(signal, alpha)
            energy = np.sum(np.abs(caf)**2)
            if energy > max_energy:
                max_energy = energy
                estimated_alpha = alpha
        
        return estimated_alpha
    
    def get_info(self) -> dict:
        """获取检测器信息"""
        return {
            'detector_type': 'Cyclostationary Detector',
            'num_samples': self.num_samples,
            'sample_rate': self.sample_rate,
            'alpha_resolution': self.alpha_resolution,
            'threshold': self.threshold
        }
