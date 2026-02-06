"""
单载波信号生成器
支持多种调制方式：BPSK, QPSK, 8PSK, 16QAM, 64QAM
"""

import numpy as np
from typing import Tuple, Optional


class SingleCarrierGenerator:
    """单载波信号生成器"""
    
    # 调制方式映射表
    MODULATION_SCHEMES = {
        'BPSK': 2,
        'QPSK': 4,
        '8PSK': 8,
        '16QAM': 16,
        '64QAM': 64
    }
    
    def __init__(self, modulation: str = 'QPSK', 
                 carrier_freq: float = 1000.0,
                 symbol_rate: float = 100.0,
                 samples_per_symbol: int = 8):
        """
        初始化单载波信号生成器
        
        Args:
            modulation: 调制方式 ('BPSK', 'QPSK', '8PSK', '16QAM', '64QAM')
            carrier_freq: 载波频率 (Hz)
            symbol_rate: 符号率 (symbols/s)
            samples_per_symbol: 每符号采样点数
        """
        if modulation not in self.MODULATION_SCHEMES:
            raise ValueError(f"不支持的调制方式: {modulation}")
        
        self.modulation = modulation
        self.carrier_freq = carrier_freq
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = samples_per_symbol
        self.sample_rate = symbol_rate * samples_per_symbol
        
    def _generate_constellation(self) -> np.ndarray:
        """生成星座图映射"""
        M = self.MODULATION_SCHEMES[self.modulation]
        
        if self.modulation == 'BPSK':
            return np.array([-1+0j, 1+0j])
        
        elif self.modulation == 'QPSK':
            return np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        
        elif self.modulation == '8PSK':
            angles = np.arange(8) * np.pi / 4
            return np.exp(1j * angles)
        
        elif self.modulation == '16QAM':
            levels = np.array([-3, -1, 1, 3])
            constellation = []
            for i in levels:
                for q in levels:
                    constellation.append(i + 1j*q)
            return np.array(constellation) / np.sqrt(10)
        
        elif self.modulation == '64QAM':
            levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            constellation = []
            for i in levels:
                for q in levels:
                    constellation.append(i + 1j*q)
            return np.array(constellation) / np.sqrt(42)
        
        return np.array([])
    
    def generate(self, num_symbols: int, 
                 normalize_power: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成单载波调制信号
        
        Args:
            num_symbols: 符号数量
            normalize_power: 是否归一化功率
            
        Returns:
            time: 时间向量
            signal: 复基带信号
        """
        # 生成随机符号
        M = self.MODULATION_SCHEMES[self.modulation]
        constellation = self._generate_constellation()
        symbol_indices = np.random.randint(0, M, num_symbols)
        symbols = constellation[symbol_indices]
        
        # 上采样和脉冲成形（简单重复）
        signal = np.repeat(symbols, self.samples_per_symbol)
        
        # 生成时间向量
        num_samples = len(signal)
        time = np.arange(num_samples) / self.sample_rate
        
        # 功率归一化
        if normalize_power:
            signal = signal / np.sqrt(np.mean(np.abs(signal)**2))
        
        return time, signal
    
    def generate_passband(self, num_symbols: int,
                          normalize_power: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成带通信号（实信号）
        
        Args:
            num_symbols: 符号数量
            normalize_power: 是否归一化功率
            
        Returns:
            time: 时间向量
            signal: 实带通信号
        """
        time, baseband = self.generate(num_symbols, normalize_power)
        
        # 上变频到载波频率
        carrier = np.exp(1j * 2 * np.pi * self.carrier_freq * time)
        passband = np.real(baseband * carrier)
        
        return time, passband
    
    def get_spectrum(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算信号频谱
        
        Args:
            signal: 输入信号
            
        Returns:
            freq: 频率向量
            spectrum: 功率谱密度 (dB)
        """
        n = len(signal)
        spectrum = np.fft.fftshift(np.fft.fft(signal))
        freq = np.fft.fftshift(np.fft.fftfreq(n, 1/self.sample_rate))
        psd = 10 * np.log10(np.abs(spectrum)**2 / n + 1e-10)
        
        return freq, psd
    
    def get_info(self) -> dict:
        """获取生成器信息"""
        return {
            'modulation': self.modulation,
            'carrier_freq': self.carrier_freq,
            'symbol_rate': self.symbol_rate,
            'sample_rate': self.sample_rate,
            'bits_per_symbol': int(np.log2(self.MODULATION_SCHEMES[self.modulation]))
        }
