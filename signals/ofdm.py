"""
OFDM信号生成器
支持可配置的子载波数、CP长度和调制方式
"""

import numpy as np
from typing import Tuple, Optional


class OFDMGenerator:
    """OFDM信号生成器"""
    
    # 调制方式映射表
    MODULATION_SCHEMES = {
        'BPSK': 2,
        'QPSK': 4,
        '16QAM': 16,
        '64QAM': 64
    }
    
    def __init__(self, 
                 num_subcarriers: int = 64,
                 cp_length: int = 16,
                 modulation: str = 'QPSK',
                 num_data_subcarriers: Optional[int] = None,
                 sample_rate: float = 1e6):
        """
        初始化OFDM信号生成器
        
        Args:
            num_subcarriers: 总子载波数 (FFT大小)
            cp_length: 循环前缀长度
            modulation: 子载波调制方式
            num_data_subcarriers: 数据子载波数 (默认为总数-导频-保护带)
            sample_rate: 采样率 (Hz)
        """
        if modulation not in self.MODULATION_SCHEMES:
            raise ValueError(f"不支持的调制方式: {modulation}")
        
        self.num_subcarriers = num_subcarriers
        self.cp_length = cp_length
        self.modulation = modulation
        self.sample_rate = sample_rate
        
        # 默认数据子载波数（排除DC和保护带）
        if num_data_subcarriers is None:
            self.num_data_subcarriers = num_subcarriers - 12  # 简化设计
        else:
            self.num_data_subcarriers = num_data_subcarriers
        
        # 子载波分配（简化模型）
        self._setup_subcarrier_allocation()
    
    def _setup_subcarrier_allocation(self):
        """设置子载波分配"""
        N = self.num_subcarriers
        
        # 数据子载波索引（排除DC和边缘）
        guard_band = (N - self.num_data_subcarriers) // 2
        self.data_indices = np.arange(guard_band, N - guard_band)
        self.data_indices = self.data_indices[self.data_indices != N // 2]  # 排除DC
        
        # 导频子载波（可选，简化设计）
        self.pilot_indices = np.array([])
        
    def _generate_constellation(self) -> np.ndarray:
        """生成星座图映射"""
        M = self.MODULATION_SCHEMES[self.modulation]
        
        if self.modulation == 'BPSK':
            return np.array([-1+0j, 1+0j])
        
        elif self.modulation == 'QPSK':
            return np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        
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
        生成OFDM时域信号
        
        Args:
            num_symbols: OFDM符号数量
            normalize_power: 是否归一化功率
            
        Returns:
            time: 时间向量
            signal: OFDM时域信号
        """
        M = self.MODULATION_SCHEMES[self.modulation]
        constellation = self._generate_constellation()
        
        # 每个OFDM符号的时域长度（含CP）
        symbol_length = self.num_subcarriers + self.cp_length
        total_samples = num_symbols * symbol_length
        
        # 生成所有OFDM符号
        signal = np.zeros(total_samples, dtype=complex)
        
        for sym_idx in range(num_symbols):
            # 频域符号（初始化为零）
            freq_domain = np.zeros(self.num_subcarriers, dtype=complex)
            
            # 生成数据符号
            num_data = len(self.data_indices)
            data_symbols = constellation[np.random.randint(0, M, num_data)]
            freq_domain[self.data_indices] = data_symbols
            
            # IFFT
            time_domain = np.fft.ifft(freq_domain) * np.sqrt(self.num_subcarriers)
            
            # 添加循环前缀
            ofdm_symbol = np.concatenate([
                time_domain[-self.cp_length:],  # CP
                time_domain
            ])
            
            # 放入输出信号
            start_idx = sym_idx * symbol_length
            signal[start_idx:start_idx + symbol_length] = ofdm_symbol
        
        # 生成时间向量
        time = np.arange(total_samples) / self.sample_rate
        
        # 功率归一化
        if normalize_power:
            signal = signal / np.sqrt(np.mean(np.abs(signal)**2))
        
        return time, signal
    
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
            'num_subcarriers': self.num_subcarriers,
            'cp_length': self.cp_length,
            'modulation': self.modulation,
            'num_data_subcarriers': len(self.data_indices),
            'sample_rate': self.sample_rate,
            'symbol_duration': (self.num_subcarriers + self.cp_length) / self.sample_rate,
            'bits_per_symbol': int(np.log2(self.MODULATION_SCHEMES[self.modulation])) * len(self.data_indices)
        }
