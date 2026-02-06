"""
门限校准器
实现自动门限校准功能
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional, Dict
import matplotlib.pyplot as plt


class ThresholdCalibrator:
    """门限校准器类"""
    
    def __init__(self, num_samples: int = 1000,
                 noise_power: float = 1.0):
        """
        初始化门限校准器
        
        Args:
            num_samples: 检测样本数
            noise_power: 噪声功率
        """
        self.num_samples = num_samples
        self.noise_power = noise_power
        
    def calibrate_energy_threshold(self, target_pf: float,
                                   method: str = 'theoretical') -> float:
        """
        校准能量检测门限
        
        给定目标虚警概率Pf，计算检测门限gamma
        
        Args:
            target_pf: 目标虚警概率
            method: 校准方法 ('theoretical', 'simulation')
            
        Returns:
            threshold: 检测门限
        """
        N = self.num_samples
        sigma2 = self.noise_power
        
        if method == 'theoretical':
            # 理论方法
            # 在H0下，检测统计量 T = (1/N)*sum(|y|^2)
            # 对于复数AWGN: T ~ Gamma(N, sigma^2/N)
            shape = N
            scale = sigma2 / N
            
            # P(T > gamma | H0) = Pf
            # gamma = quantile(1 - Pf)
            threshold = stats.gamma.ppf(1 - target_pf, shape, scale=scale)
            
        elif method == 'simulation':
            # Monte Carlo仿真方法
            num_trials = 10000
            test_stats = []
            
            for _ in range(num_trials):
                # 生成H0假设下的噪声
                noise = np.sqrt(sigma2 / 2) * (
                    np.random.randn(N) + 1j * np.random.randn(N)
                )
                test_stat = np.mean(np.abs(noise)**2)
                test_stats.append(test_stat)
            
            # 排序并找到分位数
            test_stats = np.sort(test_stats)[::-1]
            idx = int(target_pf * num_trials)
            threshold = test_stats[min(idx, num_trials - 1)]
            
        else:
            raise ValueError(f"不支持的校准方法: {method}")
        
        return threshold
    
    def get_threshold_table(self, pf_values: np.ndarray = None,
                            method: str = 'theoretical') -> Dict[float, float]:
        """
        生成Pf-门限对照表
        
        Args:
            pf_values: Pf值数组
            method: 校准方法
            
        Returns:
            table: {pf: threshold} 字典
        """
        if pf_values is None:
            pf_values = np.array([0.1, 0.05, 0.01, 0.005, 0.001])
        
        table = {}
        for pf in pf_values:
            threshold = self.calibrate_energy_threshold(pf, method)
            table[pf] = threshold
        
        return table
    
    def recommend_threshold(self, target_pf: float = 0.01,
                           target_pd: float = 0.9,
                           snr_db: Optional[float] = None) -> Dict:
        """
        推荐检测门限
        
        Args:
            target_pf: 目标虚警概率
            target_pd: 目标检测概率
            snr_db: 预期信噪比（可选）
            
        Returns:
            recommendation: 包含门限建议的字典
        """
        N = self.num_samples
        sigma2 = self.noise_power
        
        # 根据目标Pf计算门限
        threshold_pf = self.calibrate_energy_threshold(target_pf)
        
        # 计算该门限下不同SNR的Pd
        snr_analysis = {}
        snr_range = np.arange(-10, 20, 2)
        
        for snr in snr_range:
            snr_linear = 10 ** (snr / 10)
            signal_plus_noise_power = sigma2 * (1 + snr_linear)
            
            shape = N
            scale = signal_plus_noise_power / N
            pd = 1 - stats.gamma.cdf(threshold_pf, shape, scale=scale)
            snr_analysis[snr] = pd
        
        # 找到达到目标Pd所需的最小SNR
        min_snr_required = None
        for snr, pd in sorted(snr_analysis.items()):
            if pd >= target_pd:
                min_snr_required = snr
                break
        
        recommendation = {
            'target_pf': target_pf,
            'target_pd': target_pd,
            'recommended_threshold': threshold_pf,
            'threshold_normalized': threshold_pf / sigma2,
            'min_snr_for_target_pd': min_snr_required,
            'snr_pd_analysis': snr_analysis,
            'num_samples': N,
            'noise_power': sigma2
        }
        
        if snr_db is not None:
            # 计算给定SNR下的实际Pd
            snr_linear = 10 ** (snr_db / 10)
            signal_plus_noise_power = sigma2 * (1 + snr_linear)
            shape = N
            scale = signal_plus_noise_power / N
            actual_pd = 1 - stats.gamma.cdf(threshold_pf, shape, scale=scale)
            recommendation['expected_snr'] = snr_db
            recommendation['expected_pd_at_snr'] = actual_pd
        
        return recommendation
    
    def compute_required_samples(self, target_pf: float,
                                 target_pd: float,
                                 snr_db: float) -> int:
        """
        计算达到目标性能所需的样本数
        
        Args:
            target_pf: 目标虚警概率
            target_pd: 目标检测概率
            snr_db: 信噪比 (dB)
            
        Returns:
            required_samples: 所需样本数
        """
        snr_linear = 10 ** (snr_db / 10)
        sigma2 = self.noise_power
        
        # 二分搜索找到所需样本数
        n_min, n_max = 10, 100000
        
        while n_max - n_min > 10:
            n_mid = (n_min + n_max) // 2
            
            # 计算门限
            shape = n_mid
            scale = sigma2 / n_mid
            threshold = stats.gamma.ppf(1 - target_pf, shape, scale=scale)
            
            # 计算Pd
            signal_plus_noise_power = sigma2 * (1 + snr_linear)
            scale_h1 = signal_plus_noise_power / n_mid
            pd = 1 - stats.gamma.cdf(threshold, shape, scale=scale_h1)
            
            if pd >= target_pd:
                n_max = n_mid
            else:
                n_min = n_mid
        
        return n_max
    
    def plot_threshold_analysis(self, snr_range: np.ndarray = None,
                                target_pf: float = 0.01,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制门限分析图
        
        Args:
            snr_range: SNR范围
            target_pf: 目标虚警概率
            save_path: 保存路径
            
        Returns:
            fig: matplotlib图形对象
        """
        if snr_range is None:
            snr_range = np.arange(-15, 20, 1)
        
        N = self.num_samples
        sigma2 = self.noise_power
        
        # 计算门限
        threshold = self.calibrate_energy_threshold(target_pf)
        
        # 计算不同SNR下的Pd
        pd_values = []
        for snr_db in snr_range:
            snr_linear = 10 ** (snr_db / 10)
            signal_plus_noise_power = sigma2 * (1 + snr_linear)
            shape = N
            scale = signal_plus_noise_power / N
            pd = 1 - stats.gamma.cdf(threshold, shape, scale=scale)
            pd_values.append(pd)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 子图1: Pd vs SNR
        ax1.plot(snr_range, pd_values, 'b-', linewidth=2)
        ax1.axhline(y=0.9, color='g', linestyle='--', label='Pd = 0.9')
        ax1.axhline(y=target_pf, color='r', linestyle=':', label=f'Pf = {target_pf}')
        ax1.set_xlabel('信噪比 SNR (dB)', fontsize=12)
        ax1.set_ylabel('检测概率 Pd', fontsize=12)
        ax1.set_title(f'检测概率 vs SNR (N={N}, Pf={target_pf})', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # 子图2: 不同Pf下的门限
        pf_values = np.logspace(-4, -0.5, 50)
        thresholds = [self.calibrate_energy_threshold(pf) for pf in pf_values]
        
        ax2.semilogx(pf_values, thresholds, 'b-', linewidth=2)
        ax2.axvline(x=target_pf, color='r', linestyle='--', label=f'Pf = {target_pf}')
        ax2.axhline(y=threshold, color='g', linestyle=':', label=f'γ = {threshold:.4f}')
        ax2.set_xlabel('虚警概率 Pf', fontsize=12)
        ax2.set_ylabel('检测门限 γ', fontsize=12)
        ax2.set_title(f'门限 vs Pf (N={N})', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def print_recommendation(self, recommendation: Dict):
        """打印门限建议"""
        print("=" * 50)
        print("  检测门限建议")
        print("=" * 50)
        print(f"  目标虚警概率 Pf: {recommendation['target_pf']}")
        print(f"  目标检测概率 Pd: {recommendation['target_pd']}")
        print(f"  检测样本数 N: {recommendation['num_samples']}")
        print(f"  噪声功率: {recommendation['noise_power']}")
        print("-" * 50)
        print(f"  推荐门限 γ: {recommendation['recommended_threshold']:.6f}")
        print(f"  归一化门限 (γ/σ²): {recommendation['threshold_normalized']:.6f}")
        
        if recommendation['min_snr_for_target_pd'] is not None:
            print(f"  达到 Pd={recommendation['target_pd']} 所需最小 SNR: "
                  f"{recommendation['min_snr_for_target_pd']} dB")
        
        if 'expected_pd_at_snr' in recommendation:
            print(f"  在 SNR={recommendation['expected_snr']} dB 下的预期 Pd: "
                  f"{recommendation['expected_pd_at_snr']:.4f}")
        
        print("=" * 50)
