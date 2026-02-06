"""
ROC曲线生成器
生成检测器的接收机工作特性曲线
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
import matplotlib.pyplot as plt


class ROCGenerator:
    """ROC曲线生成器类"""
    
    def __init__(self, detector, num_trials: int = 1000):
        """
        初始化ROC生成器
        
        Args:
            detector: 检测器对象（需要有detect方法）
            num_trials: 每个点的Monte Carlo仿真次数
        """
        self.detector = detector
        self.num_trials = num_trials
        
    def generate_theoretical_roc(self, snr_db: float,
                                 pf_range: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成理论ROC曲线（仅适用于能量检测器）
        
        Args:
            snr_db: 信噪比 (dB)
            pf_range: 虚警概率范围
            
        Returns:
            pf: 虚警概率数组
            pd: 检测概率数组
        """
        if pf_range is None:
            pf_range = np.logspace(-4, 0, 100)
        
        pd_list = []
        
        for pf in pf_range:
            # 校准门限
            threshold = self.detector.calibrate_threshold(pf, method='theoretical')
            # 计算理论Pd
            pd = self.detector.compute_theoretical_pd(snr_db, threshold)
            pd_list.append(pd)
        
        return pf_range, np.array(pd_list)
    
    def generate_monte_carlo_roc(self, signal_generator: Callable,
                                 snr_db: float,
                                 pf_range: Optional[np.ndarray] = None,
                                 noise_power: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        通过Monte Carlo仿真生成ROC曲线
        
        Args:
            signal_generator: 信号生成函数 (num_samples) -> signal
            snr_db: 信噪比 (dB)
            pf_range: 虚警概率范围
            noise_power: 噪声功率
            
        Returns:
            pf_actual: 实际虚警概率数组
            pd_actual: 实际检测概率数组
        """
        if pf_range is None:
            pf_range = np.logspace(-3, 0, 20)
        
        N = self.detector.num_samples
        snr_linear = 10 ** (snr_db / 10)
        
        pf_actual = []
        pd_actual = []
        
        for target_pf in pf_range:
            # 校准门限
            threshold = self.detector.calibrate_threshold(target_pf, method='theoretical')
            
            false_alarms = 0
            detections = 0
            
            for _ in range(self.num_trials):
                # H0: 仅噪声
                noise_h0 = np.sqrt(noise_power / 2) * (
                    np.random.randn(N) + 1j * np.random.randn(N)
                )
                decision_h0, _ = self.detector.detect(noise_h0, threshold)
                if decision_h0:
                    false_alarms += 1
                
                # H1: 信号+噪声
                signal = signal_generator(N)
                signal = signal / np.sqrt(np.mean(np.abs(signal)**2))  # 归一化
                signal_power = noise_power * snr_linear
                signal = signal * np.sqrt(signal_power)
                
                noise_h1 = np.sqrt(noise_power / 2) * (
                    np.random.randn(N) + 1j * np.random.randn(N)
                )
                received = signal + noise_h1
                
                decision_h1, _ = self.detector.detect(received, threshold)
                if decision_h1:
                    detections += 1
            
            pf_actual.append(false_alarms / self.num_trials)
            pd_actual.append(detections / self.num_trials)
        
        return np.array(pf_actual), np.array(pd_actual)
    
    def generate_multi_snr_roc(self, signal_generator: Callable,
                               snr_db_list: List[float],
                               pf_range: Optional[np.ndarray] = None,
                               use_theoretical: bool = True) -> dict:
        """
        生成多SNR条件下的ROC曲线
        
        Args:
            signal_generator: 信号生成函数
            snr_db_list: SNR列表
            pf_range: 虚警概率范围
            use_theoretical: 是否使用理论方法
            
        Returns:
            results: {snr_db: (pf, pd)} 字典
        """
        results = {}
        
        for snr_db in snr_db_list:
            if use_theoretical:
                pf, pd = self.generate_theoretical_roc(snr_db, pf_range)
            else:
                pf, pd = self.generate_monte_carlo_roc(signal_generator, snr_db, pf_range)
            results[snr_db] = (pf, pd)
        
        return results
    
    def compute_auc(self, pf: np.ndarray, pd: np.ndarray) -> float:
        """
        计算ROC曲线下面积 (AUC)
        
        Args:
            pf: 虚警概率数组
            pd: 检测概率数组
            
        Returns:
            auc: 曲线下面积
        """
        # 确保从小到大排序
        sorted_indices = np.argsort(pf)
        pf_sorted = pf[sorted_indices]
        pd_sorted = pd[sorted_indices]
        
        # 梯形积分
        auc = np.trapz(pd_sorted, pf_sorted)
        
        return auc
    
    def plot_roc(self, results: dict,
                 title: str = "ROC曲线 (Pd vs Pf)",
                 save_path: Optional[str] = None,
                 show_auc: bool = True) -> plt.Figure:
        """
        绘制ROC曲线
        
        Args:
            results: {snr_db: (pf, pd)} 或 {label: (pf, pd)}
            title: 图标题
            save_path: 保存路径
            show_auc: 是否显示AUC
            
        Returns:
            fig: matplotlib图形对象
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
        
        for (label, (pf, pd)), color in zip(results.items(), colors):
            auc = self.compute_auc(pf, pd)
            if show_auc:
                legend_label = f"SNR = {label} dB (AUC = {auc:.4f})"
            else:
                legend_label = f"SNR = {label} dB"
            ax.semilogx(pf, pd, 'o-', color=color, label=legend_label, markersize=3)
        
        # 对角线（随机检测器）
        ax.semilogx([1e-4, 1], [1e-4, 1], 'k--', alpha=0.5, label='随机检测')
        
        ax.set_xlabel('虚警概率 Pf', fontsize=12)
        ax.set_ylabel('检测概率 Pd', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1e-4, 1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_pd_vs_snr(self, signal_generator: Callable,
                       snr_range: np.ndarray,
                       target_pf: float = 0.01,
                       title: str = "检测概率 vs SNR",
                       save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制Pd vs SNR曲线
        
        Args:
            signal_generator: 信号生成函数
            snr_range: SNR范围 (dB)
            target_pf: 目标虚警概率
            title: 图标题
            save_path: 保存路径
            
        Returns:
            fig: matplotlib图形对象
        """
        pd_theoretical = []
        pd_simulated = []
        
        for snr_db in snr_range:
            # 理论Pd
            threshold = self.detector.calibrate_threshold(target_pf, method='theoretical')
            pd_theo = self.detector.compute_theoretical_pd(snr_db, threshold)
            pd_theoretical.append(pd_theo)
            
            # 仿真Pd
            N = self.detector.num_samples
            noise_power = self.detector.noise_power
            snr_linear = 10 ** (snr_db / 10)
            
            detections = 0
            for _ in range(self.num_trials // 2):  # 减少仿真次数
                signal = signal_generator(N)
                signal = signal / np.sqrt(np.mean(np.abs(signal)**2))
                signal_power = noise_power * snr_linear
                signal = signal * np.sqrt(signal_power)
                
                noise = np.sqrt(noise_power / 2) * (
                    np.random.randn(N) + 1j * np.random.randn(N)
                )
                received = signal + noise
                
                decision, _ = self.detector.detect(received, threshold)
                if decision:
                    detections += 1
            
            pd_simulated.append(detections / (self.num_trials // 2))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(snr_range, pd_theoretical, 'b-', linewidth=2, label='理论值')
        ax.plot(snr_range, pd_simulated, 'ro', markersize=6, label='仿真值')
        
        ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='Pd = 0.9')
        ax.axhline(y=target_pf, color='r', linestyle=':', alpha=0.7, label=f'Pf = {target_pf}')
        
        ax.set_xlabel('信噪比 SNR (dB)', fontsize=12)
        ax.set_ylabel('检测概率 Pd', fontsize=12)
        ax.set_title(f'{title} (Pf = {target_pf})', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
