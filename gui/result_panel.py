"""
结果展示面板
显示检测结果和可视化图表
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QTableWidget, QTableWidgetItem, 
                             QTabWidget, QPushButton, QFileDialog, QSplitter,
                             QTextEdit)
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np


class ResultPanel(QWidget):
    """结果展示面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # ROC曲线标签页
        self.roc_tab = QWidget()
        roc_layout = QVBoxLayout(self.roc_tab)
        
        self.roc_figure = Figure(figsize=(8, 6), dpi=100)
        self.roc_canvas = FigureCanvas(self.roc_figure)
        self.roc_toolbar = NavigationToolbar(self.roc_canvas, self)
        
        roc_layout.addWidget(self.roc_toolbar)
        roc_layout.addWidget(self.roc_canvas)
        
        self.tab_widget.addTab(self.roc_tab, "ROC曲线")
        
        # 信号可视化标签页
        self.signal_tab = QWidget()
        signal_layout = QVBoxLayout(self.signal_tab)
        
        self.signal_figure = Figure(figsize=(8, 6), dpi=100)
        self.signal_canvas = FigureCanvas(self.signal_figure)
        self.signal_toolbar = NavigationToolbar(self.signal_canvas, self)
        
        signal_layout.addWidget(self.signal_toolbar)
        signal_layout.addWidget(self.signal_canvas)
        
        self.tab_widget.addTab(self.signal_tab, "信号波形")
        
        # Pd vs SNR标签页
        self.pd_snr_tab = QWidget()
        pd_snr_layout = QVBoxLayout(self.pd_snr_tab)
        
        self.pd_snr_figure = Figure(figsize=(8, 6), dpi=100)
        self.pd_snr_canvas = FigureCanvas(self.pd_snr_figure)
        self.pd_snr_toolbar = NavigationToolbar(self.pd_snr_canvas, self)
        
        pd_snr_layout.addWidget(self.pd_snr_toolbar)
        pd_snr_layout.addWidget(self.pd_snr_canvas)
        
        self.tab_widget.addTab(self.pd_snr_tab, "Pd-SNR曲线")
        
        # 门限分析标签页
        self.threshold_tab = QWidget()
        threshold_layout = QVBoxLayout(self.threshold_tab)
        
        self.threshold_figure = Figure(figsize=(8, 6), dpi=100)
        self.threshold_canvas = FigureCanvas(self.threshold_figure)
        self.threshold_toolbar = NavigationToolbar(self.threshold_canvas, self)
        
        threshold_layout.addWidget(self.threshold_toolbar)
        threshold_layout.addWidget(self.threshold_canvas)
        
        self.tab_widget.addTab(self.threshold_tab, "门限分析")
        
        layout.addWidget(self.tab_widget)
        
        # 结果表格
        result_group = QGroupBox("检测性能结果")
        result_layout = QVBoxLayout()
        
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(5)
        self.result_table.setHorizontalHeaderLabels(['SNR (dB)', '门限 γ', 'Pf', 'Pd', 'AUC'])
        self.result_table.horizontalHeader().setStretchLastSection(True)
        
        result_layout.addWidget(self.result_table)
        
        # 门限建议文本框
        self.recommendation_text = QTextEdit()
        self.recommendation_text.setReadOnly(True)
        self.recommendation_text.setMaximumHeight(100)
        result_layout.addWidget(QLabel("门限建议:"))
        result_layout.addWidget(self.recommendation_text)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # 导出按钮
        export_layout = QHBoxLayout()
        
        self.export_figure_btn = QPushButton("导出图片")
        self.export_figure_btn.clicked.connect(self._export_figure)
        export_layout.addWidget(self.export_figure_btn)
        
        self.export_data_btn = QPushButton("导出数据")
        self.export_data_btn.clicked.connect(self._export_data)
        export_layout.addWidget(self.export_data_btn)
        
        export_layout.addStretch()
        layout.addLayout(export_layout)
        
    def plot_roc(self, results: dict, title: str = "ROC曲线 (Pd vs Pf)"):
        """
        绘制ROC曲线
        
        Args:
            results: {snr_db: (pf, pd)} 字典
            title: 图标题
        """
        self.roc_figure.clear()
        ax = self.roc_figure.add_subplot(111)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        for i, (label, (pf, pd)) in enumerate(results.items()):
            color = colors[i % len(colors)]
            ax.semilogx(pf, pd, 'o-', color=color, label=f'SNR = {label} dB', 
                       markersize=4, linewidth=1.5)
        
        # 对角线
        ax.semilogx([1e-4, 1], [1e-4, 1], 'k--', alpha=0.5, label='随机检测')
        
        ax.set_xlabel('虚警概率 Pf', fontsize=11)
        ax.set_ylabel('检测概率 Pd', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([1e-4, 1])
        ax.set_ylim([0, 1.05])
        
        self.roc_figure.tight_layout()
        self.roc_canvas.draw()
        
        # 更新结果表格
        self._update_result_table(results)
        
    def plot_signal(self, time: np.ndarray, signal: np.ndarray, 
                    freq: np.ndarray = None, spectrum: np.ndarray = None,
                    title: str = "信号波形"):
        """
        绘制信号波形
        
        Args:
            time: 时间向量
            signal: 信号
            freq: 频率向量
            spectrum: 频谱
            title: 标题
        """
        self.signal_figure.clear()
        
        if freq is not None and spectrum is not None:
            ax1 = self.signal_figure.add_subplot(211)
            ax2 = self.signal_figure.add_subplot(212)
        else:
            ax1 = self.signal_figure.add_subplot(111)
            ax2 = None
        
        # 时域波形（只显示前500个点）
        n_display = min(500, len(time))
        ax1.plot(time[:n_display] * 1000, np.real(signal[:n_display]), 'b-', 
                linewidth=0.8, label='实部')
        if np.iscomplexobj(signal):
            ax1.plot(time[:n_display] * 1000, np.imag(signal[:n_display]), 'r-', 
                    linewidth=0.8, alpha=0.7, label='虚部')
            ax1.legend(fontsize=9)
        ax1.set_xlabel('时间 (ms)', fontsize=10)
        ax1.set_ylabel('幅度', fontsize=10)
        ax1.set_title(f'{title} - 时域', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 频域波形
        if ax2 is not None:
            ax2.plot(freq / 1000, spectrum, 'b-', linewidth=0.8)
            ax2.set_xlabel('频率 (kHz)', fontsize=10)
            ax2.set_ylabel('功率谱密度 (dB)', fontsize=10)
            ax2.set_title(f'{title} - 频域', fontsize=11)
            ax2.grid(True, alpha=0.3)
        
        self.signal_figure.tight_layout()
        self.signal_canvas.draw()
        
    def plot_pd_vs_snr(self, snr_range: np.ndarray, pd_theoretical: np.ndarray,
                       pd_simulated: np.ndarray = None, target_pf: float = 0.01):
        """
        绘制Pd vs SNR曲线
        """
        self.pd_snr_figure.clear()
        ax = self.pd_snr_figure.add_subplot(111)
        
        ax.plot(snr_range, pd_theoretical, 'b-', linewidth=2, label='理论值')
        if pd_simulated is not None:
            ax.plot(snr_range, pd_simulated, 'ro', markersize=6, label='仿真值')
        
        ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='Pd = 0.9')
        ax.axhline(y=target_pf, color='r', linestyle=':', alpha=0.7, label=f'Pf = {target_pf}')
        
        ax.set_xlabel('信噪比 SNR (dB)', fontsize=11)
        ax.set_ylabel('检测概率 Pd', fontsize=11)
        ax.set_title(f'检测概率 vs SNR (Pf = {target_pf})', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        self.pd_snr_figure.tight_layout()
        self.pd_snr_canvas.draw()
        
    def plot_threshold_analysis(self, pf_values: np.ndarray, thresholds: np.ndarray,
                                snr_range: np.ndarray, pd_values: np.ndarray,
                                target_pf: float, recommended_threshold: float):
        """
        绘制门限分析图
        """
        self.threshold_figure.clear()
        ax1 = self.threshold_figure.add_subplot(121)
        ax2 = self.threshold_figure.add_subplot(122)
        
        # Pd vs SNR
        ax1.plot(snr_range, pd_values, 'b-', linewidth=2)
        ax1.axhline(y=0.9, color='g', linestyle='--', label='Pd = 0.9')
        ax1.set_xlabel('信噪比 SNR (dB)', fontsize=10)
        ax1.set_ylabel('检测概率 Pd', fontsize=10)
        ax1.set_title(f'Pd vs SNR (Pf = {target_pf})', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # 门限 vs Pf
        ax2.semilogx(pf_values, thresholds, 'b-', linewidth=2)
        ax2.axvline(x=target_pf, color='r', linestyle='--', label=f'Pf = {target_pf}')
        ax2.axhline(y=recommended_threshold, color='g', linestyle=':', 
                   label=f'γ = {recommended_threshold:.4f}')
        ax2.set_xlabel('虚警概率 Pf', fontsize=10)
        ax2.set_ylabel('检测门限 γ', fontsize=10)
        ax2.set_title('门限 vs Pf', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        self.threshold_figure.tight_layout()
        self.threshold_canvas.draw()
        
    def set_recommendation(self, recommendation: dict):
        """设置门限建议"""
        text = f"""推荐门限: γ = {recommendation['recommended_threshold']:.6f}
归一化门限: γ/σ² = {recommendation['threshold_normalized']:.6f}
目标 Pf = {recommendation['target_pf']}, Pd = {recommendation['target_pd']}
达到目标 Pd 所需最小 SNR: {recommendation.get('min_snr_for_target_pd', 'N/A')} dB
检测样本数 N = {recommendation['num_samples']}"""
        
        if 'expected_pd_at_snr' in recommendation:
            text += f"\n在 SNR = {recommendation['expected_snr']} dB 下预期 Pd = {recommendation['expected_pd_at_snr']:.4f}"
        
        self.recommendation_text.setText(text)
        
    def _update_result_table(self, results: dict):
        """更新结果表格"""
        self.result_table.setRowCount(len(results))
        
        for row, (snr, (pf, pd)) in enumerate(results.items()):
            # 计算AUC
            sorted_indices = np.argsort(pf)
            auc = np.trapz(pd[sorted_indices], pf[sorted_indices])
            
            self.result_table.setItem(row, 0, QTableWidgetItem(f"{snr}"))
            self.result_table.setItem(row, 1, QTableWidgetItem("-"))  # 门限在其他地方设置
            self.result_table.setItem(row, 2, QTableWidgetItem(f"{pf[len(pf)//2]:.4f}"))
            self.result_table.setItem(row, 3, QTableWidgetItem(f"{pd[len(pd)//2]:.4f}"))
            self.result_table.setItem(row, 4, QTableWidgetItem(f"{auc:.4f}"))
            
    def _export_figure(self):
        """导出当前图片"""
        current_tab = self.tab_widget.currentIndex()
        figures = [self.roc_figure, self.signal_figure, self.pd_snr_figure, self.threshold_figure]
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", "", "PNG Files (*.png);;PDF Files (*.pdf)")
        
        if file_path:
            figures[current_tab].savefig(file_path, dpi=150, bbox_inches='tight')
            
    def _export_data(self):
        """导出数据"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存数据", "", "CSV Files (*.csv)")
        
        if file_path:
            # 导出表格数据
            with open(file_path, 'w', encoding='utf-8') as f:
                headers = []
                for col in range(self.result_table.columnCount()):
                    headers.append(self.result_table.horizontalHeaderItem(col).text())
                f.write(','.join(headers) + '\n')
                
                for row in range(self.result_table.rowCount()):
                    row_data = []
                    for col in range(self.result_table.columnCount()):
                        item = self.result_table.item(row, col)
                        row_data.append(item.text() if item else '')
                    f.write(','.join(row_data) + '\n')
