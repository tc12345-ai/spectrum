"""
主窗口
频谱检测与信号识别仿真软件主界面
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QPushButton, QMenuBar, QMenu, QAction,
                             QStatusBar, QMessageBox, QProgressBar, QApplication,
                             QToolBar, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

from .signal_panel import SignalPanel
from .detector_panel import DetectorPanel
from .result_panel import ResultPanel


class SimulationThread(QThread):
    """仿真线程"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, signal_params, detector_params):
        super().__init__()
        self.signal_params = signal_params
        self.detector_params = detector_params
        
    def run(self):
        try:
            from signals import SingleCarrierGenerator, OFDMGenerator
            from detectors import EnergyDetector, CyclostationaryDetector
            from analysis import ROCGenerator, ThresholdCalibrator
            from utils import AWGNChannel
            
            results = {}
            
            # 创建信号生成器
            if self.signal_params['signal_type'] == 'single_carrier':
                generator = SingleCarrierGenerator(
                    modulation=self.signal_params['modulation'],
                    symbol_rate=self.signal_params['symbol_rate']
                )
                num_symbols = self.signal_params['num_symbols']
                
                def signal_gen(n):
                    _, sig = generator.generate(max(n // 8, 100))
                    return sig[:n] if len(sig) >= n else np.tile(sig, n // len(sig) + 1)[:n]
            else:
                generator = OFDMGenerator(
                    num_subcarriers=self.signal_params['num_subcarriers'],
                    cp_length=self.signal_params['cp_length'],
                    modulation=self.signal_params['modulation']
                )
                num_symbols = self.signal_params['num_symbols']
                
                def signal_gen(n):
                    _, sig = generator.generate(max(n // 80, 10))
                    return sig[:n] if len(sig) >= n else np.tile(sig, n // len(sig) + 1)[:n]
            
            self.progress.emit(10)
            
            # 生成示例信号
            _, example_signal = generator.generate(num_symbols)
            freq, spectrum = generator.get_spectrum(example_signal)
            time = np.arange(len(example_signal)) / (self.signal_params.get('symbol_rate', 1e6) * 8)
            results['signal'] = (time, example_signal, freq, spectrum)
            
            self.progress.emit(20)
            
            # 创建检测器
            if self.detector_params['detector_type'] == 'energy':
                detector = EnergyDetector(
                    num_samples=self.detector_params['num_samples'],
                    noise_power=self.detector_params['noise_power']
                )
            else:
                detector = CyclostationaryDetector(
                    num_samples=self.detector_params['num_samples'],
                    alpha_resolution=self.detector_params.get('alpha_resolution', 64)
                )
            
            # 门限校准
            if self.detector_params['auto_threshold']:
                target_pf = self.detector_params['target_pf']
                method = self.detector_params['calibration_method']
                threshold = detector.calibrate_threshold(target_pf, method=method)
            else:
                threshold = self.detector_params['manual_threshold']
                detector.threshold = threshold
            
            results['threshold'] = threshold
            self.progress.emit(30)
            
            # 生成ROC曲线
            snr_min = self.signal_params['snr_min']
            snr_max = self.signal_params['snr_max']
            snr_step = self.signal_params['snr_step']
            snr_list = np.arange(snr_min, snr_max + snr_step, snr_step).tolist()
            
            if self.detector_params['detector_type'] == 'energy':
                roc_gen = ROCGenerator(detector, num_trials=self.detector_params['num_trials'])
                
                if self.detector_params['use_theoretical']:
                    roc_results = roc_gen.generate_multi_snr_roc(
                        signal_gen, snr_list, use_theoretical=True)
                else:
                    roc_results = roc_gen.generate_multi_snr_roc(
                        signal_gen, snr_list, use_theoretical=False)
                
                results['roc'] = roc_results
            
            self.progress.emit(70)
            
            # Pd vs SNR 分析
            snr_range = np.arange(snr_min, snr_max + 1, 1)
            pd_theoretical = []
            
            if self.detector_params['detector_type'] == 'energy':
                calibrator = ThresholdCalibrator(
                    num_samples=self.detector_params['num_samples'],
                    noise_power=self.detector_params['noise_power']
                )
                
                for snr in snr_range:
                    pd = detector.compute_theoretical_pd(snr, threshold)
                    pd_theoretical.append(pd)
                
                results['pd_vs_snr'] = (snr_range, np.array(pd_theoretical))
                
                # 门限分析
                pf_values = np.logspace(-4, -0.5, 50)
                thresholds = [calibrator.calibrate_energy_threshold(pf) for pf in pf_values]
                results['threshold_analysis'] = {
                    'pf_values': pf_values,
                    'thresholds': np.array(thresholds),
                    'snr_range': snr_range,
                    'pd_values': np.array(pd_theoretical),
                    'target_pf': self.detector_params['target_pf'],
                    'recommended_threshold': threshold
                }
                
                # 门限建议
                recommendation = calibrator.recommend_threshold(
                    target_pf=self.detector_params['target_pf'],
                    target_pd=0.9,
                    snr_db=0
                )
                results['recommendation'] = recommendation
            
            self.progress.emit(100)
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self):
        super().__init__()
        self._init_ui()
        self.simulation_thread = None
        
    def _init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("频谱检测与信号识别仿真软件")
        self.setMinimumSize(1200, 800)
        
        # 设置字体
        font = QFont("Microsoft YaHei", 9)
        self.setFont(font)
        
        # 创建菜单栏
        self._create_menu_bar()
        
        # 创建工具栏
        self._create_tool_bar()
        
        # 创建状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("就绪")
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主分割器
        main_splitter = QSplitter(Qt.Horizontal)
        
        # 左侧配置面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        # 信号配置
        signal_group = QGroupBox("信号配置")
        signal_layout = QVBoxLayout()
        self.signal_panel = SignalPanel()
        signal_layout.addWidget(self.signal_panel)
        signal_group.setLayout(signal_layout)
        left_layout.addWidget(signal_group)
        
        # 检测器配置
        detector_group = QGroupBox("检测器配置")
        detector_layout = QVBoxLayout()
        self.detector_panel = DetectorPanel()
        detector_layout.addWidget(self.detector_panel)
        detector_group.setLayout(detector_layout)
        left_layout.addWidget(detector_group)
        
        # 运行按钮
        self.run_button = QPushButton("▶ 运行仿真")
        self.run_button.setMinimumHeight(40)
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.run_button.clicked.connect(self._run_simulation)
        left_layout.addWidget(self.run_button)
        
        left_panel.setMaximumWidth(350)
        
        # 右侧结果面板
        self.result_panel = ResultPanel()
        
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(self.result_panel)
        main_splitter.setSizes([350, 850])
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.addWidget(main_splitter)
        
    def _create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        export_action = QAction("导出结果...", self)
        export_action.setShortcut("Ctrl+E")
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 仿真菜单
        sim_menu = menubar.addMenu("仿真(&S)")
        
        run_action = QAction("运行", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self._run_simulation)
        sim_menu.addAction(run_action)
        
        stop_action = QAction("停止", self)
        stop_action.setShortcut("Shift+F5")
        sim_menu.addAction(stop_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
    def _create_tool_bar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        run_action = QAction("运行", self)
        run_action.triggered.connect(self._run_simulation)
        toolbar.addAction(run_action)
        
        toolbar.addSeparator()
        
        export_action = QAction("导出", self)
        toolbar.addAction(export_action)
        
    def _run_simulation(self):
        """运行仿真"""
        if self.simulation_thread and self.simulation_thread.isRunning():
            QMessageBox.warning(self, "警告", "仿真正在运行中...")
            return
        
        # 获取参数
        signal_params = self.signal_panel.get_parameters()
        detector_params = self.detector_panel.get_parameters()
        
        # 更新UI状态
        self.run_button.setEnabled(False)
        self.run_button.setText("仿真中...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_bar.showMessage("正在运行仿真...")
        
        # 创建并启动仿真线程
        self.simulation_thread = SimulationThread(signal_params, detector_params)
        self.simulation_thread.progress.connect(self._on_progress)
        self.simulation_thread.finished.connect(self._on_simulation_finished)
        self.simulation_thread.error.connect(self._on_simulation_error)
        self.simulation_thread.start()
        
    def _on_progress(self, value):
        """仿真进度更新"""
        self.progress_bar.setValue(value)
        
    def _on_simulation_finished(self, results):
        """仿真完成"""
        self.run_button.setEnabled(True)
        self.run_button.setText("▶ 运行仿真")
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("仿真完成")
        
        # 显示结果
        if 'roc' in results:
            self.result_panel.plot_roc(results['roc'])
        
        if 'signal' in results:
            time, signal, freq, spectrum = results['signal']
            self.result_panel.plot_signal(time, signal, freq, spectrum)
        
        if 'pd_vs_snr' in results:
            snr_range, pd_values = results['pd_vs_snr']
            self.result_panel.plot_pd_vs_snr(snr_range, pd_values)
        
        if 'threshold_analysis' in results:
            ta = results['threshold_analysis']
            self.result_panel.plot_threshold_analysis(
                ta['pf_values'], ta['thresholds'],
                ta['snr_range'], ta['pd_values'],
                ta['target_pf'], ta['recommended_threshold']
            )
        
        if 'recommendation' in results:
            self.result_panel.set_recommendation(results['recommendation'])
        
    def _on_simulation_error(self, error_msg):
        """仿真错误"""
        self.run_button.setEnabled(True)
        self.run_button.setText("▶ 运行仿真")
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("仿真出错")
        
        QMessageBox.critical(self, "仿真错误", f"仿真过程中发生错误:\n{error_msg}")
        
    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于",
            """<h3>频谱检测与信号识别仿真软件</h3>
            <p>版本: 1.0.0</p>
            <p>用于认知无线电和频谱感知的仿真工具</p>
            <p><b>功能特点:</b></p>
            <ul>
                <li>支持单载波和OFDM信号生成</li>
                <li>能量检测和循环平稳检测</li>
                <li>自动门限校准</li>
                <li>ROC曲线生成</li>
            </ul>""")


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
