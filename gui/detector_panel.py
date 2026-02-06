"""
检测器配置面板
配置检测器参数
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                             QFormLayout, QCheckBox, QPushButton)
from PyQt5.QtCore import pyqtSignal


class DetectorPanel(QWidget):
    """检测器配置面板"""
    
    # 参数变化信号
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 检测器选择
        detector_group = QGroupBox("检测器选择")
        detector_layout = QFormLayout()
        
        self.detector_type = QComboBox()
        self.detector_type.addItems(['能量检测', '循环平稳检测'])
        detector_layout.addRow("检测器类型:", self.detector_type)
        
        detector_group.setLayout(detector_layout)
        layout.addWidget(detector_group)
        
        # 能量检测参数
        ed_group = QGroupBox("能量检测参数")
        ed_layout = QFormLayout()
        
        self.num_samples = QSpinBox()
        self.num_samples.setRange(10, 100000)
        self.num_samples.setValue(1000)
        ed_layout.addRow("检测样本数 N:", self.num_samples)
        
        self.noise_power = QDoubleSpinBox()
        self.noise_power.setRange(0.001, 100)
        self.noise_power.setValue(1.0)
        self.noise_power.setDecimals(4)
        ed_layout.addRow("噪声功率 σ²:", self.noise_power)
        
        ed_group.setLayout(ed_layout)
        layout.addWidget(ed_group)
        
        # 门限设置
        threshold_group = QGroupBox("门限设置")
        threshold_layout = QFormLayout()
        
        self.auto_threshold = QCheckBox("自动门限校准")
        self.auto_threshold.setChecked(True)
        threshold_layout.addRow("", self.auto_threshold)
        
        self.target_pf = QDoubleSpinBox()
        self.target_pf.setRange(0.0001, 0.5)
        self.target_pf.setValue(0.01)
        self.target_pf.setDecimals(4)
        self.target_pf.setSingleStep(0.001)
        threshold_layout.addRow("目标虚警概率 Pf:", self.target_pf)
        
        self.manual_threshold = QDoubleSpinBox()
        self.manual_threshold.setRange(0, 1000)
        self.manual_threshold.setValue(1.0)
        self.manual_threshold.setDecimals(6)
        self.manual_threshold.setEnabled(False)
        threshold_layout.addRow("手动门限:", self.manual_threshold)
        
        self.calibration_method = QComboBox()
        self.calibration_method.addItems(['理论方法', '仿真方法'])
        threshold_layout.addRow("校准方法:", self.calibration_method)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # 仿真参数
        sim_group = QGroupBox("仿真参数")
        sim_layout = QFormLayout()
        
        self.num_trials = QSpinBox()
        self.num_trials.setRange(100, 100000)
        self.num_trials.setValue(1000)
        sim_layout.addRow("Monte Carlo次数:", self.num_trials)
        
        self.use_theoretical = QCheckBox("使用理论ROC")
        self.use_theoretical.setChecked(True)
        sim_layout.addRow("", self.use_theoretical)
        
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)
        
        # 循环平稳参数（初始隐藏）
        self.cs_group = QGroupBox("循环平稳参数")
        cs_layout = QFormLayout()
        
        self.alpha_resolution = QSpinBox()
        self.alpha_resolution.setRange(16, 256)
        self.alpha_resolution.setValue(64)
        cs_layout.addRow("循环频率分辨率:", self.alpha_resolution)
        
        self.cs_group.setLayout(cs_layout)
        layout.addWidget(self.cs_group)
        self.cs_group.setVisible(False)
        
        layout.addStretch()
        
        # 连接信号
        self.auto_threshold.toggled.connect(self._on_auto_threshold_changed)
        self.detector_type.currentIndexChanged.connect(self._on_detector_changed)
        
    def _on_auto_threshold_changed(self, checked):
        """自动门限模式切换"""
        self.target_pf.setEnabled(checked)
        self.manual_threshold.setEnabled(not checked)
        self.calibration_method.setEnabled(checked)
        
    def _on_detector_changed(self, index):
        """检测器类型切换"""
        is_cs = (index == 1)
        self.cs_group.setVisible(is_cs)
        
    def get_parameters(self) -> dict:
        """获取当前参数配置"""
        params = {
            'detector_type': 'cyclostationary' if self.detector_type.currentIndex() == 1 else 'energy',
            'num_samples': self.num_samples.value(),
            'noise_power': self.noise_power.value(),
            'auto_threshold': self.auto_threshold.isChecked(),
            'target_pf': self.target_pf.value(),
            'manual_threshold': self.manual_threshold.value(),
            'calibration_method': 'theoretical' if self.calibration_method.currentIndex() == 0 else 'simulation',
            'num_trials': self.num_trials.value(),
            'use_theoretical': self.use_theoretical.isChecked()
        }
        
        if params['detector_type'] == 'cyclostationary':
            params['alpha_resolution'] = self.alpha_resolution.value()
        
        return params
