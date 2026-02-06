"""
信号配置面板
配置信号生成参数
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
                             QFormLayout, QRadioButton, QButtonGroup)
from PyQt5.QtCore import pyqtSignal


class SignalPanel(QWidget):
    """信号配置面板"""
    
    # 参数变化信号
    parameters_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        
        # 信号类型选择
        type_group = QGroupBox("信号类型")
        type_layout = QHBoxLayout()
        
        self.type_button_group = QButtonGroup()
        self.single_carrier_radio = QRadioButton("单载波")
        self.ofdm_radio = QRadioButton("OFDM")
        self.single_carrier_radio.setChecked(True)
        
        self.type_button_group.addButton(self.single_carrier_radio, 0)
        self.type_button_group.addButton(self.ofdm_radio, 1)
        
        type_layout.addWidget(self.single_carrier_radio)
        type_layout.addWidget(self.ofdm_radio)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # 单载波参数组
        self.sc_group = QGroupBox("单载波参数")
        sc_layout = QFormLayout()
        
        self.sc_modulation = QComboBox()
        self.sc_modulation.addItems(['BPSK', 'QPSK', '8PSK', '16QAM', '64QAM'])
        self.sc_modulation.setCurrentText('QPSK')
        sc_layout.addRow("调制方式:", self.sc_modulation)
        
        self.sc_symbol_rate = QDoubleSpinBox()
        self.sc_symbol_rate.setRange(1, 1000000)
        self.sc_symbol_rate.setValue(1000)
        self.sc_symbol_rate.setSuffix(" Hz")
        sc_layout.addRow("符号率:", self.sc_symbol_rate)
        
        self.sc_num_symbols = QSpinBox()
        self.sc_num_symbols.setRange(100, 100000)
        self.sc_num_symbols.setValue(1000)
        sc_layout.addRow("符号数:", self.sc_num_symbols)
        
        self.sc_group.setLayout(sc_layout)
        layout.addWidget(self.sc_group)
        
        # OFDM参数组
        self.ofdm_group = QGroupBox("OFDM参数")
        ofdm_layout = QFormLayout()
        
        self.ofdm_modulation = QComboBox()
        self.ofdm_modulation.addItems(['BPSK', 'QPSK', '16QAM', '64QAM'])
        self.ofdm_modulation.setCurrentText('QPSK')
        ofdm_layout.addRow("子载波调制:", self.ofdm_modulation)
        
        self.ofdm_subcarriers = QComboBox()
        self.ofdm_subcarriers.addItems(['64', '128', '256', '512', '1024', '2048'])
        self.ofdm_subcarriers.setCurrentText('64')
        ofdm_layout.addRow("子载波数:", self.ofdm_subcarriers)
        
        self.ofdm_cp_length = QSpinBox()
        self.ofdm_cp_length.setRange(0, 512)
        self.ofdm_cp_length.setValue(16)
        ofdm_layout.addRow("CP长度:", self.ofdm_cp_length)
        
        self.ofdm_num_symbols = QSpinBox()
        self.ofdm_num_symbols.setRange(1, 1000)
        self.ofdm_num_symbols.setValue(100)
        ofdm_layout.addRow("OFDM符号数:", self.ofdm_num_symbols)
        
        self.ofdm_group.setLayout(ofdm_layout)
        layout.addWidget(self.ofdm_group)
        self.ofdm_group.setVisible(False)
        
        # SNR设置
        snr_group = QGroupBox("信噪比设置")
        snr_layout = QFormLayout()
        
        self.snr_min = QDoubleSpinBox()
        self.snr_min.setRange(-20, 30)
        self.snr_min.setValue(-10)
        self.snr_min.setSuffix(" dB")
        snr_layout.addRow("最小SNR:", self.snr_min)
        
        self.snr_max = QDoubleSpinBox()
        self.snr_max.setRange(-20, 30)
        self.snr_max.setValue(10)
        self.snr_max.setSuffix(" dB")
        snr_layout.addRow("最大SNR:", self.snr_max)
        
        self.snr_step = QDoubleSpinBox()
        self.snr_step.setRange(0.5, 10)
        self.snr_step.setValue(2)
        self.snr_step.setSuffix(" dB")
        snr_layout.addRow("SNR步进:", self.snr_step)
        
        snr_group.setLayout(snr_layout)
        layout.addWidget(snr_group)
        
        layout.addStretch()
        
        # 连接信号
        self.type_button_group.buttonClicked.connect(self._on_type_changed)
        
    def _on_type_changed(self, button):
        """信号类型切换"""
        is_ofdm = (button == self.ofdm_radio)
        self.sc_group.setVisible(not is_ofdm)
        self.ofdm_group.setVisible(is_ofdm)
        
    def get_parameters(self) -> dict:
        """获取当前参数配置"""
        params = {
            'signal_type': 'ofdm' if self.ofdm_radio.isChecked() else 'single_carrier',
            'snr_min': self.snr_min.value(),
            'snr_max': self.snr_max.value(),
            'snr_step': self.snr_step.value()
        }
        
        if params['signal_type'] == 'single_carrier':
            params['modulation'] = self.sc_modulation.currentText()
            params['symbol_rate'] = self.sc_symbol_rate.value()
            params['num_symbols'] = self.sc_num_symbols.value()
        else:
            params['modulation'] = self.ofdm_modulation.currentText()
            params['num_subcarriers'] = int(self.ofdm_subcarriers.currentText())
            params['cp_length'] = self.ofdm_cp_length.value()
            params['num_symbols'] = self.ofdm_num_symbols.value()
        
        return params
