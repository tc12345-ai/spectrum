"""
频谱检测与信号识别仿真软件

主程序入口
用于认知无线电和频谱感知的仿真工具

功能特点:
- 支持单载波和OFDM信号生成
- 能量检测和循环平稳检测
- 自动门限校准（给定Pf求门限）
- ROC曲线生成（Pd vs Pf）
- 检测性能分析

作者: Spectrum Sensing Simulator
版本: 1.0.0
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import main

if __name__ == '__main__':
    main()
