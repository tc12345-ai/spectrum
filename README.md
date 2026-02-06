# Spectrum Sensing Simulator

这是一个用于认知无线电与频谱感知的仿真工具，提供单载波/OFDM 信号生成、能量检测与循环平稳检测、门限自动校准以及 ROC 曲线与检测性能分析。

## 功能概览

- 单载波与 OFDM 信号生成
- 能量检测、循环平稳检测
- 给定目标虚警概率的自动门限校准
- ROC 曲线与 Pd vs SNR 分析
- 门限建议与性能评估

## 目录结构

- `main.py`：程序入口
- `gui/`：PyQt5 图形界面
- `signals/`：信号生成器（单载波、OFDM）
- `detectors/`：检测器实现（能量检测、循环平稳检测）
- `analysis/`：ROC 与门限分析工具
- `utils/`：噪声/信道工具

## 运行方式

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 启动应用：

```bash
python main.py
```

## 说明

- 默认使用 PyQt5 的 Fusion 风格界面。
- 仿真参数可在左侧面板配置，结果展示在右侧。
