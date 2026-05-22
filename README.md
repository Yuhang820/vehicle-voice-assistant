# 车载语音智能助手

面向智能座舱场景的端到端语音意图识别系统，支持意图分类、槽位提取与自然语言响应，覆盖模型训练、API 服务部署及前端交互演示。

---

## 项目功能

- 输入车载语音文本（如"导航去北京南站"、"把空调调到26度"）
- 识别用户意图类别
- 提取结构化槽位信息（目的地、温度值、歌曲名等）
- 调用对应工具函数执行指令，返回自然语言响应

---

## 支持的意图类别（15类）

| 意图 | 说明 |
|------|------|
| navigate_to | 导航 |
| set_temperature | 空调温度 |
| play_music | 播放音乐 |
| control_window | 车窗控制 |
| control_seat | 座椅控制 |
| control_light | 车灯控制 |
| control_wiper | 雨刷控制 |
| get_weather | 查询天气 |
| get_fuel_level | 查询油量 |
| get_speed | 查询车速 |
| set_volume | 音量调节 |
| next_song | 切歌 |
| set_radio | 收音机 |
| search_car_manual | 查询车辆手册 |
| get_agent_name | 查询助手名称 |

---

## 技术栈

| 模块 | 技术 |
|------|------|
| 意图识别（基线） | TF-IDF + Logistic Regression |
| 意图识别（主模型） | BERT 微调（bert-base-chinese） |
| 槽位提取 | 通义千问 API（Function Calling） |
| 后端服务 | FastAPI |
| 前端演示 | Gradio |
| 检索增强 | RAG |
| 深度学习框架 | PyTorch |

---

## 项目结构

```
├── data/
│   └── intent_data.csv       # 自建车载语音指令数据集（220条，训练集176 / 测试集44）
├── models/                   # 模型文件（不含权重，需本地训练生成）
├── train.py                  # TF-IDF + LR 训练脚本
├── train_bert.py             # BERT 微调训练脚本
├── predictor.py              # 推理模块（IntentPredictor / BertIntentPredictor）
├── agent.py                  # 千问 Function Calling 槽位提取与工具调用
├── main.py                   # FastAPI 后端入口
├── app.py                    # Gradio 前端
├── rag.py                    # RAG 检索增强模块
├── evaluate.py               # 模型评估脚本
└── requirements.txt
```

---

## 快速开始

**安装依赖**

```bash
pip install -r requirements.txt
```

**训练模型**

```bash
# TF-IDF + LR 基线
python train.py

# BERT 微调
python train_bert.py --data data/intent_data.csv
```

**启动后端**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**启动前端**

```bash
python app.py
```

访问 `http://localhost:7860` 即可使用。

---

## 模型对比

| 模型 | 测试集准确率 | 5折交叉验证 |
|------|------------|------------|
| TF-IDF + Logistic Regression | 86.4% | 84.1% ± 4.7% |
| BERT 微调（bert-base-chinese） | 92%+ | — |

---

## 数据集

自主构建，共 220 条，覆盖 15 类车载场景意图，含人工标注。
