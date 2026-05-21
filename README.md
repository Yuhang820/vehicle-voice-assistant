# 车载语音意图识别系统

面向智能座舱场景的自然语言意图分类模型，支持 12 类车载语音指令识别。

## 项目结构

```
car_voice_intent/
├── data/
│   └── intent_data.csv      # 240 条标注训练数据
├── models/                  # 训练后自动生成
│   ├── tfidf_vectorizer.pkl
│   └── intent_classifier.pkl
├── outputs/                 # 训练后自动生成
│   └── confusion_matrix.png
├── train.py                 # 训练 + 评估脚本（Week 1）
├── requirements.txt
└── README.md
```

## 支持的意图类别（12类）

| 意图 | 英文标签 | 示例 |
|------|----------|------|
| 导航 | navigation | "导航到合肥南站" |
| 空调控制 | ac_control | "把温度调到26度" |
| 音乐控制 | music_control | "播放周杰伦的歌" |
| 电话 | phone_call | "给妈妈打电话" |
| 车窗控制 | window_control | "打开左边车窗" |
| 信息查询 | query_info | "现在几点了" |
| 灯光控制 | light_control | "打开氛围灯" |
| 车门控制 | door_control | "打开后备箱" |
| 座椅控制 | seat_control | "开启座椅加热" |
| 应用控制 | app_control | "打开地图" |
| 驾驶模式 | driving_mode | "切换到运动模式" |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练模型
python train.py
```

## 技术方案

- **特征提取**：TF-IDF 字符级 n-gram（1~3字符），无需分词
- **分类模型**：Logistic Regression（多分类 softmax）
- **评估指标**：Accuracy、Precision、Recall、F1、混淆矩阵
- **模型保存**：joblib 序列化，供后续 API 服务加载

## 下一步（Week 2）

- [ ] FastAPI 封装 `/predict` 接口
- [ ] 接入 LLM API 做槽位提取
- [ ] Docker 打包部署
