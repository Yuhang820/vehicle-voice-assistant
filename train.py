import pandas as pd
import numpy as np
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ── 1. 加载数据 ────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "data/intent_data.csv")

df = pd.read_csv(DATA_PATH)
print(f"数据集大小: {len(df)} 条")
print(f"意图类别数: {df['intent'].nunique()} 类")
print(f"\n各类别样本数:\n{df['intent'].value_counts()}\n")

X = df["text"]
y = df["intent"]

INTENT_LABELS = {
    "navigation":    "导航",
    "ac_control":    "空调控制",
    "music_control": "音乐控制",
    "phone_call":    "电话",
    "window_control":"车窗控制",
    "query_info":    "信息查询",
    "light_control": "灯光控制",
    "door_control":  "车门控制",
    "seat_control":  "座椅控制",
    "app_control":   "应用控制",
    "driving_mode":  "驾驶模式",
}


# ── 2. 划分训练集 / 测试集 ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集: {len(X_train)} 条  测试集: {len(X_test)} 条\n")


# ── 3. 特征提取：TF-IDF 字符级 n-gram ─────────────────────
#   中文不需要分词，用字符级别 (analyzer='char') 效果更好
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 3),   # 1-3 个字符的 n-gram
    max_features=5000,
    sublinear_tf=True     # 对 TF 取对数，缓解高频词影响
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"特征维度: {X_train_vec.shape[1]}")


# ── 4. 训练模型 ────────────────────────────────────────────
clf = LogisticRegression(
    max_iter=1000,
    C=5.0,           # 正则化强度，值越大正则越弱
    solver="lbfgs",
)
clf.fit(X_train_vec, y_train)


# ── 5. 评估 ────────────────────────────────────────────────
y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {acc:.4f}  ({acc*100:.1f}%)\n")

# 交叉验证（5折）
cv_scores = cross_val_score(clf, X_train_vec, y_train, cv=5, scoring="accuracy")
print(f"5折交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

print("=" * 60)
print("分类报告（每个意图类别的 precision / recall / f1）")
print("=" * 60)
print(classification_report(y_test, y_pred))


# ── 6. 混淆矩阵可视化 ──────────────────────────────────────
labels = clf.classes_
cn_labels = [INTENT_LABELS.get(l, l) for l in labels]

cm = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(cm, cmap="Blues")
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(cn_labels)))
ax.set_yticks(range(len(cn_labels)))
ax.set_xticklabels(cn_labels, rotation=45, ha="right", fontsize=10)
ax.set_yticklabels(cn_labels, fontsize=10)
ax.set_xlabel("预测标签", fontsize=12)
ax.set_ylabel("真实标签", fontsize=12)
ax.set_title(f"混淆矩阵（测试集准确率 {acc*100:.1f}%）", fontsize=13)

for i in range(len(labels)):
    for j in range(len(labels)):
        val = cm[i, j]
        color = "white" if val > cm.max() / 2 else "black"
        ax.text(j, i, str(val), ha="center", va="center",
                color=color, fontsize=9)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix.png", dpi=150)
print("\n混淆矩阵已保存到 outputs/confusion_matrix.png")


# ── 7. 保存模型 ────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(clf,        "models/intent_classifier.pkl")
print("模型已保存到 models/ 目录")


# ── 8. 快速预测函数（验证模型可用）────────────────────────
def predict_intent(text: str) -> dict:
    vec = vectorizer.transform([text])
    intent = clf.predict(vec)[0]
    proba = clf.predict_proba(vec)[0]
    confidence = proba.max()
    return {
        "text": text,
        "intent": intent,
        "intent_cn": INTENT_LABELS.get(intent, intent),
        "confidence": round(float(confidence), 4)
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("快速测试几条样本：")
    print("=" * 60)
    test_samples = [
        "导航到合肥南站",
        "把空调调到25度",
        "播放一首歌",
        "给妈妈打电话",
        "把车窗关上",
        "现在几点了",
        "开启运动模式",
        "锁车",
    ]
    for s in test_samples:
        result = predict_intent(s)
        print(f"  输入: 「{result['text']}」")
        print(f"  意图: {result['intent_cn']}  置信度: {result['confidence']:.2%}\n")