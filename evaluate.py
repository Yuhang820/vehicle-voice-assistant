import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

# 加载模型
model = joblib.load("models/intent_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# 加载数据
df = pd.read_csv("data/intent_data.csv")
X = vectorizer.transform(df["text"])
y_true = df["intent"]
y_pred = model.predict(X)

# 打印分类报告
print(classification_report(y_true, y_pred))

# 画混淆矩阵
labels = sorted(df["intent"].unique())
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
plt.title("意图识别混淆矩阵")
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix_eval.png")
plt.show()
print("混淆矩阵已保存至 outputs/confusion_matrix_eval.png")