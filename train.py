import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 真实标签和预测标签
true_labels = [2, 0, 2, 2, 0, 1]
predicted_labels = [0, 0, 2, 2, 0, 2]

# 计算混淆矩阵
cm = confusion_matrix(true_labels, predicted_labels)

# 设置类别名称
classes = ['Class 1', 'Class 2', 'Class 3']

# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()