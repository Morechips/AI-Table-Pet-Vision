import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print("⏳ 正在加载数据集...")
try:
    # 读取你刚刚录制的数据
    df = pd.read_csv('gesture_dataset.csv', header=None)
except FileNotFoundError:
    print("❌ 找不到 gesture_dataset.csv，请先运行 collect_data.py！")
    exit()

# 我们有 63 个特征 (21个点 * 3个坐标(XYZ))
feature_names = [f'v{i}' for i in range(1, 64)]
# 最后一列是刚才手敲的标签
df.columns = feature_names + ['label']

# X 是特征（动作的坐标），y 是答案（按下的01234）
X = df.drop('label', axis=1)
y = df['label']

print(f"📊 数据总量: {len(df)} 条")

# 划分训练集(80%) 和 测试集(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"🧠 训练集: {len(X_train)} 条 | 📝 测试集: {len(X_test)} 条")
print("🔥 模型开始训练 (炼丹中)...")

# 使用随机森林分类器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 测试准确率
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print("========================================")
print(f"🎯 模型测试集准确率: {accuracy:.2f}%")
print("========================================")

# 保存模型
with open('gesture_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("💾 炼丹成功！多分类大脑模型已保存为: gesture_model.pkl")
print("🚀 随时可以运行 multi_gesture_brain.py 进行实机演示了！")