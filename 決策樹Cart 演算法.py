# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:16:09 2023

@author: User
"""

# 引入所需的模組
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import openpyxl
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 下載UCI Adult資料集
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"]

data = pd.read_csv(data_url, names=column_names, sep=r'\s*,\s*', engine='python')

# 將資料分成特徵和標籤
X = data.drop("income", axis=1)
y = data["income"]

# 將資料轉為數值型態
X = pd.get_dummies(X)

# 分成訓練及測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用決策樹CART演算法訓練模型
clf = DecisionTreeClassifier(criterion='gini')
clf.fit(X_train, y_train)

# 預測訓練資料和測試資料的結果
train_preds = clf.predict(X_train)
test_preds = clf.predict(X_test)

# 計算訓練和測試的分類正確率
train_accuracy = accuracy_score(y_train, train_preds)
test_accuracy = accuracy_score(y_test, test_preds)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# 將測試資料結果輸出到Excel檔案
output = pd.DataFrame({'Actual': y_test, 'Predicted': test_preds})
output.to_excel('decision_tree_results.xlsx', index=False, engine='openpyxl')
#output.to_excel('C:\\Users\\User\\Documents\\test\\decision_tree_results.xlsx', index=False, engine='openpyxl')

# 定義三組決策樹參數（節點數及樹深度）
params = [
    {"max_depth": 5, "max_leaf_nodes": 20},
    {"max_depth": 10, "max_leaf_nodes": 50},
    {"max_depth": 15, "max_leaf_nodes": 100}
]

accuracies = []

# 根據不同的樹深度和節點數訓練三個決策樹模型並使用cost complexity pruning進行修剪
for param in params:
    clf = DecisionTreeClassifier(criterion='gini', **param)
    clf.fit(X_train, y_train)
    
    # 使用 cost_complexity_pruning_path 得到最佳的 alpha 值
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # 根據不同的alpha值訓練不同的模型
    clfs = [DecisionTreeClassifier(criterion='gini', ccp_alpha=ccp_alpha, **param) 
            for ccp_alpha in ccp_alphas]
    for clf_alpha in clfs:
        clf_alpha.fit(X_train, y_train)
    
    # 從修剪的模型中選擇最佳的
    test_scores = [clf_alpha.score(X_test, y_test) for clf_alpha in clfs]
    best_idx = test_scores.index(max(test_scores))
    best_clf = clfs[best_idx]
    
    # 繪製該決策樹
    plt.figure(figsize=(20, 10))
    plot_tree(best_clf, filled=True, feature_names=X.columns, class_names=best_clf.classes_, proportion=True)
    plt.title(f"Decision Tree with max_depth={param['max_depth']} and max_leaf_nodes={param['max_leaf_nodes']} post-pruned")
    plt.show()
    
    y_pred = best_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# 比較三種參數設定下的分類預測正確率
for idx, acc in enumerate(accuracies):
    print(f"Tree with max_depth={params[idx]['max_depth']} and max_leaf_nodes={params[idx]['max_leaf_nodes']} post-pruned has testing accuracy: {acc:.4f}")
