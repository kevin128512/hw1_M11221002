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
clf = DecisionTreeClassifier(criterion='entropy')
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


