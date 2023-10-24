# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:15:42 2023

@author: User
"""
#!pip install orange3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import Orange

# 下載UCI Adult資料集
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"]

data = pd.read_csv(data_url, names=column_names, sep=r'\s*,\s*', engine='python')

# 將資料轉為數值型態
data_dummies = pd.get_dummies(data)

# 將資料分成特徵和標籤
X = data_dummies.drop(columns=['income_<=50K', 'income_>50K'])
y = data['income']

# 分成訓練及測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用Orange3的TreeLearner
feature_vars = [Orange.data.ContinuousVariable.make(name) for name in X.columns]
class_var = Orange.data.DiscreteVariable("income", values=['<=50K', '>50K'])
domain = Orange.data.Domain(feature_vars, class_var)
train_data = Orange.data.Table(domain, X_train.values, y_train.replace({'<=50K': 0, '>50K': 1}).values)
tree_learner = Orange.classification.TreeLearner()
clf = tree_learner(train_data)

# 預測訓練資料和測試資料的結果
train_preds = clf(train_data)
test_data = Orange.data.Table(domain, X_test.values, y_test.replace({'<=50K': 0, '>50K': 1}).values)
test_preds = clf(test_data)

# 將預測結果轉為文字標籤
test_preds_labels = ['<=50K' if pred == 0 else '>50K' for pred in test_preds]

# 計算訓練和測試的分類正確率
train_accuracy = accuracy_score(y_train.replace({'<=50K': 0, '>50K': 1}).values, train_preds)
test_accuracy = accuracy_score(y_test.replace({'<=50K': 0, '>50K': 1}).values, test_preds)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# 將測試資料結果輸出到Excel檔案
output = pd.DataFrame({'Actual': y_test, 'Predicted': test_preds_labels})
output.to_excel('decision_tree_results.xlsx', index=False, engine='openpyxl')
#output.to_excel('C:\\Users\\User\\Documents\\test\\decision_tree_results.xlsx', index=False, engine='openpyxl')